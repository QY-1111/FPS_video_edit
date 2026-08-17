"""ComfyUI node for merging consecutive similar frames and keeping the sharpest one."""

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn.functional as F


def _as_rgb(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images 必须是 ComfyUI IMAGE 批次，形状为 [B,H,W,C]。")
    if images.shape[0] == 0:
        raise ValueError("输入图片批次为空。")
    if images.shape[-1] < 3:
        return images[..., :1].repeat(1, 1, 1, 3)
    return images[..., :3]


def _prepend_cover_frame(
    images: torch.Tensor,
    cover_frame: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    """Prepend one optional cover frame, resizing it to match the image batch."""
    images = _as_rgb(images)
    if cover_frame is None:
        return images, False

    cover = _as_rgb(cover_frame)[:1]
    target_height, target_width = images.shape[1:3]
    cover = cover.to(device=images.device, dtype=images.dtype)
    if cover.shape[1:3] != (target_height, target_width):
        original_dtype = cover.dtype
        cover = F.interpolate(
            cover.permute(0, 3, 1, 2).float(),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).to(dtype=original_dtype)

    return torch.cat((cover, images), dim=0), True


def _analysis_gray(images: torch.Tensor, resolution: int) -> torch.Tensor:
    rgb = _as_rgb(images).detach().float().clamp(0.0, 1.0)
    gray = (
        rgb[..., 0] * 0.299
        + rgb[..., 1] * 0.587
        + rgb[..., 2] * 0.114
    ).unsqueeze(1)
    height, width = gray.shape[-2:]
    longest = max(height, width)
    if longest > resolution:
        scale = resolution / float(longest)
        size = (max(16, round(height * scale)), max(16, round(width * scale)))
        gray = F.interpolate(gray, size=size, mode="area")
    return gray


def _crop_region(gray: torch.Tensor, region: str) -> torch.Tensor:
    height, width = gray.shape[-2:]
    if region == "中心80%":
        top, bottom = round(height * 0.10), round(height * 0.90)
        left, right = round(width * 0.10), round(width * 0.90)
    elif region == "中心60%":
        top, bottom = round(height * 0.20), round(height * 0.80)
        left, right = round(width * 0.20), round(width * 0.80)
    elif region == "上部80%(忽略底部字幕)":
        top, bottom, left, right = 0, round(height * 0.80), 0, width
    elif region == "底部40%":
        top, bottom, left, right = round(height * 0.60), height, 0, width
    else:
        return gray
    return gray[..., top:max(top + 1, bottom), left:max(left + 1, right)]


def _ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    # Local SSIM without an external OpenCV/scikit-image dependency.
    kernel = min(7, first.shape[-2], first.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    padding = kernel // 2
    mu_x = F.avg_pool2d(first, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(second, kernel, stride=1, padding=padding)
    var_x = (F.avg_pool2d(first * first, kernel, 1, padding) - mu_x * mu_x).clamp_min(0)
    var_y = (F.avg_pool2d(second * second, kernel, 1, padding) - mu_y * mu_y).clamp_min(0)
    covariance = F.avg_pool2d(first * second, kernel, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(score.mean().clamp(-1.0, 1.0).item())


def _dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    cols = torch.arange(size, device=device, dtype=dtype).unsqueeze(0)
    matrix = torch.cos(math.pi / size * (cols + 0.5) * rows)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def _phash(gray: torch.Tensor, hash_size: int = 8) -> torch.Tensor:
    size = hash_size * 4
    small = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    matrix = _dct_matrix(size, small.device, small.dtype)
    coefficients = matrix @ small @ matrix.t()
    low = coefficients[:hash_size, :hash_size].flatten()
    useful = low[1:]
    return useful > useful.median()


def _changed_area(first: torch.Tensor, second: torch.Tensor, threshold: float) -> float:
    # Light blur suppresses compression noise while retaining meaningful local changes.
    first = F.avg_pool2d(first, 3, stride=1, padding=1)
    second = F.avg_pool2d(second, 3, stride=1, padding=1)
    return float(((first - second).abs() >= threshold).float().mean().item())


def _pair_metrics(
    first: torch.Tensor,
    second: torch.Tensor,
    pixel_change_threshold: float,
) -> dict[str, float | int]:
    hash_a, hash_b = _phash(first), _phash(second)
    return {
        "ssim": _ssim(first, second),
        "phash_distance": int(torch.count_nonzero(hash_a != hash_b).item()),
        "changed_area": _changed_area(first, second, pixel_change_threshold),
    }


def _is_similar(
    metrics: dict[str, float | int],
    rule: str,
    ssim_threshold: float,
    phash_distance_threshold: int,
    changed_area_threshold: float,
) -> bool:
    ssim_ok = float(metrics["ssim"]) >= ssim_threshold
    hash_ok = int(metrics["phash_distance"]) <= phash_distance_threshold
    area_ok = float(metrics["changed_area"]) <= changed_area_threshold
    if rule == "仅SSIM":
        return ssim_ok
    if rule == "仅pHash":
        return hash_ok
    if rule == "宽松混合(满足任意两项)":
        return sum((ssim_ok, hash_ok, area_ok)) >= 2
    return ssim_ok and hash_ok and area_ok


def _sharpness_scores(gray: torch.Tensor) -> tuple[list[float], list[float]]:
    dtype, device = gray.dtype, gray.device
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    laplacian = lap.var(dim=(1, 2, 3), unbiased=False)
    tenengrad = (grad_x.square() + grad_y.square()).mean(dim=(1, 2, 3))
    return laplacian.cpu().tolist(), tenengrad.cpu().tolist()


def _normalise(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _choose_representative(
    indices: list[int],
    laplacian: list[float],
    tenengrad: list[float],
    laplacian_weight: float,
    tenengrad_weight: float,
    tie_breaker: str,
) -> tuple[int, dict[int, float]]:
    lap_norm = _normalise([laplacian[index] for index in indices])
    ten_norm = _normalise([tenengrad[index] for index in indices])
    weight_sum = max(laplacian_weight + tenengrad_weight, 1e-9)
    combined = {
        index: (laplacian_weight * lap_norm[pos] + tenengrad_weight * ten_norm[pos]) / weight_sum
        for pos, index in enumerate(indices)
    }
    best_score = max(combined.values())
    candidates = [index for index in indices if abs(combined[index] - best_score) <= 1e-9]
    if tie_breaker == "靠后帧":
        selected = candidates[-1]
    elif tie_breaker == "中间帧":
        centre = (indices[0] + indices[-1]) / 2
        selected = min(candidates, key=lambda index: (abs(index - centre), index))
    else:
        selected = candidates[0]
    return selected, combined


class SimilarFrameBestSelector:
    """Group adjacent, visually similar images and retain the sharpest image per group."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "按视频时间顺序排列的抽帧图片批次。"}),
                "判定核心": (
                    ["严格混合(三项都满足)", "宽松混合(满足任意两项)", "仅SSIM", "仅pHash"],
                    {"default": "严格混合(三项都满足)"},
                ),
                "SSIM阈值": ("FLOAT", {"default": 0.94, "min": 0.0, "max": 1.0, "step": 0.005}),
                "pHash距离阈值": ("INT", {"default": 8, "min": 0, "max": 63, "step": 1}),
                "变化面积阈值": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005}),
                "单像素变化阈值": ("FLOAT", {"default": 0.06, "min": 0.005, "max": 1.0, "step": 0.005}),
                "连续帧比较方式": (
                    ["同时比较上一帧和组首帧", "只比较组首帧", "只比较上一帧"],
                    {"default": "同时比较上一帧和组首帧"},
                ),
                "相似度检测区域": (
                    ["全图", "中心80%", "中心60%", "上部80%(忽略底部字幕)", "底部40%"],
                    {"default": "全图"},
                ),
                "清晰度检测区域": (
                    ["全图", "中心80%", "中心60%", "上部80%(忽略底部字幕)", "底部40%"],
                    {"default": "全图"},
                ),
                "Tenengrad权重": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "Laplacian权重": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "同分选择": (["中间帧", "靠前帧", "靠后帧"], {"default": "中间帧"}),
                "分析分辨率": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
            "optional": {
                "封面帧": (
                    "IMAGE",
                    {"tooltip": "可选。接入后会作为第 0 帧放在图像集前面，共同参与相似度合并和清晰度选择。"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("筛选后图片", "保留索引JSON", "分组与评分报告", "分组数量")
    FUNCTION = "merge"
    CATEGORY = "视频/关键帧"
    DESCRIPTION = "将连续的相似抽帧合并成组，并从每组中保留清晰度最高的一帧。"

    def merge(
        self,
        images: torch.Tensor,
        判定核心: str,
        SSIM阈值: float,
        pHash距离阈值: int,
        变化面积阈值: float,
        单像素变化阈值: float,
        连续帧比较方式: str,
        相似度检测区域: str,
        清晰度检测区域: str,
        Tenengrad权重: float,
        Laplacian权重: float,
        同分选择: str,
        分析分辨率: int,
        封面帧: torch.Tensor | None = None,
    ):
        image_set_count = int(images.shape[0]) if isinstance(images, torch.Tensor) and images.ndim == 4 else 0
        images, has_cover_frame = _prepend_cover_frame(images, 封面帧)
        gray = _analysis_gray(images, 分析分辨率)
        similarity_gray = _crop_region(gray, 相似度检测区域)
        sharpness_gray = _crop_region(gray, 清晰度检测区域)

        groups: list[list[int]] = [[0]]
        comparisons: list[dict[str, Any]] = []
        for current in range(1, images.shape[0]):
            previous = current - 1
            anchor = groups[-1][0]
            targets: list[tuple[str, int]] = []
            if 连续帧比较方式 in ("同时比较上一帧和组首帧", "只比较上一帧"):
                targets.append(("previous", previous))
            if 连续帧比较方式 in ("同时比较上一帧和组首帧", "只比较组首帧") and anchor != previous:
                targets.append(("anchor", anchor))
            if not targets:
                targets.append(("anchor", anchor))

            pair_results: list[dict[str, Any]] = []
            similar = True
            for role, target in targets:
                metrics = _pair_metrics(
                    similarity_gray[target : target + 1],
                    similarity_gray[current : current + 1],
                    单像素变化阈值,
                )
                accepted = _is_similar(
                    metrics,
                    判定核心,
                    SSIM阈值,
                    pHash距离阈值,
                    变化面积阈值,
                )
                pair_results.append({"与": role, "目标索引": target, **metrics, "相似": accepted})
                similar = similar and accepted

            comparisons.append({"当前索引": current, "比较": pair_results, "归入当前组": similar})
            if similar:
                groups[-1].append(current)
            else:
                groups.append([current])

        laplacian, tenengrad = _sharpness_scores(sharpness_gray)
        selected: list[int] = []
        group_reports: list[dict[str, Any]] = []
        for group_number, indices in enumerate(groups, start=1):
            representative, combined = _choose_representative(
                indices,
                laplacian,
                tenengrad,
                Laplacian权重,
                Tenengrad权重,
                同分选择,
            )
            selected.append(representative)
            group_reports.append(
                {
                    "组": group_number,
                    "成员索引": indices,
                    "保留索引": representative,
                    "评分": [
                        {
                            "索引": index,
                            "Tenengrad": round(tenengrad[index], 8),
                            "Laplacian": round(laplacian[index], 8),
                            "组内综合分": round(combined[index], 6),
                        }
                        for index in indices
                    ],
                }
            )

        report = {
            "输入数量": int(images.shape[0]),
            "图像集数量": image_set_count,
            "封面帧已接入": has_cover_frame,
            "索引说明": (
                "封面帧为索引 0，原图像集从索引 1 开始"
                if has_cover_frame
                else "索引对应原图像集，从 0 开始"
            ),
            "输出数量": len(selected),
            "保留索引": selected,
            "参数": {
                "判定核心": 判定核心,
                "SSIM阈值": SSIM阈值,
                "pHash距离阈值": pHash距离阈值,
                "变化面积阈值": 变化面积阈值,
                "单像素变化阈值": 单像素变化阈值,
                "连续帧比较方式": 连续帧比较方式,
                "相似度检测区域": 相似度检测区域,
                "清晰度检测区域": 清晰度检测区域,
                "Tenengrad权重": Tenengrad权重,
                "Laplacian权重": Laplacian权重,
            },
            "分组": group_reports,
            "相邻判定详情": comparisons,
        }
        return (
            images[selected],
            json.dumps(selected, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False, indent=2),
            len(groups),
        )


NODE_CLASS_MAPPINGS = {
    "SimilarFrameBestSelector": SimilarFrameBestSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimilarFrameBestSelector": "相似帧合并并保留最清晰帧",
}
