"""ComfyUI node for filtering empty, smeared, and unusable video frames.

The implementation intentionally has no model download or third-party runtime
dependency.  It uses image structure, brightness and directional edge metrics;
an optional person/subject MASK can be supplied by an upstream detector to make
the decision semantic instead of purely heuristic.
"""

from __future__ import annotations

import json
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


def _analysis_rgb(images: torch.Tensor, resolution: int) -> torch.Tensor:
    rgb = _as_rgb(images).detach().float().clamp(0.0, 1.0).permute(0, 3, 1, 2)
    height, width = rgb.shape[-2:]
    longest = max(height, width)
    if longest > resolution:
        scale = resolution / float(longest)
        size = (max(32, round(height * scale)), max(32, round(width * scale)))
        rgb = F.interpolate(rgb, size=size, mode="area")
    return rgb


def _prepare_masks(
    masks: torch.Tensor | None,
    batch_size: int,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if masks is None:
        return None
    if not isinstance(masks, torch.Tensor):
        raise ValueError("主体遮罩必须是 ComfyUI MASK。")
    mask = masks.detach().float().to(device=device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        # Accept both [B,1,H,W] and the occasional [B,H,W,1].
        if mask.shape[1] != 1 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        mask = mask[:, :1]
    else:
        raise ValueError("主体遮罩形状必须为 [B,H,W] 或 [B,1,H,W]。")

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.repeat(batch_size, 1, 1, 1)
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"主体遮罩数量({mask.shape[0]})与图片数量({batch_size})不一致。"
        )
    if tuple(mask.shape[-2:]) != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _edge_metrics(gray: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype, device = gray.dtype, gray.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # Ignore the artificial border created by convolution padding.
    if gray.shape[-2] > 4 and gray.shape[-1] > 4:
        grad_x = grad_x[..., 1:-1, 1:-1]
        grad_y = grad_y[..., 1:-1, 1:-1]
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    energy_x = grad_x.abs().mean(dim=(1, 2, 3))
    energy_y = grad_y.abs().mean(dim=(1, 2, 3))

    # Horizontal light trails have predominantly vertical gradients (Gy).
    horizontal_directionality = (
        (energy_y - energy_x) / (energy_y + energy_x + 1e-8)
    ).clamp(min=0.0, max=1.0)

    edge_density = (magnitude >= 0.22).float().mean(dim=(1, 2, 3))
    edge_strength = magnitude.mean(dim=(1, 2, 3))

    height, width = magnitude.shape[-2:]
    top, bottom = round(height * 0.12), round(height * 0.88)
    left, right = round(width * 0.12), round(width * 0.88)
    centre = magnitude[..., top:max(top + 1, bottom), left:max(left + 1, right)]
    centre_density = (centre >= 0.22).float().mean(dim=(1, 2, 3))

    # Measure whether strong edges repeatedly span a large part of a row.
    row_span = (magnitude >= 0.22).float().mean(dim=3).squeeze(1)
    wide_row_ratio = (row_span >= 0.28).float().mean(dim=1)

    return {
        "edge_density": edge_density,
        "edge_strength": edge_strength,
        "centre_density": centre_density,
        "horizontal_directionality": horizontal_directionality,
        "wide_row_ratio": wide_row_ratio,
    }


def _frame_metrics(
    images: torch.Tensor,
    analysis_resolution: int,
    subject_masks: torch.Tensor | None,
) -> list[dict[str, float]]:
    rgb = _analysis_rgb(images, analysis_resolution)
    gray = (
        rgb[:, 0:1] * 0.299
        + rgb[:, 1:2] * 0.587
        + rgb[:, 2:3] * 0.114
    )
    edges = _edge_metrics(gray)
    masks = _prepare_masks(
        subject_masks,
        batch_size=images.shape[0],
        size=tuple(gray.shape[-2:]),
        device=gray.device,
    )

    mean_luma = gray.mean(dim=(1, 2, 3))
    luma_std = gray.std(dim=(1, 2, 3), unbiased=False)
    black_ratio = (gray <= 0.035).float().mean(dim=(1, 2, 3))
    white_ratio = (gray >= 0.965).float().mean(dim=(1, 2, 3))
    saturation = (rgb.max(dim=1).values - rgb.min(dim=1).values).mean(dim=(1, 2))
    mask_ratio = (
        masks.mean(dim=(1, 2, 3)) if masks is not None else torch.full_like(mean_luma, -1.0)
    )

    result: list[dict[str, float]] = []
    for index in range(images.shape[0]):
        result.append(
            {
                "mean_luma": float(mean_luma[index].item()),
                "luma_std": float(luma_std[index].item()),
                "black_ratio": float(black_ratio[index].item()),
                "white_ratio": float(white_ratio[index].item()),
                "saturation": float(saturation[index].item()),
                "edge_density": float(edges["edge_density"][index].item()),
                "edge_strength": float(edges["edge_strength"][index].item()),
                "centre_density": float(edges["centre_density"][index].item()),
                "horizontal_directionality": float(
                    edges["horizontal_directionality"][index].item()
                ),
                "wide_row_ratio": float(edges["wide_row_ratio"][index].item()),
                "subject_mask_ratio": float(mask_ratio[index].item()),
            }
        )
    return result


_STRENGTH_PRESETS: dict[str, dict[str, float]] = {
    "保守": {
        "direction_scale": 1.10,
        "detail_scale": 0.72,
        "wide_row_scale": 1.25,
    },
    "均衡(推荐)": {
        "direction_scale": 1.00,
        "detail_scale": 1.00,
        "wide_row_scale": 1.00,
    },
    "强力": {
        "direction_scale": 0.88,
        "detail_scale": 1.32,
        "wide_row_scale": 0.78,
    },
}


def _classify(
    metrics: dict[str, float],
    strength: str,
    filter_solid: bool,
    filter_light_streak: bool,
    filter_blur: bool,
    dark_threshold: float,
    bright_threshold: float,
    solid_std_threshold: float,
    directionality_threshold: float,
    minimum_detail_density: float,
    minimum_subject_ratio: float,
) -> tuple[bool, list[str], dict[str, bool]]:
    preset = _STRENGTH_PRESETS[strength]
    direction_limit = min(0.95, directionality_threshold * preset["direction_scale"])
    detail_limit = min(0.50, minimum_detail_density * preset["detail_scale"])
    wide_row_limit = min(0.80, 0.10 * preset["wide_row_scale"])

    has_external_mask = metrics["subject_mask_ratio"] >= 0.0
    subject_present = (
        has_external_mask and metrics["subject_mask_ratio"] >= minimum_subject_ratio
    )

    solid = (
        metrics["luma_std"] <= solid_std_threshold
        or metrics["black_ratio"] >= 0.985
        or metrics["white_ratio"] >= 0.985
    ) and (
        metrics["mean_luma"] <= dark_threshold
        or metrics["mean_luma"] >= bright_threshold
        or metrics["edge_density"] <= detail_limit * 0.35
    )

    # A transition/light-smear frame is usually dark, directionally dominated,
    # contains wide horizontal bands, and lacks useful central structure.
    light_streak = (
        metrics["horizontal_directionality"] >= direction_limit
        and metrics["wide_row_ratio"] >= wide_row_limit
        and metrics["centre_density"] <= detail_limit
        and metrics["mean_luma"] <= 0.52
        and not subject_present
    )

    severe_blur = (
        metrics["centre_density"] <= detail_limit * 0.48
        and metrics["edge_strength"] <= 0.13
        and metrics["luma_std"] <= 0.20
        and not subject_present
    )

    decisions = {
        "纯黑白或纯色": bool(filter_solid and solid),
        "灯光拖影或甩镜": bool(filter_light_streak and light_streak),
        "严重模糊且缺少结构": bool(filter_blur and severe_blur),
        "检测到主体遮罩": bool(subject_present),
    }
    reasons = [name for name, matched in decisions.items() if matched and name != "检测到主体遮罩"]
    return bool(reasons), reasons, decisions


class MeaninglessFrameFilter:
    """Filter unusable frames while exposing both results and diagnostics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "按视频时间顺序排列的抽帧图片批次。"},
                ),
                "过滤强度": (
                    ["保守", "均衡(推荐)", "强力"],
                    {"default": "均衡(推荐)"},
                ),
                "过滤纯黑白帧": ("BOOLEAN", {"default": True}),
                "过滤灯光拖影": ("BOOLEAN", {"default": True}),
                "过滤严重模糊": ("BOOLEAN", {"default": True}),
                "暗场亮度阈值": (
                    "FLOAT",
                    {"default": 0.07, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "亮场亮度阈值": (
                    "FLOAT",
                    {"default": 0.93, "min": 0.5, "max": 1.0, "step": 0.005},
                ),
                "纯色标准差阈值": (
                    "FLOAT",
                    {"default": 0.018, "min": 0.0, "max": 0.2, "step": 0.002},
                ),
                "横向拖影阈值": (
                    "FLOAT",
                    {"default": 0.34, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "最低主体细节密度": (
                    "FLOAT",
                    {"default": 0.28, "min": 0.005, "max": 0.5, "step": 0.005},
                ),
                "主体遮罩最小占比": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "分析分辨率": (
                    "INT",
                    {"default": 384, "min": 128, "max": 1024, "step": 64},
                ),
            },
            "optional": {
                "主体遮罩": (
                    "MASK",
                    {
                        "tooltip": (
                            "可连接 YOLO、Florence 等人物/主体检测节点输出的 MASK。"
                            "有遮罩的帧会优先保护；不连接也能使用图像启发式过滤。"
                        )
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "有效图片",
        "无意义图片",
        "保留索引JSON",
        "过滤索引JSON",
        "逐帧分析报告",
        "过滤数量",
    )
    FUNCTION = "filter_frames"
    CATEGORY = "视频/关键帧"
    DESCRIPTION = "过滤纯黑白、严重模糊以及无人主体的横向灯光拖影/甩镜过渡帧。"

    def filter_frames(
        self,
        images: torch.Tensor,
        过滤强度: str,
        过滤纯黑白帧: bool,
        过滤灯光拖影: bool,
        过滤严重模糊: bool,
        暗场亮度阈值: float,
        亮场亮度阈值: float,
        纯色标准差阈值: float,
        横向拖影阈值: float,
        最低主体细节密度: float,
        主体遮罩最小占比: float,
        分析分辨率: int,
        主体遮罩: torch.Tensor | None = None,
    ):
        images = _as_rgb(images)
        metrics = _frame_metrics(images, 分析分辨率, 主体遮罩)
        kept: list[int] = []
        rejected: list[int] = []
        frames: list[dict[str, Any]] = []

        for index, frame_metrics in enumerate(metrics):
            reject, reasons, decisions = _classify(
                frame_metrics,
                strength=过滤强度,
                filter_solid=过滤纯黑白帧,
                filter_light_streak=过滤灯光拖影,
                filter_blur=过滤严重模糊,
                dark_threshold=暗场亮度阈值,
                bright_threshold=亮场亮度阈值,
                solid_std_threshold=纯色标准差阈值,
                directionality_threshold=横向拖影阈值,
                minimum_detail_density=最低主体细节密度,
                minimum_subject_ratio=主体遮罩最小占比,
            )
            (rejected if reject else kept).append(index)
            frames.append(
                {
                    "索引": index,
                    "结论": "过滤" if reject else "保留",
                    "原因": reasons,
                    "判定项": decisions,
                    "指标": {key: round(value, 6) for key, value in frame_metrics.items()},
                }
            )

        report = {
            "输入数量": int(images.shape[0]),
            "保留数量": len(kept),
            "过滤数量": len(rejected),
            "保留索引": kept,
            "过滤索引": rejected,
            "参数": {
                "过滤强度": 过滤强度,
                "过滤纯黑白帧": 过滤纯黑白帧,
                "过滤灯光拖影": 过滤灯光拖影,
                "过滤严重模糊": 过滤严重模糊,
                "暗场亮度阈值": 暗场亮度阈值,
                "亮场亮度阈值": 亮场亮度阈值,
                "纯色标准差阈值": 纯色标准差阈值,
                "横向拖影阈值": 横向拖影阈值,
                "最低主体细节密度": 最低主体细节密度,
                "主体遮罩最小占比": 主体遮罩最小占比,
                "分析分辨率": 分析分辨率,
                "已连接主体遮罩": 主体遮罩 is not None,
            },
            "逐帧": frames,
        }

        # Preserve [0,H,W,C] empty batches. This faithfully represents "none"
        # and lets callers use the count/index outputs without a fake image.
        valid_images = images[kept]
        meaningless_images = images[rejected]
        return (
            valid_images,
            meaningless_images,
            json.dumps(kept, ensure_ascii=False),
            json.dumps(rejected, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False, indent=2),
            len(rejected),
        )


NODE_CLASS_MAPPINGS = {
    "MeaninglessFrameFilter": MeaninglessFrameFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeaninglessFrameFilter": "无意义帧过滤",
}
