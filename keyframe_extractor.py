"""ComfyUI node: extract video frames using timestamps from a model response."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
from typing import Any, Iterable

import torch


LOGGER = logging.getLogger(__name__)

_TIME_KEYS = {
    "关键帧时间",
    "关键帧时间点",
    "关键时间",
    "关键时间点",
    "关键帧",
    "抽帧时间",
    "keyframetime",
    "keyframetimes",
    "keyframes",
    "timestamps",
    "timestamp",
}


def _normalise_key(value: Any) -> str:
    return re.sub(r"[\s_\-:：]", "", str(value)).lower()


def _time_from_value(value: Any) -> float | None:
    """Convert a common timestamp representation to seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    if not isinstance(value, str):
        return None

    text = value.strip().strip('"\'`').replace("，", ",")
    if not text:
        return None

    # Plain numbers in a timestamp list are interpreted as seconds.
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    # HH:MM:SS(.mmm) and MM:SS(.mmm).
    colon_match = re.fullmatch(
        r"(?:(\d+):)?([0-5]?\d):([0-5]?\d(?:[\.,]\d+)?)", text
    )
    if colon_match:
        hours = int(colon_match.group(1) or 0)
        minutes = int(colon_match.group(2))
        seconds = float(colon_match.group(3).replace(",", "."))
        return hours * 3600 + minutes * 60 + seconds

    # 1h2m3.5s / 1小时2分3.5秒, with any leading fields omitted.
    unit_match = re.fullmatch(
        r"\s*(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours|小时))?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes|分|分钟))?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds|秒))\s*",
        text,
        flags=re.IGNORECASE,
    )
    if unit_match:
        return (
            float(unit_match.group(1) or 0) * 3600
            + float(unit_match.group(2) or 0) * 60
            + float(unit_match.group(3) or 0)
        )

    return None


def _times_from_sequence(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        value = [value]

    result: list[float] = []
    for item in value:
        if isinstance(item, dict):
            for key in ("time", "timestamp", "时间", "时间点"):
                if key in item:
                    item = item[key]
                    break
        seconds = _time_from_value(item)
        if seconds is not None:
            result.append(seconds)
    return result


def _walk_for_time_field(value: Any) -> list[float]:
    if isinstance(value, dict):
        # Prefer a timestamp field at the current level.
        for key, child in value.items():
            if _normalise_key(key) in _TIME_KEYS:
                times = _times_from_sequence(child)
                if times:
                    return times
        # Some models wrap their result in data/result/content.
        for child in value.values():
            times = _walk_for_time_field(child)
            if times:
                return times
    elif isinstance(value, list):
        # Accept a bare list only if it actually consists of timestamps.
        times = _times_from_sequence(value)
        if times and len(times) == len(value):
            return times
        for child in value:
            times = _walk_for_time_field(child)
            if times:
                return times
    return []


def _json_candidates(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"```(?:json|JSON)?\s*([\s\S]*?)```", text, flags=re.MULTILINE
        )
    )
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            yield value


def parse_keyframe_times(model_text: str) -> list[float]:
    """Extract an ordered, de-duplicated list of timestamps from model text."""
    if not isinstance(model_text, str) or not model_text.strip():
        raise ValueError("模型文本为空，无法取得关键帧时间。")

    times: list[float] = []
    for candidate in _json_candidates(model_text):
        times = _walk_for_time_field(candidate)
        if times:
            break

    # Tolerate JSON-like model output containing single quotes or trailing commas.
    if not times:
        key_pattern = (
            r"(?:关键帧时间点?|关键时间点?|抽帧时间|keyframe[_\s-]*times?|"
            r"timestamps?)\s*[\"']?\s*[:：]\s*\[([\s\S]*?)\]"
        )
        match = re.search(key_pattern, model_text, flags=re.IGNORECASE)
        if match:
            # Tokenise the array without losing compound values such as
            # "1分4秒". Quoted values may contain spaces or a decimal comma.
            token_matches = re.findall(
                r'"([^"]*)"|\'([^\']*)\'|([^,，\s]+)', match.group(1)
            )
            tokens = [next(part for part in parts if part) for parts in token_matches]
            times = [seconds for token in tokens if (seconds := _time_from_value(token)) is not None]

    # Final fallback: collect explicit clock-style values only. This avoids
    # accidentally treating unrelated numbers in prose as timestamps.
    if not times:
        tokens = re.findall(
            r"(?<!\d)\d+(?::[0-5]?\d){1,2}(?:[\.,]\d+)?(?!\d)", model_text
        )
        times = [seconds for token in tokens if (seconds := _time_from_value(token)) is not None]

    unique: list[float] = []
    for seconds in times:
        if math.isfinite(seconds) and seconds >= 0 and not any(
            abs(seconds - existing) < 0.001 for existing in unique
        ):
            unique.append(seconds)

    if not unique:
        raise ValueError(
            "未在模型文本中找到关键帧时间。请让模型输出例如："
            '{"关键帧时间":["00:00:00","00:00:05"]}'
        )
    return unique


def _video_source(video: Any) -> Any:
    """Obtain a path or file-like stream from ComfyUI's VIDEO object."""
    getter = getattr(video, "get_stream_source", None)
    if callable(getter):
        source = getter()
        if source is not None:
            return source

    if isinstance(video, (str, os.PathLike, io.IOBase)):
        return video

    if isinstance(video, dict):
        for key in ("video_path", "path", "filename", "file"):
            if video.get(key) is not None:
                return video[key]

    for attribute in ("_VideoFromFile__file", "video_path", "path", "filename", "file"):
        source = getattr(video, attribute, None)
        if source is not None:
            return source

    raise TypeError(
        "无法读取这个 VIDEO 对象。请连接 ComfyUI 原生 Load Video 节点的 VIDEO 输出。"
    )


def _private_number(video: Any, name: str, default: float = 0.0) -> float:
    value = getattr(video, name, default)
    try:
        result = float(value or 0)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _frame_seconds(frame: Any, time_base: float) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None:
        return float(frame.pts * time_base)
    return None


def _extract_frames(video: Any, relative_times: list[float]) -> torch.Tensor:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "当前 ComfyUI 环境缺少 PyAV（av）。请更新 ComfyUI，或在其 Python 环境运行 pip install av。"
        ) from exc

    source = _video_source(video)
    if hasattr(source, "seek"):
        source.seek(0)

    clip_start = _private_number(video, "_VideoFromFile__start_time")
    clip_duration = _private_number(video, "_VideoFromFile__duration")

    images: list[torch.Tensor] = []
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("输入文件中没有视频轨道。")
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        if time_base <= 0:
            raise ValueError("视频时间基信息无效，无法按时间抽帧。")

        source_duration = 0.0
        if stream.duration is not None:
            source_duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            source_duration = float(container.duration / av.time_base)

        available_duration = clip_duration
        if available_duration <= 0 and source_duration > clip_start:
            available_duration = source_duration - clip_start

        for requested_time in relative_times:
            relative_time = max(0.0, requested_time)
            if available_duration > 0 and relative_time >= available_duration:
                LOGGER.warning(
                    "Requested keyframe %.3fs exceeds video duration %.3fs; using the last frame.",
                    requested_time,
                    available_duration,
                )
                # Avoid seeking exactly to EOF, where no frame can be decoded.
                fps = float(stream.average_rate or 25)
                relative_time = max(0.0, available_duration - 1.0 / max(fps, 1.0))

            target = clip_start + relative_time
            seek_pts = max(0, int(target / time_base))
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

            previous: tuple[float, Any] | None = None
            selected: Any | None = None
            clip_end = clip_start + clip_duration if clip_duration > 0 else None

            for frame in container.decode(stream):
                current_time = _frame_seconds(frame, time_base)
                if current_time is None:
                    continue
                if current_time + 1e-7 < clip_start:
                    continue
                if clip_end is not None and current_time > clip_end:
                    break
                if current_time >= target:
                    if previous is not None and abs(previous[0] - target) <= abs(current_time - target):
                        selected = previous[1]
                    else:
                        selected = frame
                    break
                previous = (current_time, frame)

            if selected is None and previous is not None:
                selected = previous[1]
            if selected is None:
                raise RuntimeError(f"无法解码 {requested_time:.3f} 秒附近的视频帧。")

            array = selected.to_ndarray(format="rgb24")
            image = torch.from_numpy(array.copy()).to(dtype=torch.float32).div_(255.0)
            images.append(image)

    if not images:
        raise RuntimeError("没有抽取到任何关键帧。")
    return torch.stack(images, dim=0)


class ModelTextKeyframeExtractor:
    """Extract the nearest video frame for every timestamp in model output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "model_text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "模型输出，建议包含关键帧时间 JSON 数组。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("关键帧图像",)
    FUNCTION = "extract"
    CATEGORY = "视频/关键帧"
    DESCRIPTION = "根据模型文本中的时间点，从 VIDEO 中抽取最接近的实际画面帧。"

    def extract(self, video: Any, model_text: str):
        timestamps = parse_keyframe_times(model_text)
        LOGGER.info("Extracting keyframes at: %s", timestamps)
        return (_extract_frames(video, timestamps),)


NODE_CLASS_MAPPINGS = {
    "ModelTextKeyframeExtractor": ModelTextKeyframeExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelTextKeyframeExtractor": "按模型文本抽取视频关键帧",
}
