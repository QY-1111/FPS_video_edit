"""ComfyUI node: conditionally extract the first frame of a video."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .keyframe_extractor import _extract_frames


COVER_DECISION_KEYS = (
    "是否取封面帧",
    "是否提取封面帧",
    "take_cover_frame",
    "extract_cover_frame",
)


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _parse_json_like_object(value: Any) -> dict[str, Any]:
    """Accept a dict, JSON text, Python-dict text, or a wrapped JSON object."""
    if isinstance(value, dict):
        return value

    text = _strip_markdown_fence(str(value or "").strip())
    if not text:
        raise ValueError("模板判断参数为空，无法判断是否需要封面帧。")

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("模板判断参数不是有效的 JSON 对象。")


def _find_cover_decision(value: Any) -> tuple[bool, Any]:
    """Find the cover-frame decision, including inside common response wrappers."""
    if isinstance(value, dict):
        for key in COVER_DECISION_KEYS:
            if key in value:
                return True, value[key]
        for child in value.values():
            found, decision = _find_cover_decision(child)
            if found:
                return True, decision
    elif isinstance(value, (list, tuple)):
        for child in value:
            found, decision = _find_cover_decision(child)
            if found:
                return True, decision
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "```")):
            try:
                nested = _parse_json_like_object(text)
            except ValueError:
                pass
            else:
                return _find_cover_decision(nested)
    return False, None


def should_extract_cover_frame(decision_json: Any) -> bool:
    """Return True only when the upstream decision explicitly requests a cover."""
    payload = _parse_json_like_object(decision_json)
    found, value = _find_cover_decision(payload)
    if not found:
        supported = "、".join(COVER_DECISION_KEYS)
        raise ValueError(f"模板判断参数中缺少封面帧字段。支持字段：{supported}。")

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    normalised = str(value).strip().lower()
    if normalised in {"是", "true", "yes", "y", "1"}:
        return True
    if normalised in {"否", "false", "no", "n", "0"}:
        return False
    raise ValueError(
        f"是否取封面帧仅支持 是/否（或 true/false），当前值：{value!r}"
    )


def _silent_execution_blocker():
    """Stop the cover branch without presenting an execution error in ComfyUI."""
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError as exc:
        raise RuntimeError(
            "当前 ComfyUI 版本不支持安全跳过分支，请更新 ComfyUI。"
        ) from exc
    return ExecutionBlocker(None)


class ConditionalCoverFrameExtractor:
    """Extract video time 0 as one IMAGE when the upstream decision says yes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decision_json": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "连接上游模板判断参数，读取‘是否取封面帧’字段。",
                    },
                ),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("封面帧",)
    FUNCTION = "extract"
    CATEGORY = "视频/关键帧"
    DESCRIPTION = "‘是否取封面帧’为‘是’时提取视频第 0 秒画面；为‘否’时静默跳过分支。"

    def extract(self, decision_json: Any, video: Any):
        if not should_extract_cover_frame(decision_json):
            return (_silent_execution_blocker(),)
        return (_extract_frames(video, [0.0]),)


NODE_CLASS_MAPPINGS = {
    "ConditionalCoverFrameExtractor": ConditionalCoverFrameExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditionalCoverFrameExtractor": "按参数提取视频封面帧（0秒）",
}
