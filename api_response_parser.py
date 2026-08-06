"""Utilities for extracting image URLs from the text-mode API response."""

from __future__ import annotations

import ast
import html
import json
import re
from typing import Any


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _parse_string_payload(text: str) -> Any:
    text = _strip_markdown_fence(text)
    if not text:
        raise ValueError("接口响应为空。")

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    errors: list[str] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            # Occasionally an HTTP node returns a JSON string containing JSON.
            if isinstance(value, str) and value.strip() != candidate.strip():
                return _parse_string_payload(value)
            return value
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

        # ParseJson/preview nodes may serialise the object as Python repr,
        # containing single quotes, None, True and False.
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as exc:
            errors.append(str(exc))

    raise ValueError("无法解析接口响应，请直接连接 BA HTTP Request 的 body 输出。")


def _parse_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return _parse_string_payload(value)
    raise TypeError(f"不支持的响应类型：{type(value).__name__}")


def _find_image_info_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        image_info_list = payload.get("gen_image_info_list")
        if isinstance(image_info_list, list):
            return image_info_list

        # Be tolerant of HTTP nodes that wrap the service response.
        for key in ("data", "result", "response", "body"):
            nested = payload.get(key)
            if isinstance(nested, str):
                try:
                    nested = _parse_string_payload(nested)
                except ValueError:
                    continue
            found = _find_image_info_list(nested)
            if found:
                return found
    return []


def extract_image_url(response_body: Any, image_index: int = 0) -> str:
    """Return gen_image_info_list[image_index].image_url as a plain string."""
    payload = _parse_payload(response_body)

    if isinstance(payload, dict):
        status_code = payload.get("status_code")
        if status_code not in (None, 0, "0"):
            status_msg = payload.get("status_msg") or "未知错误"
            raise RuntimeError(f"接口返回失败：status_code={status_code}, {status_msg}")

    image_info_list = _find_image_info_list(payload)
    if not image_info_list:
        keys = ", ".join(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"响应中没有 gen_image_info_list。当前顶层内容：{keys}")

    if image_index < 0 or image_index >= len(image_info_list):
        raise IndexError(
            f"图片索引 {image_index} 超出范围；接口共返回 {len(image_info_list)} 张图片。"
        )

    image_info = image_info_list[image_index]
    if not isinstance(image_info, dict):
        raise TypeError(f"第 {image_index} 项不是图片信息对象。")

    image_url = image_info.get("image_url")
    if not image_url:
        raise ValueError(f"第 {image_index} 项没有 image_url 字段或字段为空。")

    image_url = html.unescape(str(image_url).strip())
    if not image_url.startswith(("http://", "https://")):
        raise ValueError(f"提取到的 image_url 不是 HTTP 地址：{image_url}")
    return image_url


class ExtractTextModeImageUrl:
    """Extract one signed image URL from the text-mode service response."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response_body": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "直接连接 BA HTTP Request 的 body 输出。",
                    },
                ),
                "image_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999,
                        "step": 1,
                        "tooltip": "0 表示第一张图片，1 表示第二张图片。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
    FUNCTION = "extract"
    CATEGORY = "视频/API 工具"
    DESCRIPTION = "从图文接口响应的 gen_image_info_list 中提取一个纯 image_url 字符串。"

    def extract(self, response_body: Any, image_index: int = 0):
        return (extract_image_url(response_body, image_index),)


NODE_CLASS_MAPPINGS = {
    "ExtractTextModeImageUrl": ExtractTextModeImageUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTextModeImageUrl": "提取图文接口 Image URL",
}
