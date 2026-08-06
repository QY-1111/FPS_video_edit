"""ComfyUI node for the text-to-image-template HTTP endpoint."""

from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


TEXT_MODE_API_URL = "https://api5-normal-hl.amemv.com/aweme/v1/gen/pic_from_text/"
TEXT_MODE_PPE_ENV = "ppe_text_mode_no_login"
DEFAULT_TEMPLATE_ID = "7659352668552298542"

DEFAULT_EXTRA_PARAMS = [
    {"key": "is_long_text", "val": "false"},
    {"key": "enable_douyin_emoji", "val": "true"},
]

DEFAULT_AB_PARAMS = [
    {"key": "text_mode_ugc_server_opti", "val": "true"},
    {"key": "text_mode_ugc_word_segment_strategy", "val": "revert_qianwen"},
    {"key": "is_support_reframe", "val": "true"},
]


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def parse_text_and_title(value: Any, title: str = "") -> tuple[str, str]:
    """Accept plain text or a {title, body} JSON/Python-dict string."""
    explicit_title = str(title or "").strip()

    if isinstance(value, dict):
        payload = value
        original_text = ""
    else:
        original_text = str(value or "").strip()
        payload = None
        candidate = _strip_markdown_fence(original_text)
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                try:
                    payload = ast.literal_eval(candidate)
                except (SyntaxError, ValueError):
                    payload = None

    if isinstance(payload, dict):
        body = payload.get("body")
        if body is None:
            body = payload.get("正文")
        parsed_title = payload.get("title")
        if parsed_title is None:
            parsed_title = payload.get("标题")

        body = str(body or "").strip()
        parsed_title = str(parsed_title or "").strip()
        if body:
            return body, explicit_title or parsed_title

    return original_text, explicit_title


def _json_form_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).strip()


def parse_template_id_list(value: Any) -> list[str]:
    """Accept a JSON array, comma-separated IDs, or one template ID."""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value or "").strip()
        if not text:
            return [DEFAULT_TEMPLATE_ID]
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"template_id_list 不是有效的 JSON 数组：{exc}") from exc
            if not isinstance(parsed, list):
                raise ValueError("template_id_list 的 JSON 根节点必须是数组。")
            items = parsed
        else:
            items = re.split(r"[,，、;；\n]+", text)

    result = [str(item).strip() for item in items if str(item).strip()]
    if not result:
        raise ValueError("template_id_list 不能为空。")
    return result


def build_text_mode_form(
    text: str,
    template_id_list: Any = None,
    width: int = 1080,
    height: int = 1440,
    title: str = "",
) -> dict[str, Any]:
    text, title = parse_text_and_title(text, title)
    if not text:
        raise ValueError("文案 text 不能为空。")

    return {
        "text": text,
        "template_id_list": parse_template_id_list(template_id_list),
        "time_zone": "Asia/Shanghai",
        "width": int(width),
        "height": int(height),
        "mood_swing": {"custom": True},
        "split_text_timeout": 1500,
        "enable_lite_tokenize": False,
        "enable_image_base64": False,
        "enable_vedit_gateway": False,
        "enable_segment_cache_opt": True,
        "title": title,
        "scene": "main_camera",
        "extra_params": DEFAULT_EXTRA_PARAMS,
        "ab_params": DEFAULT_AB_PARAMS,
        "media_lists": [],
    }


def encode_text_mode_form(form: dict[str, Any]) -> str:
    return urllib.parse.urlencode(
        {key: _json_form_value(value) for key, value in form.items()}
    )


def post_text_mode_api(
    text: str,
    template_id_list: Any = None,
    width: int = 1080,
    height: int = 1440,
    title: str = "",
    timeout_seconds: int = 30,
) -> tuple[str, int]:
    """Send the request and return (response body, HTTP status code)."""
    form = build_text_mode_form(text, template_id_list, width, height, title)
    body = encode_text_mode_form(form).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Tt-Env": TEXT_MODE_PPE_ENV,
        "X-Use-Ppe": "1",
    }
    request = urllib.request.Request(
        TEXT_MODE_API_URL,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return response_body, int(response.getcode())
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return response_body, int(exc.code)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"图文模板接口连接失败：{reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"图文模板接口请求超过 {timeout_seconds} 秒。") from exc


class TextModeTemplateApiRequest:
    """Build and send a dynamic text-to-template request."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "支持纯正文，或包含 title/body（标题/正文）的 JSON 文本。",
                    },
                ),
                "template_id_list": (
                    "STRING",
                    {
                        "default": f'["{DEFAULT_TEMPLATE_ID}"]',
                        "multiline": False,
                        "tooltip": "支持 JSON 数组、逗号分隔或单个模板 ID。",
                    },
                ),
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 2}),
                "height": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": 2}),
                "timeout_seconds": (
                    "INT",
                    {"default": 30, "min": 1, "max": 300, "step": 1},
                ),
            },
            "optional": {
                "title": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": False,
                        "tooltip": "可选：连接上游模型输出的标题。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("response_body", "http_status_code")
    FUNCTION = "request"
    CATEGORY = "视频/API 工具"
    DESCRIPTION = "接收动态文案，构造表单并请求文字生成图文模板接口。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Image URLs are signed and expire, so the request must not reuse a
        # cached response even when the text is unchanged.
        return float("nan")

    def request(
        self,
        text: str,
        template_id_list: str,
        width: int,
        height: int,
        timeout_seconds: int,
        title: str = "",
    ):
        return post_text_mode_api(
            text=text,
            template_id_list=template_id_list,
            width=width,
            height=height,
            title=title,
            timeout_seconds=timeout_seconds,
        )


NODE_CLASS_MAPPINGS = {
    "TextModeTemplateApiRequest": TextModeTemplateApiRequest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextModeTemplateApiRequest": "图文模板 API 请求（动态文案）",
}
