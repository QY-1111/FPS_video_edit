# 按模型文本抽取视频关键帧

这是一个 ComfyUI 自定义节点。它接收：

- ComfyUI 原生 `VIDEO`；
- 上游模型输出的 `STRING` 文本。

节点读取文本里的关键帧时间，从视频中抽出对应画面，并以一个 `IMAGE` 批次输出。批次中的图片顺序与文本中的时间顺序一致。

## 安装

将整个 `ComfyUI-ModelText-KeyframeExtractor` 文件夹复制到：

```text
ComfyUI/custom_nodes/
```

然后完全重启 ComfyUI。节点位于：

```text
视频 → 关键帧 → 按模型文本抽取视频关键帧
```

该节点使用 ComfyUI 自带的 PyTorch 和 PyAV，不需要单独安装依赖。如果控制台提示缺少 `av`，请更新 ComfyUI，或在 ComfyUI 自己的 Python 环境安装 `av`。

## 连接方法

```text
Load Video 的 VIDEO ───────────────┐
                                   ├─ 按模型文本抽取视频关键帧 ─→ IMAGE
模型判断节点的 STRING 文本 ────────┘
```

输出的 `IMAGE` 是批次，可连接 `Preview Image`、`Save Image` 或后续图像处理节点。

## 推荐的模型输出

```json
{
  "视频类型": "口播",
  "抽帧方式": "多帧",
  "关键帧时间": [
    "00:00:00",
    "00:00:05.500",
    "00:00:10"
  ],
  "是否应用图文模板": "是"
}
```

模型在 JSON 外增加说明文字或 Markdown 的 `json` 代码块也可以识别。时间还支持：

- `HH:MM:SS`，如 `00:01:03.5`；
- `MM:SS`，如 `01:03.5`；
- 秒数，如 `5`、`5.2s`、`5.2秒`；
- 中文时长，如 `1分3秒`。

## 注意

- 时间以输入视频片段的开头为 `00:00:00`。
- 节点选择离目标时间最近的实际视频帧。
- 重复时间会自动去重。
- 超出视频时长的时间会使用最后一帧，并在 ComfyUI 控制台显示提醒。
- 当前输入类型针对截图里的 ComfyUI 原生 `Load Video`。如果使用 Video Helper Suite 的旧版加载节点并输出 `IMAGE`，请直接使用其图片批次，或先转换为原生 `VIDEO`。

## 提取图文接口 Image URL

插件还提供 `视频 → API 工具 → 提取图文接口 Image URL` 节点，用于解析文字图文接口的响应。

连接方式：

```text
BA HTTP Request.body
        ↓
提取图文接口 Image URL
        ↓
DownloadImageByUrl.url
```

不需要再连接 `ParseJson`。`image_index` 为 `0` 时提取第一张图片，为 `1` 时提取第二张，以此类推。节点兼容标准 JSON、Python 字典文本以及被 HTTP 节点二次包装在 `data`、`result`、`response` 或 `body` 中的响应。

## 图文模板 API 请求（动态文案）

`视频 → API 工具 → 图文模板 API 请求（动态文案）` 会直接完成表单编码和 HTTP 请求，不再需要手工维护固定的 cURL。

```text
上游正文 STRING ─→ text                                response_body
上游标题 STRING ─→ title（可选）  图文模板 API 请求 ───────────→ 提取图文接口 Image URL
模板 ID / 宽 / 高 ─→ 节点控件                         http_status_code
```

- `template_id_list` 支持 JSON 数组、逗号分隔 ID 或单个 ID。
- `title` 可以不连接，接口会收到空标题。
- `text` 可直接接纯正文，也可接 `{"title":"...","body":"..."}`；节点会自动拆出标题和正文。中文字段 `标题`、`正文` 也支持。
- 如果同时连接了可选的 `title` 输入，外部标题会覆盖 JSON 中的标题。
- 节点每次排队都会重新请求，避免 ComfyUI 缓存已经过期的签名图片地址。
- URL、Headers、时区和实验参数均使用 `business.py` 中的固定配置。
