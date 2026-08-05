# deepseek-vlm

**给纯文本 LLM 补一双眼睛。** 把一个没有视觉能力的模型（deepseek-v4-flash-vl）封装成一个"看起来会看图"的多模态 API 节点——对外完全兼容 OpenAI 和 Anthropic 协议，客户端零改动就能用上"识图"能力。

---

## 场景：为什么需要它？

**你有模型，但它看不见。** deepseek-v4-flash 这类纯文本模型，推理强、便宜、快，就是不支持图片输入。想让它在"对话里出现一张截图、一张表格、一张产品图"时能接住话，通常只能：

- 换一个多模态模型（贵、可能不如你现有的文本模型好用）
- 让用户手打描述（体验差，信息丢失）

deepseek-vlm 解决了这个矛盾：**保留你现有的文本模型，在它前面加一层"看图描述"代理**。

典型场景：

- **群聊/客服机器人**：用户发截图问问题，机器人能看懂图再回答
- **Agent 工具链**：让模型"读"一张图片文件、识别 UI 截图、分析图表
- **想用 deepseek 但又要多模态**：白嫖文本模型的长处，图片交给 VLM 补位

## 价值：钱省了，能力没少

| 方案 | 成本 | 说明 |
|---|---|---|
| 直接用多模态大模型 | 高 | 每张图按 token 计费，贵几倍到几十倍 |
| deepseek-vlm | 极低 | 图片描述走便宜 VLM（单图约 ¥0.0006），正文推理走 deepseek |

- **无图请求零开销**：纯文本直接透传 deepseek，不经过任何 VLM，延迟、成本都跟直接用 deepseek 一样
- **有图才花钱**：只有图片进对话时才调用 VLM 描述，且描述有缓存（同一张图同一问题不重复算）

## 架构：一个节点，双协议

```
客户端（Claude Code / Hermes / 任意 OpenAI 客户端）
        │
        ▼
┌─────────────────────────────────────┐
│  deepseek-vlm 节点（FastAPI）        │
│  /v1/chat/completions  (OpenAI)      │
│  /v1/messages          (Anthropic)   │
│                                     │
│  无图 ──► 直接透传 deepseek          │
│  有图 ──► VLM-1 + VLM-2 并发描述     │
│           │      │                  │
│           ▼      ▼                  │
│       qwen3-vl-flash（便宜 VLM）     │
│           │      │                  │
│           └─合并描述─► deepseek 推理 │
└─────────────────────────────────────┘
```

- **双协议**：OpenAI Chat Completions + Anthropic Messages 都支持，客户端按自己习惯连
- **模型名**：对外声称 `deepseek-v4-flash-vl`（也兼容旧名 `deepseek-v4-flash`）
- **鉴权**：Bearer token / x-api-key，环境变量配置

## 做法：为什么是两个 VLM？

核心设计是**双通道描述**——仿照人类看图的方式：先整体扫一眼，再针对问题细看。

| 通道 | 输入 | 输出 | 类比 |
|---|---|---|---|
| **VLM-1** | 纯图片 | 画面整体结构化描述（布局、元素、文字、表格/图表结构） | 扫一眼"这是什么场景" |
| **VLM-2** | 图片 + 用户问题 | 问题驱动的聚焦描述（"这个按钮在哪"→ 找按钮） | 带着问题细看 |

合并规则：

- 两路描述**拼接**（字段不相交：VLM-1 管整体，VLM-2 管聚焦）
- 冲突以 VLM-2（聚焦）为准
- 最后把用户问题拼进去，让 deepseek 在完整上下文里推理

**为什么不做单 VLM 全量描述？**

1. **成本**：全量描述输出 token 是最大开销，聚焦通道只输出问题相关的部分，省一半
2. **准确率**：让一个 VLM 同时"描述全图 + 回答问题"，容易顾此失彼；拆成两路各干各的，聚焦通道能真正盯住问题
3. **上下文**：deepseek 拿到的是"结构化的图信息"而不是"图片像素"，token 消耗可控

其他关键设计：

- **图片预处理**：下载 → 压缩到 1024 长边 → base64，控制上传体积和 VLM token
- **多图支持**：每轮最多 10 张，按原始顺序编号，同轮去重
- **图片缓存**：`图片hash|问题指纹` 为键，同一图同一问题不重复调 VLM
- **并发控制**：VLM 信号量（默认 60），防止打爆 DashScope 限流
- **失败不静默**：VLM 挂了返回 502，不拿空描述糊弄

## 快速开始

```bash
# 1. 配置（复制模板填真实值）
cp .env.example .env
#   编辑 .env：填 DEEPSEEK_VLM_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY

# 2. 安装依赖
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 启动
uvicorn src.server:app --host 0.0.0.0 --port 8000

# 4. 测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_VLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash-vl","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

**接入客户端**：

```python
# OpenAI 兼容（base_url 指向节点）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="你的key")
resp = client.chat.completions.create(
    model="deepseek-v4-flash-vl",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "这图里有什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]}],
)
```

```python
# Anthropic 兼容
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="你的key")
resp = client.messages.create(
    model="deepseek-v4-flash-vl",
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "这图里有什么？"},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
    ]}],
)
```

## 配置项（.env）

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_VLM_API_KEY` | 对外鉴权 key（客户端用它连本节点） |
| `DASHSCOPE_API_KEY` | 阿里云百炼 key（调 qwen3-vl-flash VLM） |
| `DEEPSEEK_API_KEY` | deepseek key（正文推理） |
| `DEEPSEEK_BASE_URL` | deepseek 上游地址（默认官方） |
| `DEEPSEEK_MODEL` | 对外模型名（默认 `deepseek-v4-flash-vl`） |
| `PORT` | 监听端口（默认 8000） |

## 目录结构

```
deepseek-vlm/
├── prompts/    # VLM-1 / VLM-2 / 正文 LLM 的 system prompt 模板
├── src/        # 路由、VLM 客户端、合并器、双协议服务层
│   ├── server.py            # FastAPI 入口 + 鉴权 + 三端点
│   ├── router.py            # OpenAI 路由 + 图片提取/缓存/合并
│   ├── anthropic_route.py   # Anthropic 路由（/v1/messages）
│   ├── anthropic_protocol.py# OpenAI→Anthropic 协议转换 + SSE
│   ├── vlm_client.py        # qwen3-vl-flash 调用（并发/超时/重试）
│   ├── llm_client.py        # deepseek 转发
│   ├── image_utils.py       # 图片下载/压缩/SSRF 防护
│   ├── merger.py            # 双通道合并
│   └── config.py            # 环境配置
├── tests/      # 单元测试 + SDK 合规测试
├── DESIGN.md   # 详细设计文档
└── README.md
```

## 测试

```bash
# 单元测试（协议转换、路由、图片工具）
pytest tests/unit -v

# SDK 合规测试（用官方 anthropic/openai SDK 端到端验证）
pytest tests/sdk_compat_test.py tests/openai_deep_test.py -v
```

## 许可证

Apache-2.0
