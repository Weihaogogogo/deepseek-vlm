# deepseek-vlm

DeepSeek V4 Flash 本身不能看图。这个项目给它加一层图片理解，让它"会看图"。

## 为什么

能识图的模型很多，MiniMax 3、各家旗舰多模态模型都行——那为什么还要费劲给一个纯文本模型加眼睛？

因为日常使用里，大部分请求根本用不到图。写代码、改 bug、看日志，是纯文本；开会记笔记、回消息，也是纯文本。真正需要看图的是少数场景：Demo 做完了截个图确认效果、页面出 bug 截图让 AI 改、办公时甩过来一张表格截图或 PDF 要解读、或者想确认某个界面长什么样。粗算下来，图片场景可能连 20% 都不到。

这些场景有个共同点：我们需要的不是像素级的还原，而是**了解这张图**——图里有什么、结构怎么样、哪里有异常。没有谁会要求模型照着截图把界面重新画一遍，我们要的只是它"看懂了"。

主模型我们用的是 deepseek-v4-flash——便宜、按量调用、文本推理强，不想因为那 20% 就换掉。

所以思路是：主模型不动，在前面加一个"看图"的层。没图的时候请求原样透传，跟直接用 deepseek 一模一样；有图的时候，先用视觉模型把图描述成文字，再交给 deepseek 推理。

这套做法最直接的收益是成本——视觉模型便宜（qwen3-vl-flash，单图几乎可以忽略），图片场景又只占两成，摊下来几乎不花钱。直接换多模态模型，整条对话都按多模态计价，那八成的纯文本请求都在为用不上的能力买单。另外文本模型没有被图像权重稀释，文本推理更干净；图片进模型是 base64，一张图就是几千 token，前置 VLM 把图变成文字描述，上下文增长也慢得多。

## 这个思路常见，难在怎么做好

"前置 VLM、后置 LLM"听起来很常规——把图翻译成文字再喂给模型，很多人第一反应是"那肯定不如真多模态，信息都丢了"。

关键就在这里：**翻译成什么、怎么翻译，决定了这个方案的上限**。如果只是简单粗暴地让 VLM"描述这张图"，那确实会丢信息——模型不知道用户关心什么，只会给一段泛泛的画面描述。

我们花力气的地方在于，把"看图"这件事拆成了有章法的两步：

**第一步，看整体。** 让 VLM-1 只对着图片，输出画面整体的结构化描述——布局、元素、文字、表格长什么样。这一步对应人扫一眼"这是什么场景"。

**第二步，带着问题细看。** 让 VLM-2 同时拿到图片和用户的问题，只输出问题驱动的聚焦描述——用户问"这个按钮在哪"，它就去找按钮，而不是把整张图再复述一遍。

为什么拆两步，而不是一个 VLM 全量描述？因为"描述全图 + 回答问题"是两件互相干扰的事：全量描述的输出 token 是最大成本项，而且模型一边报流水账一边找答案，容易顾此失彼。拆开之后，VLM-1 专心输出结构，VLM-2 专心回答问题，聚焦通道反而能真正盯住用户关心的部分。

两路描述怎么合？字段不相交，直接拼接——VLM-1 管整体、VLM-2 管聚焦，冲突以 VLM-2 为准，最后把用户问题一起交给 deepseek。deepseek 拿到的不是一段"图片文字版"，而是"整体结构 + 问题答案 + 原问题"的完整上下文。

**还有一层保障，来自 Agent 本身。** VLM 描述得再细致，也仍然可能漏读——模型没注意到的细节，描述里就是没有。但 Agent 是循环的：一轮工具调用结束、结果回到上下文之后，deepseek 发现自己需要的信息在描述里缺失时，它自己就会意识到"刚才看图没看仔细"，然后主动再读一次图。而第二次读取时，VLM-2 会带着当前的上下文倾向——模型关注的、高关联的部分——去聚焦描述。所以漏读不是信息的终点，它只是触发了一次更聚焦的重读。从体感上说，这套方案和直接的多模态模型几乎没有差别。

这套设计在几个真实场景里验证过：Demo 截图确认、报错截图定位、表格/表单解读，都能给出和真多模态模型同级别的回答。**边界也明确**：适合"看懂"图片——图里有什么、报错在哪、表单长什么样；如果是要像素级复刻网页、照图画一个一模一样的界面，那必须让模型直接看原图，文字描述不够。

## 架构

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

对外是一个 FastAPI 节点，同时兼容 OpenAI 和 Anthropic 两套协议——客户端改个 base_url 就能接，不用换 SDK。

## 其他细节

- **图片预处理**：下载 → 压缩到 1024 长边 → base64，控制上传体积和 VLM token
- **多图支持**：每轮最多 10 张，按原始顺序编号，同轮去重
- **图片缓存**：同一张图、同一个问题，第二次直接命中缓存，不重复调 VLM
- **并发控制**：VLM 信号量默认 60，防止打爆 DashScope 限流
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
| `DEEPSEEK_MODEL` | 上游 deepseek 模型名（默认 `deepseek-v4-flash`） |
| `PUBLIC_MODEL_NAME` | 对外模型名（默认 `deepseek-v4-flash-vl`） |
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
