# fake-vlm API 节点设计（v1.0）

## 目标

对外提供 **OpenAI 兼容 HTTP API**，客户端可把它当作 deepseek 直接调用（换 base_url 即用）：

- **无图请求**：原样透传 deepseek-v4-flash，零额外延迟，不注入任何东西
- **有图请求**：并发调用两个 qwen3-vl-flash 描述画面 → 合并 → 注入预制 system prompt → 转发 deepseek-v4-flash

## 非目标（v1 不做）

- 不接入 Hermes；不做多租户/计费/额度/密钥轮换
- 不做 loop 补查（VLM 输出不满足时按「不确定:/无法确认:」行触发的第三轮补查）
- 不做多图理解（多图时取第一张，日志告警）

## 技术栈

- Python 3.11 + FastAPI + uvicorn（服务）
- openai SDK（调 qwen3-vl-flash 与 deepseek-v4-flash）
- Pillow（图片压缩到 1024 长边）
- python-dotenv（配置）

## API 规格

### `POST /v1/chat/completions`

请求：OpenAI 标准 chat completions。支持字段：

- `messages`：system / user / assistant / tool 角色，内容支持字符串或 content 数组（`type: text` / `type: image_url`）
- `stream`：true / false
- `model`：接受任意值，不校验（映射内部模型）
- `temperature` / `top_p` / `max_tokens` / `stop` 等：透传 deepseek

图片格式（image_url.url）支持两种：http(s) URL、base64 data URL（`data:image/...;base64,...`）。

响应：

- 非流式：标准 chat completion JSON（id / object / created / model / choices / usage）
- 流式：SSE，`data: {...}` 逐 chunk，结尾 `data: [DONE]`；chunk 格式与 deepseek 原生一致（含 reasoning_content 时原样保留），仅 `model` 字段改写为请求中的 model 值

### `GET /v1/models`

返回 `{"object": "list", "data": [{"id": "fake-vlm", "object": "model", "owned_by": "fake-vlm"}]}`（客户端探活用）。

### 鉴权

`Authorization: Bearer <FAKE_VLM_API_KEY>`，常量比对；缺失/错误返回 401，错误体 OpenAI 格式：
`{"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}`

### 错误码

| 场景 | 状态码 | 说明 |
|---|---|---|
| 鉴权失败 | 401 | 格式同上 |
| messages 缺失/为空/格式非法 | 400 | |
| VLM 调用失败（重试后仍失败） | 502 | `{"error": {"message": "vision backend unavailable", ...}}`，**不静默降级** |
| deepseek 端错误 | 原样透传 | 状态码与错误体照抄 |

## 路由逻辑（核心）

```
收到请求
├─ 校验鉴权
├─ 解析 messages：
│   ├─ 提取"当前轮"（最后一条 user 消息）中的图片 + 文本
│   ├─ 历史消息中的图片 → 剥离并丢弃（deepseek 无视觉，直接透传会出错）
│   └─ 当前轮图片数量 > 1 → 取第一张，其余丢弃，日志告警
├─ 无图？
│   └─ 是 → 原样转发 deepseek（不动任何消息，不注入 system）
│        ├─ stream=true → SSE 透传（model 字段改写）
│        └─ stream=false → JSON 透传
└─ 有图？
    ├─ 压缩图片到长边 ≤1024px（PIL，JPEG/PNG 保持），转 base64 data URL
    ├─ 并发调用两个 VLM（asyncio.gather，各自独立异常捕获）：
    │   ├─ VLM-1：user content = [image]，无文本；temperature=0.5；不设 max_tokens
    │   ├─ VLM-2：user content = [image, text]，
    │   │   text = 用户提问：「<剥离图片后的当前轮纯文本>」\n请按系统规范输出聚焦描述。
    │   │   temperature=0.5；不设 max_tokens
    │   └─ 当前轮纯文本为空/过短（≤1 字符）→ 跳过 VLM-2
    ├─ VLM-2 输出不含「# 聚焦描述」→ 重试一次（提示语："你没有按规范输出，请严格遵守系统规范，只输出聚焦描述"）
    ├─ 任一 VLM 最终失败 → 502
    ├─ 合并器拼装（见下）→ 得到「图片信息」文本块
    ├─ 构造 deepseek 请求：
    │   ├─ system：预制 llm_system.md 全文（第一条 system）
    │   │          + 用户传入的所有 system 消息（原样跟在后面，共存不冲突）
    │   ├─ 其余消息原样，但当前轮 user 消息的 content 替换为：
    │   │   【图片·整体】<VLM-1 输出原文>
    │   │   【图片·重点】<VLM-2 输出原文>（跳过 VLM-2 时省略该段）
    │   │   【用户问题】<剥离图片后的纯文本>
    │   └─ stream / 参数透传
    └─ 转发 deepseek，SSE 或 JSON 透传
```

### 合并器

- 纯拼接，不做字段级合并（v1）：VLM-1 全貌（转录+符号图）、VLM-2 聚焦（关注点/观察/推断/无法确认），分工天然不相交
- 拼接顺序固定：整体 → 重点 → 用户问题
- 标签必须用「图片·」前缀（LLM 会复读标签，措辞必须"复读也安全"）
- v1 不做符号图行号后处理（记录为已知限制）

### 预制 system prompt 与用户 system 共存

- 有图时：messages = `[ {system: llm_system.md}, ...用户 system, ...历史消息, {user: 合并文本} ]`
- 无图时：完全不注入预制 system（deepseek 原生行为）
- 多 system 消息共存合法（OpenAI 协议允许），llm_system.md 内规则 6 已兜底"无图信息段时正常回答"——但无图时不注入，双保险

## 配置（.env，项目根目录）

```
FAKE_VLM_API_KEY=<对外鉴权 key，自签>
DASHSCOPE_API_KEY=<已提供>
DEEPSEEK_API_KEY=<服务器 ~/.hermes/.env 已有>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
PORT=8000
```

实现时 .env 用模板 `.env.example`，真实 key 由部署者填，**不得提交真实 key 进 git**（.gitignore 加 .env）。

## 文件结构

```
fake-vlm/
├── prompts/          # 已有（vlm1/vlm2/llm system），代码只读不写
├── src/
│   ├── server.py     # FastAPI 入口 + 鉴权 + /v1/chat/completions + /v1/models
│   ├── router.py     # 路由逻辑（无图透传/有图双通道）
│   ├── image_utils.py# 图片解析（URL/data URL）、下载、压缩、转 base64
│   ├── vlm_client.py # qwen3-vl-flash 调用（并发、重试）
│   ├── merger.py     # 合并器（三段拼接）
│   ├── llm_client.py # deepseek 转发（流式/非流式）
│   └── config.py     # 环境变量加载
├── .env.example
├── .gitignore
└── requirements.txt  # fastapi uvicorn openai pillow python-dotenv
```

## 验证计划（实现完成后逐项执行）

1. `uvicorn src.server:app --port 8000` 启动，`GET /v1/models` 返回正常
2. **无图非流式**：普通文本请求 → 响应与直连 deepseek 一致（可比对内容）
3. **无图流式**：`stream=true` → SSE chunk 逐块到达，`data: [DONE]` 收尾
4. **有图非流式**：URL 图片 + 文本（"这个按钮为什么点不了"类）→ 返回基于图片内容的回答
5. **有图流式**：SSE 正常，首 chunk 前有 VLM 处理延迟（数秒）属正常
6. **只发图无文本**：跳过 VLM-2，仍能回答"这是什么"
7. **多轮对话**：历史文本消息 + 当前轮带图 → 正常
8. **鉴权**：错误 key → 401
9. **VLM 失败路径**：临时把 DASHSCOPE_API_KEY 改错 → 502，不降级

## 已知边界与决策记录

- VLM 不设 max_tokens（用户决策，2026-08-05）：依赖模型默认上限；输出超长截断时靠「置信度:」尾行缺失检测，v1 不重试，记录待观察
- VLM temperature=0.5（2026-08-05 决策：转录确定性由 prompt 硬规则保障，0.5 利于冠名联想探索；若盲测发现同图文字转录漂移再单独压温）
- 多图取第一张；历史消息图片剥离丢弃（deepseek 无视觉，避免报错）
- VLM-2 位置描述（相对方位）与符号图行序是两套坐标，LLM 靠语义对应，交叉引用不精确
- 人脸识别为 qwen3-vl-flash 模型层边界，"这是谁"类问题回答质量受限（prompts 已知边界）
- 尾部怠工：长输出后「置信度:」可能虚高，影响有限暂不处理
