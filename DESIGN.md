# deepseek-vlm API 节点设计（v2）

## 目标

对外提供 **OpenAI 兼容 HTTP API** 与 **Anthropic Messages 兼容端点**，客户端可把它当作 deepseek 直接调用（换 base_url 即用）：

- **无图请求**：原样透传 deepseek-v4-flash-vl，零额外延迟，不注入任何东西（历史含图且缓存命中时，注入最近一张历史图描述，见「图片描述缓存」）
- **有图请求**：并发调用两个 qwen3-vl-flash 描述画面 → 合并 → 注入预制 prompt → 转发 deepseek-v4-flash-vl
- **多图支持（v2）**：每轮最多 10 张，编号按原始发送顺序，同轮按图片内容哈希去重；兼容 WorkBuddy 式「图片消息 + 文本消息分离」的形态

## 非目标（v1/v2 不做）

- 不接入 Hermes；不做多租户/计费/额度/密钥轮换
- 不做 loop 补查（VLM 输出不满足时按「不确定:/无法确认:」行触发的第三轮补查）
- 不做 VLM 输出后处理（如符号图行号校正）

## 技术栈

- Python 3.11 + FastAPI + uvicorn（服务）
- openai SDK（调 qwen3-vl-flash 与 deepseek-v4-flash-vl）
- Pillow（图片压缩到 1024 长边）
- python-dotenv（配置）

## API 规格

### `POST /v1/chat/completions`

请求：OpenAI 标准 chat completions。支持字段：

- `messages`：system / user / assistant / tool 角色，内容支持字符串或 content 数组（`type: text` / `type: image_url`；`image_url` 兼容 `{url}` 对象与裸字符串两种形态——WorkBuddy 发过字符串）
- `stream`：true / false（字符串 `'false'`/`'0'` 也按 false 处理，客户端常把 stream 发成字符串）
- `model`：接受任意值，不校验（映射内部模型）
- `temperature` / `top_p` / `max_tokens` / `stop` 等：透传 deepseek
- `stream_options`：双向收口，见「决策记录」

图片格式（image_url.url）支持两种：http(s) URL、base64 data URL（`data:image/...;base64,...`）。

响应：

- 非流式：标准 chat completion JSON（id / object / created / model / choices / usage）
- 流式：SSE，`data: {...}` 逐 chunk，结尾 `data: [DONE]`；chunk 格式与 deepseek 原生一致（含 reasoning_content 时原样保留），仅 `model` 字段改写为请求中的 model 值

### `POST /v1/messages`

Anthropic Messages 兼容端点，见下文专节。

### `GET /v1/models`

返回 `{"object": "list", "data": [{"id": "<DEEPSEEK_MODEL>", ...}]}`（客户端探活用）。

### 鉴权

`Authorization: Bearer <DEEPSEEK_VLM_API_KEY>` 或 `x-api-key: <DEEPSEEK_VLM_API_KEY>`（Anthropic 客户端两种都常见），常量比对；缺失/错误返回 401，错误体 OpenAI 格式。

### 错误码

| 场景 | 状态码 | 说明 |
|---|---|---|
| 鉴权失败 | 401 | |
| messages 缺失/为空/格式非法 | 400 | |
| 图片 URL 非法（含 SSRF 拦截） | 400 | `invalid_image_url` |
| 请求体超过 50MB | 413 | 流式读取中途截断，不解析 |
| VLM 调用失败（重试后仍失败） | 502 | **不静默降级** |
| deepseek 端错误 | 原样透传 | 状态码与错误体照抄 |

## 路由逻辑（核心）

```
收到请求
├─ 校验鉴权
├─ 请求体 ≤ 50MB 限制（超限 413，不进入解析）
├─ 解析 messages（_validate_messages：角色合法、content 结构合法）
├─ 提取"当前轮"图片 _extract_current_images：
│   ├─ 从最后一条消息倒序扫描 user 消息；跳过无图的 user（WorkBuddy 把图片消息
│   │   与文本消息分成两条 user）；遇到第一条非 user 消息（assistant/system/tool）
│   │   即停——其后才是"本轮"，之前是历史
│   ├─ 倒序收集完成后反转回原始发送顺序（最早的图编号为 1）
│   ├─ 按内容哈希去重（保留首次出现）；截断到最多 10 张
│   └─ 无图且最后一条是纯图 tool 消息 → 回退取该消息的图（agent 本轮回调 Read）
├─ 提取焦点上下文 _pick_focus_text（≤1000 字符预算）：
│   ├─ 倒序扫描 user + assistant 文本（agent 自述如"我需要先看图片X再决定"
│   │   承载真实意图，仅看最新 user 不够）
│   └─ tool 消息不参与（tool_result 解析后是 role=tool）
├─ 无图？
│   ├─ 逐消息剥离历史图片 _strip_message_images（见决策记录）
│   ├─ 查历史图缓存：最近一张有缓存的图 → 合并描述（整体+重点+当前问题）
│   │   注入到最后一条 user 消息之前（LLM_SYSTEM 在前、描述在后），与有图轮
│   │   布局一致 → deepseek prefix cache 命中历史（见决策记录）
│   ├─ assistant tool_calls 消息补 reasoning_content 空串
│   └─ 转发 deepseek（SSE 或 JSON 透传，model 改写）
└─ 有图？
    ├─ 逐图查描述缓存（键=图片hash|问题指纹）——下载前查，命中完全跳过
    │   下载/压缩/VLM（http URL 的键是 url 字符串，data URL 是内容哈希）
    ├─ 仅对未命中的图：下载（SSRF 防护 + 20MB 上限）→ 压缩（长边 ≤1024）→ base64
    ├─ 每张图并发跑 VLM-1 全貌 + VLM-2 聚焦（当前轮文本 ≤1 字符跳过 VLM-2）：
    │   信号量 4 上限并发；429/5xx/连接错误退避重试；任一最终失败 → 502
    ├─ VLM-2 输出不含「# 聚焦描述」→ 重试一次
    ├─ 结果写缓存（LRU 200）
    ├─ 合并器拼装（见下）→「图片信息」文本块
    ├─ 构造 deepseek 请求（前缀缓存对齐）：
    │   ├─ system 消息保持原位（共享前缀的一部分）
    │   ├─ 最后一条 user 消息位置替换为：{user: llm_system.md} + {user: 合并文本}
    │   ├─ 其余消息剥离图片后原样
    │   └─ assistant tool_calls 补 reasoning_content 空串
    └─ 转发 deepseek，SSE 或 JSON 透传
```

### 合并器（v2 多图）

- 纯拼接，不做字段级合并：VLM-1 全貌（转录+符号图）、VLM-2 聚焦（关注点/观察/推断/无法确认），分工天然不相交
- 拼接顺序固定：整体 → 重点 → 用户问题
- 标签必须用「图片·」前缀（LLM 会复读标签，措辞必须"复读也安全"）
- **单图**：`【图片·整体】` / `【图片·重点】`（可省略）`【用户问题】`，无编号——与 v1 单图输出完全一致，缓存的历史单图描述形状不变
- **多图**：`【图片·整体】1`、`【图片·重点】1`、`【图片·整体】2`…编号按原始发送顺序（最早的图是 1）
- v1 不做符号图行号后处理（记录为已知限制）

### 前缀缓存对齐策略（v2）

deepseek 的 prefix cache 对「请求开头部分」命中；请求头部任何变化都会让整条缓存失效（已验证）。

- **LLM_SYSTEM 不再作为第一条 system 注入**，而是作为 user 消息放在最后一条 user 消息位置（有图轮替换原 user；无图轮仅在历史图缓存命中时插在原 user 之前）——两条路径中 LLM_SYSTEM 之前的历史前缀完全一致，历史 tokens 每轮都能命中缓存
- 有图轮的「历史+system」与无图轮的「历史+system」形状一致（无图轮同样会剥离历史图片；缓存命中时同样注入 LLM_SYSTEM + 描述）
- Anthropic 端点显式 system 消息放到末尾（不进共享前缀，避免污染缓存）；OpenAI 端点 system 保持原位
- 无图且无历史图缓存时不注入任何东西（deepseek 原生行为，最省）

## 配置（.env，项目根目录）

```
DEEPSEEK_VLM_API_KEY=<对外鉴权 key，自签>
DASHSCOPE_API_KEY=<已提供>
DEEPSEEK_API_KEY=<服务器 ~/.hermes/.env 已有>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
PUBLIC_MODEL_NAME=deepseek-v4-flash-vl
PORT=8000
```

实现时 .env 用模板 `.env.example`，真实 key 由部署者填，**不得提交真实 key 进 git**（.gitignore 加 .env）。

## 文件结构

```
deepseek-vlm/
├── prompts/          # vlm1/vlm2/llm system，代码只读不写
├── src/
│   ├── server.py             # FastAPI 入口 + 鉴权 + 50MB body 限制 + 路由分发
│   ├── router.py             # OpenAI 端点路由核心（无图透传/有图双通道/缓存/多图）
│   ├── anthropic_route.py    # Anthropic 端点：解析复用 + 协议转换 + 错误转换
│   ├── anthropic_protocol.py # Anthropic <-> OpenAI 协议适配（请求解析/响应转换/SSE 状态机）
│   ├── image_utils.py        # 图片解析（URL/data URL）、下载（SSRF 防护）、压缩、哈希
│   ├── vlm_client.py         # qwen3-vl-flash 调用（信号量 4、退避重试）
│   ├── merger.py             # 合并器（单图/多图编号拼接）
│   ├── llm_client.py         # deepseek 转发（流式/非流式、stream_options 收口）
│   └── config.py             # 环境变量加载
├── tests/
│   ├── unit/                 # pytest 纯函数单元测试（无网络，SimpleNamespace mock chunk）
│   ├── openai_deep_test.py   # 集成回归脚本（需服务运行）
│   └── sdk_compat_test.py    # SDK 兼容回归脚本（需服务运行）
├── .env.example
├── .gitignore
└── requirements.txt
```

## 验证计划（v1 基线，仍有效）

1. `uvicorn src.server:app --port 8000` 启动，`GET /v1/models` 返回正常
2. **无图非流式**：普通文本请求 → 响应与直连 deepseek 一致（可比对内容）
3. **无图流式**：`stream=true` → SSE chunk 逐块到达，`data: [DONE]` 收尾
4. **有图非流式**：URL 图片 + 文本（"这个按钮为什么点不了"类）→ 返回基于图片内容的回答
5. **有图流式**：SSE 正常，首 chunk 前有 VLM 处理延迟（数秒）属正常
6. **只发图无文本**：跳过 VLM-2，仍能回答"这是什么"
7. **多轮对话**：历史文本消息 + 当前轮带图 → 正常
8. **鉴权**：错误 key → 401
9. **VLM 失败路径**：临时把 DASHSCOPE_API_KEY 改错 → 502，不降级

v2 增补：多图（10 张/顺序编号/同轮去重）、WorkBuddy 双 user 形态、缓存命中（同图同问题第二次请求秒回）、Claude Code 工具循环（Read 图 → tool_result 回填）、Anthropic 流式工具混合流。协议层回归由 `tests/unit/`（pytest）覆盖。

## 已知边界与决策记录

### 历史决策（仍有效）

- VLM 不设 max_tokens（用户决策，2026-08-05）：依赖模型默认上限；输出超长截断时靠「置信度:」尾行缺失检测，v1 不重试，记录待观察
- VLM temperature=0.5（2026-08-05 决策：转录确定性由 prompt 硬规则保障，0.5 利于冠名联想探索；若盲测发现同图文字转录漂移再单独压温）
- VLM-2 位置描述（相对方位）与符号图行序是两套坐标，LLM 靠语义对应，交叉引用不精确
- 人脸识别为 qwen3-vl-flash 模型层边界，"这是谁"类问题回答质量受限（prompts 已知边界）
- 尾部怠工：长输出后「置信度:」可能虚高，影响有限暂不处理

### v2 决策记录

- **多图支持**：每轮最多 10 张（超出按原始顺序截断保留后 10 张）；编号按原始发送顺序（合并器「整体」1/2/…）；同轮按内容哈希去重（http URL 的哈希键是 url 字符串——不下载无法做内容哈希，设计取舍：同一张图用不同 URL 发送不会去重，data URL 则按真实内容哈希，两种形态同内容同键）
- **图片描述缓存**：键=`图片hash|问题指纹`（VLM-2 的 focus 是问题驱动的，同一张图不同问题必须命中不同条目——焦点串味修复）；LRU 200；**下载前查缓存**，命中完全跳过下载+压缩+VLM；无图轮查历史图缓存（见下）；冷启动（缓存未命中）时纯图 tool_result 回填占位文本「【图片】此消息包含一张图片，内容未经视觉解析（缓存未命中）」而非空串——曾回填空串，deepseek 误判"read_file 返回空"；命中则回填整体描述（`_find_cached_overall` 忽略问题指纹，纯图 tool_result 没有问题可用）
- **无图轮历史图注入**：无图但历史含图且缓存命中时，把最近一张有缓存图片的合并描述（整体+重点+当前问题）注入到最后一条 user 之前，让 deepseek 保留视觉上下文（agent 读图工具循环必需）；**多轮历史多图只注入最近一张**（防上下文膨胀）
- **工具消息重排（_normalize_tool_pairing）**：deepseek 要求 tool 消息**紧跟**对应 assistant tool_calls；Claude Code 会在 tool_use 与 tool_result 之间插入 user 纯文本 → 重排到 assistant 之后；孤儿 tool 消息（无匹配 assistant）排到末尾兜底；重复 tool_call_id 跨轮复用不崩（只挂第一次出现处）
- **reasoning_content 空串补位**：deepseek thinking 模式要求 assistant tool_calls 消息携带 reasoning_content；Anthropic 历史无此字段 → 缺失时补空串（空串被接受），已有值不覆盖
- **历史图片剥离与回填（_strip_message_images）**：text+image 混合压平成纯字符串（deepseek 兼容层拒绝数组 content 的 tool/assistant 消息）；纯图 user 历史消息直接丢弃；纯图 tool 消息按缓存回填（见上）
- **anthropic_sse 状态机**（已修复两个高危 bug，回归测试覆盖）：
  - 文本块与工具块**共享全局递增的块索引计数器**——曾各自从 0 计数，混合流（文本+工具同轮）双 index 0 冲突，Anthropic SDK 解析失败
  - **finish_reason 在 delta 之后处理**——尾 chunk 同时带 delta 与 finish_reason 时，先发完该块所有 content_block_delta 再发 content_block_stop；曾 stop 先于尾块 delta，流式末尾丢字
  - usage 搬运：deepseek 把 usage 挂在带 choices 的尾 chunk，OpenAI 标准也可能是「空 choices + usage」的尾 chunk，统一搬运到 message_delta
  - 空流不 hang：只有 message_start / message_delta / message_stop
- **SSRF 防护**：下载前 getaddrinfo 解析主机名，拒绝回环/私网（RFC1918）/链路本地（含 169.254.169.254 元数据地址）/未指定地址（IPv4-mapped 先还原再判）；**每个重定向跳转重新校验**；下载 20MB 上限；服务端 50MB 请求体上限（413）
- **VLM 并发治理**：全局信号量 4（多图并发上限）；429/5xx/连接错误退避重试 2 次（1s/2s）；最终失败 502 不降级；VLM-2 缺「# 聚焦描述」头重试一次
- **stream_options 双向收口**：Anthropic 流式强制 `include_usage=true`（Claude Code 的 /context 与 token 统计依赖）；OpenAI 流式默认 true、显式 `include_usage=false` 尊重原样（透传语义）；非流式剥离 stream_options（deepseek 拒绝该参数）
- **qwen3-vl-flash 对抽象 logo 冠名不稳定**：无文字的抽象图形命名可能漂移（盲测观察中），prompts 已要求"无法确认就写不确定"
- 多图轮图片描述是 VLM 逐张独立产出后拼接，跨图关系（如"两图对比"）依赖 LLM 推理，不保证准确

---

# Anthropic 兼容端点（v1.1 → v2）

## 目标

新增 `POST /v1/messages`（Anthropic Messages API 兼容），让 OpenCode / Claude Code 等 agent 工具以 anthropic 协议直连。内部路由逻辑（无图透传 / 有图双 VLM / 缓存 / 多图）完全复用，只做协议转换层。

## 鉴权

`Authorization: Bearer <DEEPSEEK_VLM_API_KEY>` 或 `x-api-key: <DEEPSEEK_VLM_API_KEY>` 均接受（Anthropic 客户端两种都常见）。

## 请求解析（Anthropic → OpenAI 内部格式）

| Anthropic | OpenAI 内部 |
|---|---|
| 顶层 `system`（字符串或 text 块数组） | 解析为 system 文本（无图路径置首；有图路径同样置首，LLM_SYSTEM 以 user 消息放在最后一条 user 位置——前缀缓存对齐） |
| user content 字符串 | user content 字符串 |
| user content 块 `{type:"text"}` | `{type:"text"}` |
| user content 块 `{type:"image", source:{type:"base64", media_type, data}}` | `{type:"image_url", image_url:{url:"data:<media_type>;base64,<data>"}}` |
| user content 块 `{type:"image", source:{type:"url", url}}` | `{type:"image_url", image_url:{url}}` |
| user content 块 `{type:"tool_result", tool_use_id, content}` | 独立消息 `{role:"tool", tool_call_id, content}`（紧跟对应 assistant 消息）；content 为纯 image 块时保留为 image_url（Claude Code Read 工具形态，路由层才能提取进视觉管线） |
| assistant content 字符串 | assistant content 字符串 |
| assistant 块 `{type:"tool_use", id, name, input}` | assistant 消息 `tool_calls:[{id, type:"function", function:{name, arguments:json.dumps(input)}}]` |
| assistant 块 `{type:"thinking", ...}` | 忽略（deepseek 不需要） |
| `tools:[{name, description, input_schema}]` | `[{type:"function", function:{name, description, parameters}}]` |
| `tool_choice: "auto"|"any"|"none"|{type:"tool", name}` | `"auto"|"required"|"none"|{type:"function", function:{name}}` |
| `max_tokens`（必填） | 透传 |
| `temperature`/`top_p`/`stop_sequences` | `temperature`/`top_p`/`stop` |
| `top_k` | 忽略（OpenAI 协议无此参数） |

## 响应转换（OpenAI → Anthropic）

非流式：

- OpenAI `choices[0].message.content` → `content: [{type:"text", text}]`
- OpenAI `tool_calls` → `content: [{type:"tool_use", id, name, input: json.loads(arguments)}]`（arguments 非 JSON 时 input 为空对象）
- `finish_reason`：stop→`end_turn`，tool_calls→`tool_use`，length→`max_tokens`，缺省→`end_turn`
- `usage.prompt_tokens/completion_tokens` → `input_tokens/output_tokens`
- 顶层 `{id, type:"message", role:"assistant", model, stop_reason, stop_sequence:null, usage}`

流式（OpenAI chunk → Anthropic SSE 事件，v2 状态机）：

| OpenAI chunk 内容 | Anthropic 事件 |
|---|---|
| 首个带 choices 的 chunk | `message_start`（初始 usage 与 model） |
| `delta.content` 增量 | `content_block_start(type:"text", index=块计数器++)`（首次）→ `content_block_delta(type:"text_delta", text)` |
| `delta.tool_calls[i]`（id/name/arguments 片断） | `content_block_start(type:"tool_use", index=块计数器++, id, name)`（该 src index 首次）→ `content_block_delta(type:"input_json_delta", partial_json)` |
| `finish_reason` | 先统一关闭所有已开块（`content_block_stop`，**在全部 delta 之后**）→ `message_delta(delta:{stop_reason}, usage)` |
| 流结束 | `message_stop` |
| 空流 | 仅 `message_start`（id=msg_empty）/ `message_delta` / `message_stop`，客户端不 hang |
| `usage`（deepseek 挂最后 chunk；也可能空 choices 尾 chunk） | 搬运到 `message_delta.usage` |
| `reasoning_content` | 忽略（deepseek 思考链不转 thinking 块，避免客户端未启用 thinking 时协议报错） |
| 最终 `[DONE]` | 不输出（message_stop 已收尾） |

块索引：文本块与工具块**共享全局递增计数器**，混合流（文本+工具同轮）索引必然唯一——曾双 index 0 冲突导致 Anthropic SDK 解析失败（已修复，回归测试覆盖）。

## 错误格式

Anthropic 风格：`{"type":"error","error":{"type":"api_error"|"invalid_request_error"|"authentication_error",...}}`
映射：401→authentication_error；400→invalid_request_error；413→invalid_request_error（body 超限）；VLM 失败→502 api_error；deepseek 错误→原状态码+错误体转 Anthropic 格式；流式上游错误且未发流前 → 直接输出 error 事件。

## 路由

与 OpenAI 端点共用核心逻辑（`_extract_current_images` / `_pick_focus_text` / 图片缓存 / VLM 双通道 / 多图合并 / 图片剥离 / 工具消息重排 / reasoning_content 补位全部复用），差异仅在：

- 顶层 system → 第一条 system 消息；显式 system 消息放到请求末尾（不进共享前缀，保护 prefix cache）
- 流式强制 `stream_options.include_usage=true`（Claude Code /context 与 token 统计依赖）
- 流式响应经 `anthropic_sse` 状态机转换；非流式经 `to_anthropic_message` 转换
