"""Anthropic Messages API <-> OpenAI protocol adapter.

Request side: Anthropic -> internal OpenAI-format messages + passthrough params.
Response side: OpenAI responses -> Anthropic message / SSE event stream.
"""
import json
import logging

logger = logging.getLogger(__name__)


# ---------- Request parsing: Anthropic -> OpenAI format ----------

def parse_system(system) -> str:
    """Anthropic top-level system field -> plain text (or '')."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _content_to_openai(content) -> list[dict]:
    """Anthropic content array -> OpenAI content array (images to image_url)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    out: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            source = block.get("source") or {}
            stype = source.get("type")
            if stype == "base64":
                media = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                out.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
                )
            elif stype == "url":
                out.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
        # tool_result blocks are handled by the caller (become separate tool messages)
    return out


def _assistant_content_to_openai(content) -> tuple[list[dict], list[dict], str]:
    """Anthropic assistant content -> (openai content parts, tool_calls list, thinking text)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}], [], ""
    parts: list[dict] = []
    tool_calls: list[dict] = []
    thinking_parts: list[str] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
    return parts, tool_calls, "\n\n".join(p for p in thinking_parts if p)


def parse_messages(anthropic_msgs: list) -> list[dict]:
    """Anthropic messages -> OpenAI messages (tool_result becomes role=tool).

    tool_result blocks are emitted immediately as role=tool messages so they
    stay right after the assistant turn they reply to (OpenAI protocol
    requires tool messages to directly follow the matching assistant message).
    """
    if not isinstance(anthropic_msgs, list):
        raise ValueError("messages must be an array")
    out: list[dict] = []
    for msg in anthropic_msgs:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                user_blocks = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
                results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if user_blocks:
                    out.append({"role": "user", "content": _content_to_openai(user_blocks)})
                for res in results:
                    tid = res.get("tool_use_id", "")
                    rcontent = res.get("content", "")
                    if isinstance(rcontent, list):
                        # Keep image blocks as image_url so the route layer can
                        # extract them for the vision pipeline (agent tools like
                        # Claude Code's Read return images inside tool_result).
                        parts: list[dict] = []
                        for b in rcontent:
                            if not isinstance(b, dict):
                                continue
                            bt = b.get("type")
                            if bt == "text":
                                parts.append({"type": "text", "text": b.get("text", "")})
                            elif bt == "image":
                                src = b.get("source") or {}
                                if src.get("type") == "base64":
                                    media = src.get("media_type", "image/jpeg")
                                    parts.append(
                                        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{src.get('data','')}"}}
                                    )
                                elif src.get("type") == "url":
                                    parts.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
                        out.append({"role": "tool", "tool_call_id": tid, "content": parts if parts else ""})
                    else:
                        out.append({"role": "tool", "tool_call_id": tid, "content": str(rcontent)})
                continue
            out.append({"role": "user", "content": _content_to_openai(content) if isinstance(content, list) else content})
        elif role == "assistant":
            if isinstance(content, list):
                parts, tool_calls, thinking_text = _assistant_content_to_openai(content)
                msg_out: dict = {"role": "assistant", "content": parts if parts else None}
                if tool_calls:
                    msg_out["tool_calls"] = tool_calls
                else:
                    msg_out["content"] = parts if parts else ""
                if thinking_text:
                    msg_out["reasoning_content"] = thinking_text
                out.append(msg_out)
            else:
                out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content if isinstance(content, str) else parse_system(content)})
        else:
            raise ValueError(f"unsupported anthropic role: {role}")
    return out


def parse_tools(tools) -> list[dict]:
    """Anthropic tools -> OpenAI tools."""
    if not tools:
        return []
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def parse_tool_choice(tool_choice):
    """Anthropic tool_choice -> OpenAI tool_choice (or None)."""
    if tool_choice is None or tool_choice == "auto":
        return "auto"
    if tool_choice == "none":
        return "none"
    if tool_choice == "any":
        return "required"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return "auto"


def parse_body(body: dict) -> tuple[list[dict], dict, str]:
    """Parse an Anthropic /v1/messages body.

    Returns (openai_messages, passthrough_params, top_level_system_text).
    """
    system_text = parse_system(body.get("system"))
    messages = parse_messages(body.get("messages", []))
    params: dict = {}
    if body.get("max_tokens") is not None:
        params["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        params["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        params["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        params["stop"] = body["stop_sequences"]
    tools = parse_tools(body.get("tools"))
    if tools:
        params["tools"] = tools
    tc = parse_tool_choice(body.get("tool_choice"))
    if tc is not None:
        params["tool_choice"] = tc
    return messages, params, system_text


# ---------- Response conversion: OpenAI -> Anthropic ----------

def _stop_reason(finish_reason: str | None) -> str:
    return {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(
        finish_reason or "", "end_turn"
    )


def to_anthropic_message(openai_resp: dict, model: str, vision_prefix: str | None = None) -> dict:
    """Non-streaming OpenAI chat completion dict -> Anthropic message dict."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict] = []
    if vision_prefix:
        content.insert(0, {"type": "thinking", "thinking": vision_prefix})
    # deepseek thinking 模式的 reasoning_content -> Anthropic thinking 块，
    # 保证客户端下一轮回传完整思考，命中前缀缓存（同流式路径逻辑）。
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        content.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": arguments,
            }
        )
    usage = openai_resp.get("usage") or {}
    return {
        "id": openai_resp.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            # 缓存命中数（deepseek prompt_cache_hit_tokens -> Anthropic cache_read_input_tokens）
            "cache_read_input_tokens": usage.get("prompt_cache_hit_tokens", 0),
        },
    }


def anthropic_error(status: int, message: str, etype: str | None = None) -> dict:
    """Anthropic-style error body."""
    if etype is None:
        etype = {
            401: "authentication_error",
            400: "invalid_request_error",
            404: "not_found_error",
            429: "rate_limit_error",
            500: "api_error",
            502: "api_error",
        }.get(status, "api_error")
    return {"type": "error", "error": {"type": etype, "message": message}}


async def anthropic_sse(chunk_iter, model: str, vision_prefix: str | None = None):
    """Translate an OpenAI chunk async iterable into Anthropic SSE events.

    Yields raw SSE lines ("event: ...\n...\n\n" style consumed by clients).
    """
    import json as _json

    sent_start = False
    text_index = None
    reasoning_index = None  # deepseek reasoning_content -> Anthropic thinking block
    tool_indices: dict[int, int] = {}  # deepseek tc.index -> anthropic block index
    block_counter = 1 if vision_prefix else 0
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    stop_reason = None
    blocks_done = False

    def evt(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

    async for chunk in chunk_iter:
        # deepseek attaches usage to the FINAL chunk (choices may be non-empty);
        # OpenAI-standard "empty choices + usage" chunks are handled below too.
        if chunk.usage:
            usage = {
                "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                "cache_read_input_tokens": getattr(
                    chunk.usage, "prompt_cache_hit_tokens", 0
                ),
            }
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if not sent_start:
            yield evt(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": chunk.id or "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                    },
                },
            )
            if vision_prefix:
                yield evt(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
                for i in range(0, len(vision_prefix), 2000):
                    yield evt(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": vision_prefix[i : i + 2000],
                            },
                        },
                    )
                yield evt(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
            sent_start = True

        if delta:
            # deepseek thinking 模式下，reasoning_content 先于 content/tool_calls 流出。
            # 转成 Anthropic thinking 块回传，客户端下一轮才能完整带回，
            # 命中 deepseek 前缀缓存（否则回传空 reasoning -> 缓存全 miss）。
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if reasoning_index is None:
                    reasoning_index = block_counter
                    block_counter += 1
                    yield evt(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": reasoning_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )
                yield evt(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": reasoning_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )
            if delta.content:
                if text_index is None:
                    text_index = block_counter
                    block_counter += 1
                    yield evt(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield evt(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": delta.content},
                    },
                )
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    src_idx = tc.index or 0
                    fn = tc.function or {}
                    if src_idx not in tool_indices:
                        tool_indices[src_idx] = block_counter
                        block_counter += 1
                        yield evt(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_indices[src_idx],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.id or f"toolu_{src_idx}",
                                    "name": fn.name or "",
                                    "input": {},
                                },
                            },
                        )
                    if fn.arguments:
                        yield evt(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_indices[src_idx],
                                "delta": {"type": "input_json_delta", "partial_json": fn.arguments},
                            },
                        )

        if choice.finish_reason:
            stop_reason = _stop_reason(choice.finish_reason)
            # Anthropic requires a content_block_stop after each block's deltas.
            if not blocks_done:
                if reasoning_index is not None:
                    yield evt(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": reasoning_index},
                    )
                if text_index is not None:
                    yield evt(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_index},
                    )
                for idx in sorted(tool_indices.values()):
                    yield evt(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                blocks_done = True

    if not sent_start:
        # Empty stream: emit a minimal message so clients don't hang.
        yield evt(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_empty",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                },
            },
        )
    elif not blocks_done:
        # Stream ended without an explicit finish_reason: close open blocks.
        if reasoning_index is not None:
            yield evt(
                "content_block_stop",
                {"type": "content_block_stop", "index": reasoning_index},
            )
        if text_index is not None:
            yield evt(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_index},
            )
        for idx in sorted(tool_indices.values()):
            yield evt(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )
    yield evt(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
            "usage": usage,
        },
    )
    yield evt("message_stop", {"type": "message_stop"})
