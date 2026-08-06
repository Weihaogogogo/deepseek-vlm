"""Routing: no-image passthrough to deepseek; image requests go through dual VLM + merge."""
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, StreamingResponse

from . import config, image_utils, merger
from .image_utils import ImageParseError
from .llm_client import DeepSeekClient
from .vlm_client import VLMClient, VisionBackendError

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
VALID_ROLES = ("system", "user", "assistant", "tool")


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


VLM1_SYSTEM = _read_prompt("vlm1_system.md")
VLM2_SYSTEM = _read_prompt("vlm2_system.md")
VLM3_SYSTEM = _read_prompt("vlm3_system.md")
LLM_SYSTEM = _read_prompt("llm_system.md")

_llm = DeepSeekClient()
_vlm = VLMClient(config.DASHSCOPE_API_KEY)


class ClientRequestError(Exception):
    def __init__(self, message: str, code: str = "invalid_messages"):
        super().__init__(message)
        self.message = message
        self.code = code


class VisionUnavailable(Exception):
    pass


def _parse_content(content) -> tuple[str, list[str]]:
    """Returns (text, image_urls) from a string or content-array message content."""
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "image_url":
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    images.append(url)
            elif ptype is None:
                if "text" in part:
                    text_parts.append(part.get("text", ""))
                elif "image_url" in part:
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if isinstance(url, str) and url:
                        images.append(url)
        return "".join(text_parts), images
    raise ClientRequestError("invalid message content")


FOCUS_BUDGET_CHARS = 1000
_MAX_IMAGES_PER_TURN = 10


def _pick_focus_text(messages: list) -> str:
    """Recent conversational context for VLM-2 focus.

    Scans user + assistant messages newest-first, accumulating text until the
    char budget is exhausted, then returns the collected context in
    chronological order with role prefixes. This covers agent self-talk — an
    assistant message like "I need to check image X before deciding" carries
    the real intent of the current agent loop, which the latest user message
    alone may not express. Tool messages never contribute (tool_result blocks
    are role=tool after parsing).
    """
    chunks: list[tuple[bool, str]] = []  # (is_user, text), newest first
    total = 0
    for m in reversed(messages):
        if m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content")
        if content is None:
            continue  # assistant tool_calls messages have no content
        t, _ = _parse_content(content)
        t = t.strip()
        if not t:
            continue
        if chunks and total + len(t) > FOCUS_BUDGET_CHARS:
            break  # budget exhausted; do not scan further back
        chunks.append((m.get("role") == "user", t))
        total += len(t)
    if not chunks:
        return ""
    chunks.reverse()  # chronological order
    parts = [f"用户: {t}" if is_user else f"助手: {t}" for is_user, t in chunks]
    return "\n".join(parts)


def _current_question_text(messages: list, last_user_idx: int | None = None) -> str:
    """Text of the CURRENT user question only — no conversation history.

    VLM-3 (judgment) must see just the current image + current question so its
    first-intuition judgment is not polluted by prior turns. Unlike
    _pick_focus_text (VLM-2), no user/assistant history is scanned.
    """
    if last_user_idx is None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user" and messages[i].get("content") is not None:
                last_user_idx = i
                break
    if last_user_idx is None:
        return ""
    t, _ = _parse_content(messages[last_user_idx].get("content"))
    return (t or "").strip()


def _extract_current_images(messages: list) -> list[str]:
    """Images belonging to the CURRENT turn only, in original sending order,
    deduplicated by content hash, max 10.

    - Collects images from user messages scanning backwards; skips user
      messages without images (clients like WorkBuddy split image messages
      and the text message into separate user messages).
    - Stops at the first non-user message (assistant/system/tool) — that is
      the "current turn" boundary: everything after the last assistant reply
      belongs to this request, everything before it is history.
    - After collecting newest-first, reverses to the original sending order
      (the earliest image is numbered 1 in merge_multi_image), then dedupes
      by image_hash keeping each image's FIRST occurrence.
    - Falls back to the last message IF it is a tool result (agent just Read
      a file this turn).
    History images (earlier turns) are intentionally ignored.
    """
    collected: list[str] = []
    # [diag] 全序列扫描：找出所有带图片的 tool 消息及其位置，判断
    # Claude Code Read 的图片到底在不在、在哪个位置、什么形态。
    if messages:
        img_tools = []
        for idx, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            c = m.get("content")
            if isinstance(c, list):
                types = [p.get("type") for p in c if isinstance(p, dict)]
                has_img = any(p.get("type") == "image_url" for p in c if isinstance(p, dict))
                if has_img:
                    img_tools.append(f"#{idx}[list:{types}]")
            elif isinstance(c, str):
                if "image" in c[:120].lower() or "【图片】" in c:
                    img_tools.append(f"#{idx}[str:{c[:50]}]")
        tail = messages[-3:]
        diag = []
        for m in tail:
            c = m.get("content")
            if isinstance(c, list):
                types = [p.get("type") for p in c if isinstance(p, dict)]
                diag.append(f"{m.get('role')}[list:{types}]")
                # [diag] user 消息里的图片 url 前缀，确认 Claude Code 发的是
                # 什么形态（data:/file:/http:）
                if m.get("role") == "user":
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "image_url":
                            u = p.get("image_url") or {}
                            url = u.get("url", "") if isinstance(u, dict) else ""
                            diag.append(f"  img_url_prefix={str(url)[:80]}")
            else:
                diag.append(f"{m.get('role')}[str:{str(c)[:60]}]")
        logger.info(
            "[diag-extract] total=%d tail=%s img_tools=%s",
            len(messages),
            " | ".join(diag),
            img_tools if img_tools else "无",
        )
    for m in reversed(messages):
        if m.get("role") == "system":
            # Claude Code 会在消息末尾注入 system 提示（如 ToolSearch 说明），
            # 不能因此中断扫描——跳过继续往前找 user 消息里的图片。
            continue
        if m.get("role") != "user":
            break
        content = m.get("content")
        if content is None:
            continue
        _, imgs = _parse_content(content)
        if imgs:
            collected.extend(reversed(imgs))
    if collected:
        collected.reverse()  # original sending order
        seen: set[str] = set()
        unique: list[str] = []
        for url in collected:
            h = image_utils.image_hash(url)
            if h in seen:
                continue
            seen.add(h)
            unique.append(url)
        return unique[-_MAX_IMAGES_PER_TURN:]
    # tool 图片回退：Claude Code 的 agent 循环里，Read 的 tool_result 图片
    # 不一定正好是最后一条（后面可能跟着 assistant 回复或新的 user 消息），
    # 但图片属于"最后一个用户请求"的 agent 执行段。从最后往前收集 tool
    # 消息里的图片，遇到倒数第二个 user 消息（即上一个用户回合）就停。
    for m in reversed(messages):
        if m.get("role") == "system":
            continue
        if m.get("role") == "user":
            # 最后一个 user 之后的 tool 图片已收集完；再往前就是历史回合
            break
        if m.get("role") == "tool" and isinstance(m.get("content"), list):
            _, imgs = _parse_content(m.get("content"))
            if imgs:
                collected.extend(reversed(imgs))
    if collected:
        collected.reverse()  # original sending order
        seen: set[str] = set()
        unique: list[str] = []
        for url in collected:
            h = image_utils.image_hash(url)
            if h in seen:
                continue
            seen.add(h)
            unique.append(url)
        return unique[-_MAX_IMAGES_PER_TURN:]
    else:
        # [diag] 最后一段没有 tool 图片时说明原因
        last = messages[-1] if messages else None
        if last is not None:
            logger.info(
                "[diag-extract] no tool images in current segment, last role=%s type=%s",
                last.get("role"),
                type(last.get("content")).__name__,
            )
    return []


def _normalize_tool_pairing(messages: list) -> list:
    """deepseek requires tool messages to IMMEDIATELY follow the assistant
    tool_calls they answer. Anthropic clients may interleave user text
    messages between tool_use and tool_result (observed from Claude Code);
    reorder so each tool message sits right after its matching assistant.
    Unmatched tool messages are appended at the end as a fallback.
    """
    tools_by_id: dict[str, list] = {}
    tool_order: list[str] = []
    others: list[dict] = []
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            tools_by_id.setdefault(m["tool_call_id"], []).append(m)
            tool_order.append(m["tool_call_id"])
        else:
            others.append(m)
    out: list[dict] = []
    used: set[str] = set()
    for m in others:
        out.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id")
                if tid in tools_by_id and tid not in used:
                    out.extend(tools_by_id[tid])
                    used.add(tid)
    for tid in tool_order:
        if tid not in used:
            out.extend(tools_by_id[tid])
    return out


def _ensure_reasoning_content(messages: list) -> list:
    """deepseek thinking mode requires assistant tool_calls messages to carry
    reasoning_content back. Anthropic-format history has no such field, so pad
    it with an empty string when missing (empty string is accepted)."""
    out = []
    for m in messages:
        if (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and "reasoning_content" not in m
        ):
            out.append({**m, "reasoning_content": ""})
        else:
            out.append(m)
    return out


def _strip_message_images(message: dict, current_image_urls: list[str] | None = None):
    """Returns the message with image parts replaced by placeholders, or None
    if it becomes empty.

    - user/assistant history images -> "[历史图片]" text parts (order kept)
    - current-turn user images -> "[图片 N]" text parts (N = index in
      current_image_urls + 1, matching the merged block order)
    - pure-image user messages are kept (placeholder text is non-empty)
    - text-only arrays are flattened to plain strings (deepseek's
      compatibility layer rejects array content on tool/assistant messages)

    Tool messages keep the existing backfill logic: image parts are dropped,
    and an image-only tool result gets the cached VLM overall description or
    a placeholder written in as its text — that is the agent read-image
    scenario, not part of the placeholder system.
    """
    content = message.get("content")
    if content is None:
        return message if message.get("tool_calls") else None
    if isinstance(content, str):
        return message
    if isinstance(content, list):
        if message.get("role") == "tool":
            kept = [
                part
                for part in content
                if not (isinstance(part, dict) and part.get("type") == "image_url")
            ]
            if not kept:
                # Image-only tool result (history): no cache exists (stateless).
                # Write a NEUTRAL placeholder so the backend sees content, not an
                # empty tool message (empty -> deepseek concludes "read_file
                # returned nothing" and the agent re-reads in a loop). The
                # description itself lives in the assistant reasoning_content /
                # thinking block of the same turn, so the placeholder must NOT
                # claim the image was unparsed (that poisons the LLM's belief
                # about what it has seen).
                return {
                    **message,
                    "content": "【图片】此消息包含一张图片",
                }
            if all(isinstance(p, dict) and p.get("type") == "text" for p in kept):
                texts = [p.get("text", "") for p in kept]
                return {**message, "content": "".join(texts)}
            return {**message, "content": kept}
        had_image = False
        replaced = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url")
                u = url.get("url") if isinstance(url, dict) else None
                if isinstance(u, str) and u:
                    had_image = True
                    if current_image_urls is not None and u in current_image_urls:
                        n = current_image_urls.index(u) + 1
                        replaced.append({"type": "text", "text": f"[图片 {n}]"})
                    else:
                        replaced.append({"type": "text", "text": "[历史图片]"})
                continue
            replaced.append(part)
        if not replaced:
            return None
        if (
            not had_image
            and all(isinstance(p, dict) and p.get("type") == "text" for p in replaced)
        ):
            texts = [p.get("text", "") for p in replaced]
            return {**message, "content": "".join(texts)}
        return {**message, "content": replaced}
    return message


def _parse_stream(value) -> bool:
    """Accept 'false'/'0' strings as False (clients send stream as a string)."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0")
    return bool(value)


def _validate_messages(messages: list) -> None:
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ClientRequestError(f"each message must be an object with a role (index {i})")
        if message["role"] not in VALID_ROLES:
            raise ClientRequestError(f"unsupported role: {message['role']} (index {i})")
        if message.get("content") is None and "tool_calls" not in message:
            raise ClientRequestError(f"message missing content (index {i})")
        if message.get("content") is not None:
            try:
                _parse_content(message["content"])
            except ClientRequestError as exc:
                raise ClientRequestError(
                    f"invalid message content at index {i} role={message['role']} "
                    f"ctype={type(message['content']).__name__}: {exc}"
                ) from exc


async def _forward(
    messages: list,
    body: dict,
    model: str,
    stream: bool,
    vision_prefix: str | None = None,
):
    messages = _normalize_tool_pairing(messages)
    if stream:
        gen = await _llm.stream(messages, body, model)
        if vision_prefix:
            # VLM 描述先进 reasoning_content：客户端把它当思考内容，
            # 描述由此进入 harness 上下文成为跨轮资产（界面不直接展示）。
            async def gen_with_prefix(upstream=gen):
                prefix_chunk = {
                    "id": "chatcmpl-" + uuid.uuid4().hex[:16],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": vision_prefix,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield "data: " + json.dumps(prefix_chunk, ensure_ascii=False) + "\n\n"
                async for line in upstream:
                    yield line

            gen = gen_with_prefix()
        return StreamingResponse(gen, media_type="text/event-stream")
    data = await _llm.complete(messages, body)
    # Keep the model field consistent with what the client requested (streaming
    # chunks already rewrite it; non-streaming must match).
    data["model"] = model
    if vision_prefix:
        msg = data.get("choices", [{}])[0].get("message", {})
        if msg:
            old = msg.get("reasoning_content") or ""
            msg["reasoning_content"] = (
                vision_prefix + "\n\n" + old if old else vision_prefix
            )
    return JSONResponse(content=data)


async def route_chat_completions(body: dict):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ClientRequestError("messages must be a non-empty array")
    _validate_messages(messages)

    stream = _parse_stream(body.get("stream", False))
    model = body.get("model")
    if model is None or model == "":
        model = config.PUBLIC_MODEL_NAME
    elif not isinstance(model, str):
        raise ClientRequestError("model must be a string", code="invalid_model")

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if (
            messages[i].get("role") == "user"
            and messages[i].get("content") is not None
        ):
            last_user_idx = i
            break
    if last_user_idx is None:
        raise ClientRequestError("messages must include at least one user message")

    cur_images = _extract_current_images(messages)
    cur_text = _pick_focus_text(messages)
    logger.info(
        "route: last_user_idx=%d msg_count=%d cur_text_len=%d cur_images=%d",
        last_user_idx,
        len(messages),
        len(cur_text),
        len(cur_images),
    )
    if cur_images:
        try:
            layout = [
                {
                    "i": i,
                    "role": m.get("role"),
                    "ctype": type(m.get("content")).__name__ if m.get("content") is not None else "None",
                    "txt": (
                        (m["content"][:60] if isinstance(m.get("content"), str) else "")
                        if isinstance(m.get("content"), str)
                        else None
                    ),
                }
                for i, m in enumerate(messages)
            ]
            logger.info(
                "image url: %s... hash=%s layout=%s",
                cur_images[0][:60],
                image_utils.image_hash(cur_images[0])[:16],
                layout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("image url log failed: %s", exc)
    if not cur_images:
        # Log message content structure to diagnose client image formats.
        try:
            structure = [
                {
                    "role": m.get("role"),
                    "content_type": (
                        type(m.get("content")).__name__
                        if m.get("content") is not None
                        else None
                    ),
                    "part_types": (
                        [p.get("type") for p in m["content"] if isinstance(p, dict)]
                        if isinstance(m.get("content"), list)
                        else None
                    ),
                    "img_part": (
                        [
                            str(p)[:150]
                            for p in m["content"]
                            if isinstance(p, dict) and p.get("type") == "image_url"
                        ][:1]
                        if isinstance(m.get("content"), list)
                        else None
                    ),
                    "keys": (
                        list(m["content"][0].keys()) if isinstance(m.get("content"), list) and m["content"] else None
                    ),
                }
                for m in messages
            ]
            logger.info("no image detected; message structure=%s", structure)
        except Exception as exc:  # noqa: BLE001
            logger.warning("structure log failed: %s", exc)
    if not cur_images:
        stripped = []
        for message in messages:
            s = _strip_message_images(message)
            if s is not None:
                stripped.append(s)
        stripped = _ensure_reasoning_content(stripped)
        return await _forward(stripped, body, model, stream)

    run_vlm2 = len(cur_text.strip()) > 1

    per_image: list[dict | None] = []
    for url in cur_images:
        per_image.append(None)
    pending_urls = list(cur_images)

    if pending_urls:
        try:
            data_urls = await asyncio.gather(
                *[image_utils.prepare_image(url) for url in pending_urls]
            )
        except ImageParseError as exc:
            raise ClientRequestError(
                f"invalid image url: {exc}", code="invalid_image_url"
            ) from exc

        async def vlm_pair(data_url: str, k: int) -> tuple[str, str | None, str | None]:
            # 告知 VLM 本轮图片总数与当前序号：防止它看到问题文本提到其他图
            # 就误报"缺失"（每张图独立调用，VLM 只能看到自己这一张）。
            total = len(cur_images)
            focus_q = f"本轮共 {total} 张图，你正在看第 {k} 张（仅此一张）。用户问题：{cur_text}"
            judgment_q = (
                f"本轮共 {total} 张图，你正在看第 {k} 张（仅此一张）。用户问题："
                f"{_current_question_text(messages, last_user_idx)}"
            )
            if run_vlm2:
                results = await asyncio.gather(
                    _vlm.describe_overall(VLM1_SYSTEM, data_url),
                    _vlm.describe_focus(VLM2_SYSTEM, data_url, focus_q),
                    _vlm.describe_judgment(VLM3_SYSTEM, data_url, judgment_q),
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, Exception):
                        raise VisionUnavailable(str(r)) from r
                return results[0], results[1], results[2]
            results = await asyncio.gather(
                _vlm.describe_overall(VLM1_SYSTEM, data_url),
                _vlm.describe_judgment(VLM3_SYSTEM, data_url, judgment_q),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    raise VisionUnavailable(str(r)) from r
            return results[0], None, results[1]

        pair_results = await asyncio.gather(
            *[vlm_pair(d, k + 1) for k, d in enumerate(data_urls)],
            return_exceptions=True,
        )
        pi = 0
        for i, item in enumerate(per_image):
            if item is not None:
                continue
            res = pair_results[pi]
            pi += 1
            if isinstance(res, Exception):
                logger.error("VLM failed: %s", res)
                raise VisionUnavailable from res
            overall, focus, judgment = res
            per_image[i] = {"overall": overall, "focus": focus, "judgment": judgment}

    merged_blocks = merger.merge_multi_image(per_image)

    # Cache-friendly assembly: keep the shared prefix (system messages +
    # history) identical across image and no-image turns; the current user
    # message stays with [图片 N] placeholders, then merged blocks, then
    # LLM_SYSTEM at the END as a user message so deepseek's prefix cache hits
    # on history.
    new_messages: list[dict] = []
    for i, message in enumerate(messages):
        if i == last_user_idx:
            stripped = _strip_message_images(message, current_image_urls=cur_images)
            if stripped:
                new_messages.append(stripped)  # 保留原消息（问题文本 + [图片 N]）
            continue
        if message.get("role") == "system":
            new_messages.append(message)
            continue
        stripped = _strip_message_images(message)  # 历史：[历史图片]
        if stripped:
            new_messages.append(stripped)
    new_messages.append({"role": "user", "content": merged_blocks})
    new_messages.append({"role": "user", "content": LLM_SYSTEM})

    new_messages = _ensure_reasoning_content(new_messages)
    vision_prefix = "\n\n".join(b["text"] for b in merged_blocks)
    return await _forward(new_messages, body, model, stream, vision_prefix=vision_prefix)
