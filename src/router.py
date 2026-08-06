"""Routing: no-image passthrough to deepseek; image requests go through dual VLM + merge."""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
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

# VLM 描述缓存：'图片hash|问题指纹' -> {"overall": VLM-1输出, "focus": VLM-2输出}。
# 有图轮存，无图轮查（历史图片跨轮保留视觉信息，agent 工具读图场景必需）。
# 键含问题指纹：VLM-2 的 focus 是问题驱动的，同一张图不同问题必须命中不同条目。
_DESC_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_DESC_CACHE_MAX = 200
_MAX_IMAGES_PER_TURN = 10


def _focus_fingerprint(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def _desc_cache_key(image_key: str, question: str) -> str:
    return f"{image_key}|{_focus_fingerprint(question)}"


def _find_cached_overall(image_key: str) -> str | None:
    """Overall description of the most recently cached entry for an image,
    regardless of question (used by tool-result backfill, which has no text)."""
    prefix = f"{image_key}|"
    for k in reversed(_DESC_CACHE):
        if k.startswith(prefix):
            return _DESC_CACHE[k].get("overall") or None
    return None


def _cache_desc(key: str, overall: str, focus: str | None) -> None:
    _DESC_CACHE[key] = {"overall": overall, "focus": focus}
    _DESC_CACHE.move_to_end(key)
    while len(_DESC_CACHE) > _DESC_CACHE_MAX:
        _DESC_CACHE.popitem(last=False)


def _find_cached_history_image(messages: list, question: str) -> str | None:
    """Merged description (overall + focus + current question) of the MOST
    RECENT history image, if its VLM output was cached."""
    for m in reversed(messages):
        if m.get("role") not in ("user", "tool"):
            continue
        content = m.get("content")
        if content is None:
            continue
        _, imgs = _parse_content(content)
        if not imgs:
            continue
        for url in reversed(imgs):
            key = _desc_cache_key(image_utils.image_hash(url), question)
            if key in _DESC_CACHE:
                cached = _DESC_CACHE[key]
                return merger.merge_image_info(
                    cached.get("overall", ""), cached.get("focus"), question
                )
        return None  # has images but none cached; stop at the newest one
    return None


def _inject_history_description(stripped: list, merged: str) -> list:
    """Insert LLM_SYSTEM + cached description right BEFORE the last user
    message, mirroring the image-turn layout so the shared prefix stays
    identical (deepseek prefix cache) and the model gets vision context."""
    out = list(stripped)
    insert_pos = len(out)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            insert_pos = i
            break
    out.insert(insert_pos, {"role": "user", "content": LLM_SYSTEM})
    out.insert(insert_pos + 1, {"role": "user", "content": merged})
    return out


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
    for m in reversed(messages):
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
    if messages and messages[-1].get("role") == "tool" and isinstance(
        messages[-1].get("content"), list
    ):
        _, imgs = _parse_content(messages[-1].get("content"))
        if imgs:
            return imgs[-_MAX_IMAGES_PER_TURN:]
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


def _strip_message_images(message: dict):
    """Returns the message with image parts removed, or None if it becomes empty.

    Text-only arrays are flattened to plain strings (deepseek's compatibility
    layer rejects array content on tool/assistant messages).

    Tool messages whose content was ONLY an image get the cached VLM
    description written in as their text — otherwise deepseek sees an empty
    tool result and concludes "read_file returned nothing" (observed from
    Claude Code reading PNGs: its tool_result is a bare image block).
    """
    content = message.get("content")
    if content is None:
        return message if message.get("tool_calls") else None
    if isinstance(content, str):
        return message
    if isinstance(content, list):
        img_keys = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url")
                u = url.get("url") if isinstance(url, dict) else None
                if isinstance(u, str) and u:
                    img_keys.append(image_utils.image_hash(u))
        kept = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]
        if not kept:
            if message.get("role") == "tool":
                # Image-only tool result: fill with the cached overall
                # description (no question is available for a focus lookup) so
                # the backend sees content, not an empty tool message.
                for key in img_keys:
                    overall = _find_cached_overall(key)
                    if overall:
                        return {**message, "content": overall}
                return {
                    **message,
                    "content": "【图片】此消息包含一张图片，内容未经视觉解析（缓存未命中）",
                }
            logger.warning(
                "dropping history message (role=%s) whose content was only an image",
                message.get("role"),
            )
            return None
        if all(isinstance(p, dict) and p.get("type") == "text" for p in kept):
            texts = [p.get("text", "") for p in kept]
            return {**message, "content": "".join(texts)}
        return {**message, "content": kept}
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
        cached = _find_cached_history_image(messages, cur_text)
        if cached is not None:
            stripped = _inject_history_description(stripped, cached)
            logger.info("history image description injected (len=%d)", len(cached))
        stripped = _ensure_reasoning_content(stripped)
        return await _forward(stripped, body, model, stream)

    # Cache lookup BEFORE any download: per-image key needs only the URL
    # (data URLs hash their raw bytes, http URLs hash the url string), so
    # cache hits skip download + compression + VLM entirely.
    run_vlm2 = len(cur_text.strip()) > 1

    per_image: list[dict | None] = []
    pending_urls: list[str] = []  # urls that missed the cache and need download
    for url in cur_images:
        cache_key = _desc_cache_key(image_utils.image_hash(url), cur_text)
        cached = _DESC_CACHE.get(cache_key)
        if cached is not None:
            per_image.append(cached)
        else:
            per_image.append(None)
            pending_urls.append(url)

    if pending_urls:
        try:
            data_urls = await asyncio.gather(
                *[image_utils.prepare_image(url) for url in pending_urls]
            )
        except ImageParseError as exc:
            raise ClientRequestError(
                f"invalid image url: {exc}", code="invalid_image_url"
            ) from exc

        async def vlm_pair(data_url: str) -> tuple[str, str | None]:
            if run_vlm2:
                overall, focus = await asyncio.gather(
                    _vlm.describe_overall(VLM1_SYSTEM, data_url),
                    _vlm.describe_focus(VLM2_SYSTEM, data_url, cur_text),
                )
                return overall, focus
            overall = await _vlm.describe_overall(VLM1_SYSTEM, data_url)
            return overall, None

        pair_results = await asyncio.gather(
            *[vlm_pair(d) for d in data_urls], return_exceptions=True
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
            overall, focus = res
            cached = {"overall": overall, "focus": focus}
            per_image[i] = cached
            _cache_desc(
                _desc_cache_key(
                    image_utils.image_hash(pending_urls[pi - 1]), cur_text
                ),
                overall,
                focus,
            )

    merged = merger.merge_multi_image(per_image, cur_text)

    # Cache-friendly assembly: keep the shared prefix (system messages +
    # history) identical across image and no-image turns; LLM_SYSTEM goes at
    # the END as a user message so deepseek's prefix cache hits on history.
    new_messages: list[dict] = []
    for i, message in enumerate(messages):
        if i == last_user_idx:
            new_messages.append({"role": "user", "content": LLM_SYSTEM})
            new_messages.append({"role": "user", "content": merged})
            continue
        if message.get("role") == "system":
            new_messages.append(message)
            continue
        stripped = _strip_message_images(message)
        if stripped:
            new_messages.append(stripped)

    new_messages = _ensure_reasoning_content(new_messages)
    return await _forward(new_messages, body, model, stream, vision_prefix=merged)
