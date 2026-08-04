"""Routing: no-image passthrough to deepseek; image requests go through dual VLM + merge."""
import asyncio
import logging
from pathlib import Path

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
    """Images belonging to the CURRENT turn only.

    - Images in the last user message (user just attached them), else
    - Images in the last message IF that message is a tool result (agent just
      Read a file this turn).
    History images (earlier turns) are intentionally ignored: they were
    already processed when first sent; re-processing them on every later
    text-only turn is what made follow-ups slow.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if content is None:
                break
            _, imgs = _parse_content(content)
            if imgs:
                return imgs[-1:]
            break
    if messages and messages[-1].get("role") == "tool" and isinstance(
        messages[-1].get("content"), list
    ):
        _, imgs = _parse_content(messages[-1].get("content"))
        if imgs:
            return imgs[-1:]
    return []


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
    """
    content = message.get("content")
    if content is None:
        return message if message.get("tool_calls") else None
    if isinstance(content, str):
        return message
    if isinstance(content, list):
        kept = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]
        if not kept:
            if message.get("role") == "tool":
                # Tool messages MUST survive even empty: they pair with the
                # preceding assistant tool_calls. deepseek rejects orphan
                # tool_calls (assistant tool_calls without matching tool msg).
                return {**message, "content": ""}
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


async def _forward(messages: list, body: dict, model: str, stream: bool):
    if stream:
        gen = await _llm.stream(messages, body, model)
        return StreamingResponse(gen, media_type="text/event-stream")
    data = await _llm.complete(messages, body)
    # Keep the model field consistent with what the client requested (streaming
    # chunks already rewrite it; non-streaming must match).
    data["model"] = model
    return JSONResponse(content=data)


async def route_chat_completions(body: dict):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ClientRequestError("messages must be a non-empty array")
    _validate_messages(messages)

    stream = bool(body.get("stream", False))
    model = body.get("model") or config.DEEPSEEK_MODEL

    last_user_idx = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and m.get("content") is not None
        ),
        None,
    )
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
                    "keys": (
                        list(m["content"][0].keys()) if isinstance(m.get("content"), list) and m["content"] else None
                    ),
                }
                for m in messages
            ]
            logger.info("no image detected; message structure=%s", structure)
        except Exception as exc:  # noqa: BLE001
            logger.warning("structure log failed: %s", exc)
    if len(cur_images) > 1:
        logger.warning(
            "current turn has %d images; keeping the first, discarding the rest",
            len(cur_images),
        )
        cur_images = cur_images[:1]

    if not cur_images:
        stripped = []
        for message in messages:
            s = _strip_message_images(message)
            if s is not None:
                stripped.append(s)
        stripped = _ensure_reasoning_content(stripped)
        return await _forward(stripped, body, model, stream)

    try:
        data_url = await image_utils.prepare_image(cur_images[0])
    except ImageParseError as exc:
        raise ClientRequestError(f"invalid image url: {exc}", code="invalid_image_url") from exc

    run_vlm2 = len(cur_text.strip()) > 1

    async def vlm1_task():
        return await _vlm.describe_overall(VLM1_SYSTEM, data_url)

    async def vlm2_task():
        return await _vlm.describe_focus(VLM2_SYSTEM, data_url, cur_text)

    tasks = [vlm1_task()]
    if run_vlm2:
        tasks.append(vlm2_task())
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if isinstance(results[0], Exception):
        logger.error("VLM-1 failed: %s", results[0])
        raise VisionUnavailable from results[0]
    overall = results[0]

    focus = None
    if run_vlm2:
        if isinstance(results[1], Exception):
            logger.error("VLM-2 failed: %s", results[1])
            raise VisionUnavailable from results[1]
        focus = results[1]

    merged = merger.merge_image_info(overall, focus, cur_text)

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
    return await _forward(new_messages, body, model, stream)
