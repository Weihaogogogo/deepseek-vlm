"""Anthropic /v1/messages route: reuses the OpenAI-format routing core.

Request flow: parse Anthropic body -> OpenAI-format messages ->
no-image: passthrough to deepseek (with Anthropic system prepended)
image: dual-VLM + merge + LLM_SYSTEM injection -> forward.
Response flow: OpenAI response -> Anthropic message / SSE events.
"""
import asyncio
import logging

from fastapi.responses import JSONResponse, StreamingResponse

from . import anthropic_protocol as ap
from . import config, image_utils, merger
from .anthropic_protocol import anthropic_error, anthropic_sse, to_anthropic_message
from .image_utils import ImageParseError
from .llm_client import LLMBackendError, DeepSeekClient
from .router import (
    PROMPTS_DIR,
    ClientRequestError,
    VisionUnavailable,
    _ensure_reasoning_content,
    _extract_current_images,
    _parse_content,
    _pick_focus_text,
    _strip_message_images,
    _validate_messages,
)
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)

LLM_SYSTEM = (PROMPTS_DIR / "llm_system.md").read_text(encoding="utf-8")
VLM1_SYSTEM = (PROMPTS_DIR / "vlm1_system.md").read_text(encoding="utf-8")
VLM2_SYSTEM = (PROMPTS_DIR / "vlm2_system.md").read_text(encoding="utf-8")

_llm = DeepSeekClient()
_vlm = VLMClient(config.DASHSCOPE_API_KEY)

_MODEL_NAME = config.DEEPSEEK_MODEL


def _find_last_user_idx(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and messages[i].get("content") is not None:
            return i
    raise ClientRequestError("messages must include at least one user message")


def _prepend_system(messages: list, system_text: str) -> list:
    """Return messages with the Anthropic top-level system as the first system message."""
    if not system_text.strip():
        return messages
    out = [{"role": "system", "content": system_text}]
    # Keep any explicit system messages from the parsed messages after it.
    out.extend(m for m in messages if m.get("role") != "system")
    out.extend(m for m in messages if m.get("role") == "system")
    return out


def _llm_error_response(exc: LLMBackendError):
    """Convert an upstream deepseek error into an Anthropic error body."""
    status = exc.status_code
    try:
        msg = exc.body.get("error", {}).get("message", "upstream error")
    except AttributeError:
        msg = "upstream error"
    return JSONResponse(status_code=status, content=anthropic_error(status, msg))


async def route_messages(body: dict):
    try:
        messages, params, system_text = ap.parse_body(body)
    except ValueError as exc:
        raise ClientRequestError(str(exc)) from exc

    _validate_messages(messages)
    stream = bool(body.get("stream", False))
    model = body.get("model") or _MODEL_NAME

    last_user_idx = _find_last_user_idx(messages)
    cur_images = _extract_current_images(messages)
    cur_text = _pick_focus_text(messages)
    logger.info(
        "anthropic route: msg_count=%d cur_text_len=%d cur_images=%d",
        len(messages),
        len(cur_text),
        len(cur_images),
    )
    if len(cur_images) > 1:
        logger.warning("anthropic: %d images, keeping the first", len(cur_images))
        cur_images = cur_images[:1]

    if not cur_images:
        stripped = []
        for message in messages:
            s = _strip_message_images(message)
            if s is not None:
                stripped.append(s)
        stripped = _ensure_reasoning_content(stripped)
        fwd_messages = _prepend_system(stripped, system_text)
        return await _forward_anthropic(fwd_messages, params, model, stream)

    try:
        data_url = await image_utils.prepare_image(cur_images[0])
    except ImageParseError as exc:
        raise ClientRequestError(f"invalid image: {exc}", code="invalid_image") from exc

    run_vlm2 = len(cur_text.strip()) > 1
    tasks = [_vlm.describe_overall(VLM1_SYSTEM, data_url)]
    if run_vlm2:
        tasks.append(_vlm.describe_focus(VLM2_SYSTEM, data_url, cur_text))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if isinstance(results[0], Exception):
        logger.error("anthropic VLM-1 failed: %s", results[0])
        raise VisionUnavailable from results[0]
    overall = results[0]
    focus = None
    if run_vlm2:
        if isinstance(results[1], Exception):
            logger.error("anthropic VLM-2 failed: %s", results[1])
            raise VisionUnavailable from results[1]
        focus = results[1]

    merged = merger.merge_image_info(overall, focus, cur_text)

    # Cache-friendly assembly: keep the SHARED prefix (system_text + history)
    # identical across image and no-image turns, and put LLM_SYSTEM + merged
    # description at the END as user messages. deepseek's prefix cache then
    # hits on all history tokens on every turn (verified: any change at the
    # start of the request invalidates the whole cache).
    new_messages: list[dict] = []
    if system_text.strip():
        new_messages.append({"role": "system", "content": system_text})
    for i, message in enumerate(messages):
        if message.get("role") == "system":
            continue  # explicit systems appended at the end (never in the shared prefix)
        if i == last_user_idx:
            new_messages.append({"role": "user", "content": LLM_SYSTEM})
            new_messages.append({"role": "user", "content": merged})
            continue
        stripped = _strip_message_images(message)
        if stripped:
            new_messages.append(stripped)
    new_messages.extend(m for m in messages if m.get("role") == "system")

    new_messages = _ensure_reasoning_content(new_messages)
    return await _forward_anthropic(new_messages, params, model, stream)


async def _forward_anthropic(messages: list, params: dict, model: str, stream: bool):
    """Forward to deepseek and translate the response to Anthropic format."""
    body_params = dict(params)
    if stream:
        try:
            chunk_iter = _llm.stream_chunks(messages, body_params)
        except LLMBackendError as exc:
            _log_message_pairs(messages, exc)
            return _llm_error_response(exc)

        async def _stream_with_error_log():
            started = False
            try:
                async for line in anthropic_sse(chunk_iter, model):
                    started = started or "message_start" in line
                    yield line
            except LLMBackendError as exc:
                _log_message_pairs(messages, exc)
                if not started:
                    # Nothing sent yet: emit an Anthropic error event so the
                    # client gets a parseable failure instead of a dead stream.
                    import json as _json

                    yield (
                        "event: error\n"
                        f"data: {_json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': 'upstream error'}}, ensure_ascii=False)}\n\n"
                    )
                    return
                raise

        return StreamingResponse(
            _stream_with_error_log(), media_type="text/event-stream"
        )
    try:
        data = await _llm.complete(messages, body_params)
    except LLMBackendError as exc:
        _log_message_pairs(messages, exc)
        return _llm_error_response(exc)
    return JSONResponse(content=to_anthropic_message(data, model))


def _log_message_pairs(messages: list, exc: Exception) -> None:
    """Log tool_calls/tool pairing structure on deepseek errors."""
    try:
        summary = []
        for i, m in enumerate(messages):
            entry = {
                "i": i,
                "role": m.get("role"),
                "content": (
                    None if m.get("content") is None else type(m.get("content")).__name__
                ),
            }
            if m.get("tool_calls"):
                entry["tool_calls"] = [tc.get("id") for tc in m["tool_calls"]]
            if m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            summary.append(entry)
        logger.warning("deepseek error %s; message pairs: %s", exc, summary)
    except Exception:  # noqa: BLE001
        pass
