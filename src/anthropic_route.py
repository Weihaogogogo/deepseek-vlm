"""Anthropic /v1/messages route: reuses the OpenAI-format routing core.

Request flow: parse Anthropic body -> OpenAI-format messages ->
no-image: passthrough to deepseek (with Anthropic system prepended)
image: triple-VLM (overall + focus + judgment) + merge + LLM_SYSTEM injection -> forward.
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
    _current_question_text,
    _extract_current_images,
    _normalize_tool_pairing,
    _parse_content,
    _parse_stream,
    _pick_focus_text,
    _strip_message_images,
    _validate_messages,
)
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)

LLM_SYSTEM = (PROMPTS_DIR / "llm_system.md").read_text(encoding="utf-8")
VLM1_SYSTEM = (PROMPTS_DIR / "vlm1_system.md").read_text(encoding="utf-8")
VLM2_SYSTEM = (PROMPTS_DIR / "vlm2_system.md").read_text(encoding="utf-8")
VLM3_SYSTEM = (PROMPTS_DIR / "vlm3_system.md").read_text(encoding="utf-8")

_llm = DeepSeekClient()
_vlm = VLMClient(config.DASHSCOPE_API_KEY)

_MODEL_NAME = config.PUBLIC_MODEL_NAME


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

    messages = _validate_messages(messages)
    stream = _parse_stream(body.get("stream", False))
    model = body.get("model")
    if model is None or model == "":
        model = _MODEL_NAME
    elif not isinstance(model, str):
        raise ClientRequestError("model must be a string", code="invalid_model")

    last_user_idx = _find_last_user_idx(messages)
    cur_images = _extract_current_images(messages)
    cur_text = _pick_focus_text(messages)
    logger.info(
        "anthropic route: msg_count=%d cur_text_len=%d cur_images=%d",
        len(messages),
        len(cur_text),
        len(cur_images),
    )
    if not cur_images:
        stripped = []
        for message in messages:
            s = _strip_message_images(message)
            if s is not None:
                stripped.append(s)
        stripped = _ensure_reasoning_content(stripped)
        fwd_messages = _prepend_system(stripped, system_text)
        return await _forward_anthropic(fwd_messages, params, model, stream)

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
            raise ClientRequestError(f"invalid image: {exc}", code="invalid_image") from exc

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
                logger.error("anthropic VLM failed: %s", res)
                raise VisionUnavailable from res
            overall, focus, judgment = res
            per_image[i] = {"overall": overall, "focus": focus, "judgment": judgment}

    merged_blocks = merger.merge_multi_image(per_image)

    # Cache-friendly assembly: keep the SHARED prefix (system_text + history)
    # identical across image and no-image turns, keep the current user message
    # with [图片 N] placeholders, and put merged blocks + LLM_SYSTEM at the END
    # as user messages. deepseek's prefix cache then hits on all history tokens
    # on every turn (verified: any change at the start of the request
    # invalidates the whole cache). Explicit systems stay appended at the end
    # (existing behavior).
    new_messages: list[dict] = []
    if system_text.strip():
        new_messages.append({"role": "system", "content": system_text})
    for i, message in enumerate(messages):
        if message.get("role") == "system":
            continue  # explicit systems appended at the end (never in the shared prefix)
        if i == last_user_idx:
            stripped = _strip_message_images(message, current_image_urls=cur_images)
            if stripped:
                new_messages.append(stripped)  # 保留原消息（问题文本 + [图片 N]）
            continue
        stripped = _strip_message_images(message)  # 历史：[历史图片]
        if stripped:
            new_messages.append(stripped)
    new_messages.append({"role": "user", "content": merged_blocks})
    new_messages.append({"role": "user", "content": LLM_SYSTEM})
    new_messages.extend(m for m in messages if m.get("role") == "system")

    new_messages = _ensure_reasoning_content(new_messages)
    vision_prefix = "\n\n".join(b["text"] for b in merged_blocks)
    return await _forward_anthropic(
        new_messages, params, model, stream, vision_prefix=vision_prefix
    )


async def _forward_anthropic(
    messages: list,
    params: dict,
    model: str,
    stream: bool,
    vision_prefix: str | None = None,
):
    """Forward to deepseek and translate the response to Anthropic format."""
    messages = _normalize_tool_pairing(messages)
    body_params = dict(params)
    upstream_model = config.resolve_upstream_model(model)
    if stream:
        try:
            # Claude Code's /context and token accounting depend on usage in
            # the streamed response, so force include_usage=true here.
            chunk_iter = _llm.stream_chunks(
                messages, body_params, force_usage=True, upstream_model=upstream_model
            )
        except LLMBackendError as exc:
            _log_message_pairs(messages, exc)
            return _llm_error_response(exc)

        async def _stream_with_error_log():
            started = False
            try:
                async for line in anthropic_sse(chunk_iter, model, vision_prefix):
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
        data = await _llm.complete(messages, body_params, upstream_model=upstream_model)
    except LLMBackendError as exc:
        _log_message_pairs(messages, exc)
        return _llm_error_response(exc)
    return JSONResponse(content=to_anthropic_message(data, model, vision_prefix))


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
