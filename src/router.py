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


def _strip_message_images(message: dict):
    """Returns the message with image parts removed, or None if it becomes empty."""
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
        if kept:
            return {**message, "content": kept}
        logger.warning(
            "dropping history message (role=%s) whose content was only an image",
            message.get("role"),
        )
        return None
    return message


def _validate_messages(messages: list) -> None:
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ClientRequestError("each message must be an object with a role")
        if message["role"] not in VALID_ROLES:
            raise ClientRequestError(f"unsupported role: {message['role']}")
        if message.get("content") is None and "tool_calls" not in message:
            raise ClientRequestError("message missing content")
        if message.get("content") is not None:
            _parse_content(message["content"])


async def _forward(messages: list, body: dict, model: str, stream: bool):
    if stream:
        gen = await _llm.stream(messages, body, model)
        return StreamingResponse(gen, media_type="text/event-stream")
    data = await _llm.complete(messages, body)
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

    cur_text, cur_images = _parse_content(messages[last_user_idx].get("content"))
    if len(cur_images) > 1:
        logger.warning(
            "current turn has %d images; keeping the first, discarding the rest",
            len(cur_images),
        )
        cur_images = cur_images[:1]

    if not cur_images:
        # No image in the current turn: forward untouched, no system injection.
        return await _forward(messages, body, model, stream)

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

    new_messages: list[dict] = [{"role": "system", "content": LLM_SYSTEM}]
    for i, message in enumerate(messages):
        if i == last_user_idx:
            new_messages.append({"role": "user", "content": merged})
            continue
        if message.get("role") == "system":
            stripped = _strip_message_images(message)
            if stripped:
                new_messages.append(stripped)
        else:
            stripped = _strip_message_images(message)
            if stripped:
                new_messages.append(stripped)

    return await _forward(new_messages, body, model, stream)
