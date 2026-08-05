"""qwen3-vl-flash calls: VLM-1 overall transcription, VLM-2 focused description."""
import asyncio
import logging

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL = "qwen3-vl-flash"
VLM_TIMEOUT = 120.0

VLM2_HEADER = "# 聚焦描述"
VLM2_RETRY_PROMPT = "你没有按规范输出，请严格遵守系统规范，只输出聚焦描述"

_MAX_CONCURRENCY = 4
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = (1.0, 2.0)

_semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)


class VisionBackendError(Exception):
    """Raised when a VLM call ultimately fails."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


class VLMClient:
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=DASHSCOPE_BASE_URL, timeout=VLM_TIMEOUT
        )

    @staticmethod
    def _image_part(data_url: str) -> dict:
        return {"type": "image_url", "image_url": {"url": data_url}}

    @staticmethod
    def _text_part(text: str) -> dict:
        return {"type": "text", "text": text}

    async def _complete(self, system_prompt: str, user_content: list) -> str:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with _semaphore:
                    resp = await self._client.chat.completions.create(
                        model=VLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.5,
                    )
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    raise VisionBackendError("vision backend returned empty content")
                return content
            except VisionBackendError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                    break
                logger.warning(
                    "VLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    _RETRY_BACKOFF_SECONDS[attempt],
                    exc,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS[attempt])
        assert last_exc is not None
        logger.error("VLM call failed: %s", last_exc)
        raise VisionBackendError(f"vision backend error: {last_exc}") from last_exc

    async def describe_overall(self, system_prompt: str, data_url: str) -> str:
        """VLM-1: image only, no text."""
        return await self._complete(system_prompt, [self._image_part(data_url)])

    async def describe_focus(self, system_prompt: str, data_url: str, question: str) -> str:
        """VLM-2: image + wrapped context; retries once if the header is missing."""
        text = f"对话上下文（最新用户提问在末尾）：\n{question}\n请按系统规范输出聚焦描述。"
        content = await self._complete(
            system_prompt, [self._image_part(data_url), self._text_part(text)]
        )
        if VLM2_HEADER not in content:
            logger.warning("VLM-2 output missing '%s', retrying once", VLM2_HEADER)
            text = f"{text}\n{VLM2_RETRY_PROMPT}"
            content = await self._complete(
                system_prompt, [self._image_part(data_url), self._text_part(text)]
            )
            if VLM2_HEADER not in content:
                logger.warning("VLM-2 retry still missing '%s'", VLM2_HEADER)
        return content
