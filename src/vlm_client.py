"""qwen3-vl-flash calls: VLM-1 overall transcription, VLM-2 focused description."""
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL = "qwen3-vl-flash"
VLM_TIMEOUT = 120.0

VLM2_HEADER = "# 聚焦描述"
VLM2_RETRY_PROMPT = "你没有按规范输出，请严格遵守系统规范，只输出聚焦描述"


class VisionBackendError(Exception):
    """Raised when a VLM call ultimately fails."""


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
        try:
            resp = await self._client.chat.completions.create(
                model=VLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.5,
            )
        except Exception as exc:
            logger.error("VLM call failed: %s", exc)
            raise VisionBackendError(f"vision backend error: {exc}") from exc
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise VisionBackendError("vision backend returned empty content")
        return content

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
