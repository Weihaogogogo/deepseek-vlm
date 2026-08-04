"""Deepseek forwarding: SSE passthrough (model rewritten) and JSON passthrough."""
import logging

from openai import APIConnectionError, AsyncOpenAI

from . import config

logger = logging.getLogger(__name__)

DEEPSEEK_TIMEOUT = 300.0

_PASSTHROUGH_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "logprobs",
    "top_logprobs",
    "tools",
    "tool_choice",
    "user",
    "n",
    "stream_options",
}


class LLMBackendError(Exception):
    """Carries the exact upstream status code and error body to passthrough."""

    def __init__(self, status_code: int, body: dict):
        super().__init__(f"deepseek error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class DeepSeekClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
        )
        self.model = config.DEEPSEEK_MODEL

    @staticmethod
    def _passthrough_params(body: dict) -> dict:
        return {k: v for k, v in body.items() if k in _PASSTHROUGH_PARAMS and v is not None}

    @staticmethod
    def _extract_error(exc: Exception) -> tuple[int, dict]:
        if isinstance(exc, APIConnectionError):
            return 502, {
                "error": {
                    "message": "deepseek backend unavailable",
                    "type": "api_error",
                    "code": "deepseek_connection_error",
                }
            }
        status = getattr(exc, "status_code", None) or 500
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                return status, response.json()
            except Exception:
                return status, {
                    "error": {
                        "message": response.text or str(exc),
                        "type": "api_error",
                        "code": "upstream_error",
                    }
                }
        return status, {
            "error": {
                "message": str(exc),
                "type": "api_error",
                "code": "upstream_error",
            }
        }

    @staticmethod
    def _raise_backend(exc: Exception) -> None:
        raise LLMBackendError(*DeepSeekClient._extract_error(exc)) from exc

    @staticmethod
    def _chunk_line(chunk, rewrite_model: str) -> str:
        chunk.model = rewrite_model
        return "data: " + chunk.model_dump_json() + "\n\n"

    async def complete(self, messages: list, body: dict) -> dict:
        """Non-streaming: JSON passthrough (content-identical)."""
        params = self._passthrough_params(body)
        try:
            resp = await self._client.chat.completions.create(
                model=self.model, messages=messages, **params
            )
        except Exception as exc:
            self._raise_backend(exc)
        return resp.model_dump()

    async def stream(self, messages: list, body: dict, rewrite_model: str):
        """Streaming: SSE passthrough with the model field rewritten per chunk.

        Fetches the first chunk eagerly so upstream errors surface before the
        response starts; returns an async generator of SSE lines.
        """
        params = self._passthrough_params(body)
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **params
            )
        except Exception as exc:
            self._raise_backend(exc)

        iterator = stream.__aiter__()
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            first = None
        except Exception as exc:
            await stream.close()
            self._raise_backend(exc)

        async def gen():
            try:
                if first is not None:
                    yield self._chunk_line(first, rewrite_model)
                async for chunk in iterator:
                    yield self._chunk_line(chunk, rewrite_model)
                yield "data: [DONE]\n\n"
            finally:
                await stream.close()

        return gen()
