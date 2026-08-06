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
        # max_retries=0：避免 SDK 重试与网关总预算叠加，失败快速上抛
        # （60s 请求预算由 server.py 中间件兜底）。
        self._client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
            max_retries=0,
        )
        self.model = config.DEEPSEEK_MODEL

    @staticmethod
    def _passthrough_params(
        body: dict, stream: bool = False, force_usage: bool = False
    ) -> dict:
        params = {k: v for k, v in body.items() if k in _PASSTHROUGH_PARAMS and v is not None}
        if not stream:
            # deepseek rejects stream_options on non-streaming calls.
            params.pop("stream_options", None)
            return params
        so = params.get("stream_options")
        if force_usage or so is None:
            # Anthropic path (Claude Code /context, token accounting) and the
            # default OpenAI streaming case both need per-response usage. Only
            # an explicit stream_options.include_usage=false from the OpenAI
            # client is respected as-is (passthrough semantics).
            if so is None:
                params["stream_options"] = {"include_usage": True}
            elif not so.get("include_usage"):
                params["stream_options"] = {**so, "include_usage": True}
        return params

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

    async def stream_chunks(
        self, messages: list, body: dict, force_usage: bool = False
    ):
        """Streaming: returns the raw OpenAI chunk async iterable (for protocol adapters).

        Upstream errors surface before the first chunk is yielded.
        """
        params = self._passthrough_params(body, stream=True, force_usage=force_usage)
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **params
            )
        except Exception as exc:
            self._raise_backend(exc)
        try:
            async for chunk in stream:
                yield chunk
        except Exception as exc:
            await stream.close()
            logger.error("stream interrupted mid-iteration: %s", exc, exc_info=True)
            self._raise_backend(exc)
        finally:
            await stream.close()

    async def stream(self, messages: list, body: dict, rewrite_model: str):
        """Streaming: SSE passthrough with the model field rewritten per chunk.

        Fetches the first chunk eagerly so upstream errors surface before the
        response starts; returns an async generator of SSE lines.
        """
        params = self._passthrough_params(body, stream=True)
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
            except Exception as exc:
                logger.error("stream interrupted mid-iteration: %s", exc, exc_info=True)
                raise
            finally:
                await stream.close()

        return gen()
