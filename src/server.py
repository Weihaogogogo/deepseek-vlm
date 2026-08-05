"""FastAPI entry: auth + POST /v1/chat/completions + GET /v1/models."""
import hmac
import logging

import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .llm_client import LLMBackendError
from .anthropic_route import route_messages as route_anthropic_messages
from .router import ClientRequestError, VisionUnavailable, route_chat_completions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="fake-vlm",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    """Reject request bodies over 50MB with a 413 before the route parses them."""
    if request.url.path not in ("/v1/chat/completions", "/v1/messages"):
        return await call_next(request)
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            if request.url.path == "/v1/messages":
                return JSONResponse(
                    status_code=413,
                    content={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "request body too large (max 50MB)",
                        },
                    },
                )
            return _error(
                413,
                "request body too large (max 50MB)",
                "invalid_request_error",
                "payload_too_large",
            )
        chunks.append(chunk)
    # Cache the read body so the route's request.json() does not re-read the stream.
    request._body = b"".join(chunks)
    return await call_next(request)


def _error(status: int, message: str, type_: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
    )


class AuthError(Exception):
    pass


def _check_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    expected = config.FAKE_VLM_API_KEY
    if not expected:
        raise AuthError
    token = ""
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
    else:
        # Anthropic clients often send x-api-key instead of Authorization.
        token = request.headers.get("x-api-key", "")
    if not token or not hmac.compare_digest(token, expected):
        raise AuthError


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError):
    return _error(401, "Invalid API key", "invalid_request_error", "invalid_api_key")


@app.exception_handler(ClientRequestError)
async def _client_error_handler(request: Request, exc: ClientRequestError):
    return _error(400, exc.message, "invalid_request_error", exc.code)


@app.exception_handler(VisionUnavailable)
async def _vision_error_handler(request: Request, exc: VisionUnavailable):
    return _error(502, "vision backend unavailable", "api_error", "vision_backend_unavailable")


@app.exception_handler(LLMBackendError)
async def _llm_error_handler(request: Request, exc: LLMBackendError):
    return JSONResponse(status_code=exc.status_code, content=exc.body)


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    return JSONResponse(
        content={
            "object": "list",
            "data": [{"id": config.DEEPSEEK_MODEL, "object": "model", "owned_by": config.DEEPSEEK_MODEL}],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid JSON body", "invalid_request_error", "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object", "invalid_request_error", "invalid_json")
    return await route_chat_completions(body)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "invalid JSON body"},
            },
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "request body must be a JSON object"},
            },
        )
    try:
        return await route_anthropic_messages(body)
    except ClientRequestError as exc:
        logger.exception("anthropic 400 (stack): %s", exc.message)
        try:
            block_types = [
                [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in m["content"]]
                if isinstance(m.get("content"), list)
                else None
                for m in (body.get("messages") or [])
            ]
            summary = [
                {
                    "role": m.get("role"),
                    "ctype": type(m.get("content")).__name__,
                    "blocks": block_types[i],
                }
                for i, m in enumerate(body.get("messages") or [])
            ]
            logger.warning("anthropic 400 %s; body summary: %s", exc.message, summary)
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": exc.message},
            },
        )
    except VisionUnavailable:
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": "vision backend unavailable"}},
        )
    except LLMBackendError as exc:
        status = exc.status_code
        try:
            msg = exc.body.get("error", {}).get("message", "upstream error")
        except AttributeError:
            msg = "upstream error"
        return JSONResponse(
            status_code=status,
            content={"type": "error", "error": {"type": "api_error", "message": msg}},
        )


def _check_config() -> None:
    missing = [k for k, v in {
        "FAKE_VLM_API_KEY": config.FAKE_VLM_API_KEY,
        "DASHSCOPE_API_KEY": config.DASHSCOPE_API_KEY,
        "DEEPSEEK_API_KEY": config.DEEPSEEK_API_KEY,
    }.items() if not v]
    if missing:
        logger.warning("missing env vars in .env: %s", ", ".join(missing))


_check_config()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
