"""FastAPI entry: auth + POST /v1/chat/completions + GET /v1/models."""
import asyncio
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
    title="deepseek-vlm",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# 必须注册在最外层：WorkBuddy 中转层在 keep-alive 连接上重放旧报文导致
# 400；所有业务响应带 Connection: close 后 h11 响应完即断连，补发只能走
# 新连接正常组装。内层中间件（body 大小 413 / 请求预算 502）的早退响应
# 也会经过本中间件补上该头。
@app.middleware("http")
async def _connection_close(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path in ("/v1/chat/completions", "/v1/messages"):
        resp.headers["connection"] = "close"
    return resp


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    """Reject request bodies over 50MB with a 413 before the route parses them."""
    if request.url.path not in ("/v1/chat/completions", "/v1/messages"):
        return await call_next(request)
    import time as _time

    t_header = _time.monotonic()
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
    t_body_done = _time.monotonic()
    # Cache the read body so the route's request.json() does not re-read the stream.
    request._body = b"".join(chunks)
    resp = await call_next(request)
    t_done = _time.monotonic()
    logger.info(
        "[timing] %s size=%.1fKB header_to_body=%.2fs body_to_done=%.2fs total=%.2fs",
        request.url.path,
        size / 1024,
        t_body_done - t_header,
        t_done - t_body_done,
        t_done - t_header,
    )
    return resp


# 请求总预算：上游（DashScope/deepseek）故障时网关必须快速失败，而不是让
# 客户端无限等待。必须大于 VLM_TIMEOUT(120s) + deepseek 调用时间，否则正常
# 请求（VLM 慢但成功）会被误杀。150s 给完整链路留余量，故障时仍兜底。
REQUEST_TIMEOUT_SECONDS = 150


@app.middleware("http")
async def _request_timeout_guard(request: Request, call_next):
    """Fail fast with 502 when the whole request chain exceeds the budget."""
    if request.url.path not in ("/v1/chat/completions", "/v1/messages"):
        return await call_next(request)
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            "request exceeded %ds budget, returning 502: %s",
            REQUEST_TIMEOUT_SECONDS,
            request.url.path,
        )
        if request.url.path == "/v1/messages":
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "gateway timeout (60s)",
                    },
                },
            )
        return _error(
            502,
            "gateway timeout (60s)",
            "api_error",
            "gateway_timeout",
        )


def _error(status: int, message: str, type_: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
    )


def _parse_wrapped_http(raw: bytes) -> dict:
    """Split a raw HTTP-message-wrapped body into request line / headers / JSON head.

    WorkBuddy sends the full outgoing HTTP request as the POST body; splitting
    the message lets the failure log show which inner request was wrapped and
    its internal Content-Length.
    """
    text = raw.decode("utf-8", errors="replace")
    first_line = text.split("\r\n", 1)[0] if text else ""
    header_block = ""
    json_head = ""
    if "\r\n\r\n" in text:
        header_block, json_head = text.split("\r\n\r\n", 1)
    return {
        "first_line": first_line,
        "header_block": header_block,
        "json_head": json_head,
    }


async def _load_json_body(request: Request) -> dict:
    """Parse the request body as JSON, honoring gzip Content-Encoding.

    Some desktop harnesses (WorkBuddy) gzip large request bodies; Starlette's
    ``request.json()`` does NOT auto-decompress, so compressed bytes fail
    json.loads -> 400 "invalid JSON body". Decompress first, then parse.
    On failure, logs a detailed diagnostic (encoding, lengths, head/tail of
    the raw bytes) so the offending request can be identified without asking
    the user to reproduce again.
    """
    import gzip
    import json as _json

    raw = await request.body()  # cached by the size-limiting middleware
    enc = (request.headers.get("content-encoding") or "").lower().strip()
    if enc == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "gzip decompress failed: %s len=%d head=%r", exc, len(raw), raw[:120]
            )
            raise
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("decompressed body exceeds 50MB")
    elif enc and enc != "identity":
        logger.warning(
            "unsupported content-encoding %r (len=%d) — treating as plain JSON",
            enc,
            len(raw),
        )
    try:
        return _json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        wrapped = _parse_wrapped_http(raw)
        logger.error(
            "invalid JSON body: enc=%r len=%d content-length=%r first_line=%r "
            "header_block=%r json_head=%r tail=%r err=%s",
            enc,
            len(raw),
            request.headers.get("content-length"),
            wrapped["first_line"][:120],
            wrapped["header_block"][:500],
            wrapped["json_head"][:300],
            raw[-80:],
            exc,
        )
        raise


class AuthError(Exception):
    pass


def _check_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    expected = config.DEEPSEEK_VLM_API_KEY
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
    # 对外模型列表：只暴露 PUBLIC_MODEL_NAME（deepseek-v4-flash-vl）。
    # 旧客户端如需使用，改自己的 model 配置即可（节点对任意 model 名都接受）。
    ids = [config.PUBLIC_MODEL_NAME]
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": mid}
                for mid in dict.fromkeys(ids)
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    try:
        body = await _load_json_body(request)
    except Exception:
        return _error(400, "invalid JSON body", "invalid_request_error", "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object", "invalid_request_error", "invalid_json")
    return await route_chat_completions(body)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    _check_auth(request)
    try:
        body = await _load_json_body(request)
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
        "DEEPSEEK_VLM_API_KEY": config.DEEPSEEK_VLM_API_KEY,
        "DASHSCOPE_API_KEY": config.DASHSCOPE_API_KEY,
        "DEEPSEEK_API_KEY": config.DEEPSEEK_API_KEY,
    }.items() if not v]
    if missing:
        logger.warning("missing env vars in .env: %s", ", ".join(missing))


_check_config()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
