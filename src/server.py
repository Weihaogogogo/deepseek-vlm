"""FastAPI entry: auth + POST /v1/chat/completions + GET /v1/models."""
import hmac
import logging

import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .llm_client import LLMBackendError
from .router import ClientRequestError, VisionUnavailable, route_chat_completions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="fake-vlm",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


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
    if not expected or not auth.startswith("Bearer "):
        raise AuthError
    token = auth[len("Bearer "):].strip()
    if not hmac.compare_digest(token, expected):
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
            "data": [{"id": "fake-vlm", "object": "model", "owned_by": "fake-vlm"}],
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
