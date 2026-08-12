"""server.py 请求总预算中间件单元测试（无网络、不启动服务）。"""
import asyncio
import json

import pytest

from fastapi.responses import JSONResponse

from src import server


class _FakeURL:
    def __init__(self, path: str):
        self.path = path


class _FakeRequest:
    def __init__(self, path: str):
        self.url = _FakeURL(path)


class _BodyRequest:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


async def _never_next(request):
    await asyncio.sleep(5)
    return "should not happen"


async def _ok_next(request):
    return "ok"


def test_timeout_openai_returns_502_json(monkeypatch):
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_SECONDS", 0.01)
    resp = asyncio.run(server._request_timeout_guard(_FakeRequest("/v1/chat/completions"), _never_next))
    assert resp.status_code == 502
    body = json.loads(resp.body)
    assert body["error"]["code"] == "gateway_timeout"


def test_timeout_anthropic_returns_502_json(monkeypatch):
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_SECONDS", 0.01)
    resp = asyncio.run(server._request_timeout_guard(_FakeRequest("/v1/messages"), _never_next))
    assert resp.status_code == 502
    body = json.loads(resp.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"


def test_non_completion_path_not_wrapped(monkeypatch):
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_SECONDS", 0.01)
    async def slow_next(request):
        await asyncio.sleep(5)
        return "slow-ok"

    # 非网关路径（如 /v1/models）不进入 wait_for，原样放行
    resp = asyncio.run(server._request_timeout_guard(_FakeRequest("/v1/models"), _ok_next))
    assert resp == "ok"


def test_under_budget_passes_through(monkeypatch):
    resp = asyncio.run(server._request_timeout_guard(_FakeRequest("/v1/chat/completions"), _ok_next))
    assert resp == "ok"


def test_parse_wrapped_http_extracts_parts():
    raw = (
        b"POST http://127.0.0.1:8000/v1/chat/completions HTTP/1.1\r\n"
        b"Content-Length: 123\r\n"
        b"Host: localhost\r\n"
        b"x-stainless-helper-method: core\r\n"
        b"\r\n"
        b'{"model": "deepseek-v4-flash-vl", "tools": [{"type": "function"}]}'
    )
    parts = server._parse_wrapped_http(raw)
    assert parts["first_line"] == "POST http://127.0.0.1:8000/v1/chat/completions HTTP/1.1"
    assert "Content-Length: 123" in parts["header_block"]
    assert "x-stainless-helper-method" in parts["header_block"]
    assert parts["json_head"].startswith('{"model":')
    assert "tools" in parts["json_head"]


def test_invalid_json_body_logs_wrapped_http_details(caplog):
    raw = (
        b"POST http://x HTTP/1.1\r\n"
        b"Content-Length: 999\r\n"
        b"\r\n"
        b'{"model": "m"}'
    )
    request = _BodyRequest(raw, {"content-length": "111"})
    with pytest.raises(ValueError):
        asyncio.run(server._load_json_body(request))
    msg = next(r.message for r in caplog.records if "invalid JSON body" in r.message)
    assert "POST http://x HTTP/1.1" in msg
    assert "Content-Length: 999" in msg
    assert '"model": "m"' in msg


async def _json_ok_next(request):
    return JSONResponse({"ok": True})


def test_connection_close_chat_completions():
    resp = asyncio.run(server._connection_close(_FakeRequest("/v1/chat/completions"), _json_ok_next))
    assert resp.headers["connection"] == "close"


def test_connection_close_messages():
    resp = asyncio.run(server._connection_close(_FakeRequest("/v1/messages"), _json_ok_next))
    assert resp.headers["connection"] == "close"


def test_connection_close_not_on_models():
    resp = asyncio.run(server._connection_close(_FakeRequest("/v1/models"), _json_ok_next))
    assert "connection" not in resp.headers


def test_connection_close_e2e(monkeypatch):
    """完整中间件栈：业务路径成功/失败响应都带 connection: close，探活路径不带。"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server.config, "DEEPSEEK_VLM_API_KEY", "test-key")

    async def fake_chat(body):
        return JSONResponse({"id": "x", "object": "chat.completion", "choices": []})

    async def fake_messages(body):
        return JSONResponse({"id": "x", "type": "message", "content": []})

    monkeypatch.setattr(server, "route_chat_completions", fake_chat)
    monkeypatch.setattr(server, "route_anthropic_messages", fake_messages)

    client = TestClient(server.app)
    auth = {"Authorization": "Bearer test-key"}

    resp = client.post("/v1/chat/completions", json={"model": "m"}, headers=auth)
    assert resp.status_code == 200
    assert resp.headers["connection"] == "close"

    resp = client.post("/v1/messages", json={"model": "m", "messages": []}, headers=auth)
    assert resp.status_code == 200
    assert resp.headers["connection"] == "close"

    resp = client.post("/v1/chat/completions", content=b"not json", headers=auth)
    assert resp.status_code == 400
    assert resp.headers["connection"] == "close"

    resp = client.get("/v1/models", headers=auth)
    assert resp.status_code == 200
    assert "connection" not in resp.headers


def test_models_lists_both_public_model_ids(monkeypatch):
    """/v1/models 同时暴露 flash-vl 与 pro-vl 两个对外模型 id。"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server.config, "DEEPSEEK_VLM_API_KEY", "test-key")
    client = TestClient(server.app)
    auth = {"Authorization": "Bearer test-key"}

    resp = client.get("/v1/models", headers=auth)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == [
        server.config.PUBLIC_MODEL_NAME,
        server.config.PUBLIC_PRO_MODEL_NAME,
    ]
    assert "deepseek-v4-flash-vl" in ids
    assert "deepseek-v4-pro-vl" in ids
