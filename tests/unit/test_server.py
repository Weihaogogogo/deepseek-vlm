"""server.py 请求总预算中间件单元测试（无网络、不启动服务）。"""
import asyncio
import json

from src import server


class _FakeURL:
    def __init__(self, path: str):
        self.path = path


class _FakeRequest:
    def __init__(self, path: str):
        self.url = _FakeURL(path)


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
