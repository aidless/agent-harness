"""Serve layer: protocol frames, request/response bridge, cancel, env config."""

import asyncio
import base64
import json

from agent_harness import serve
from agent_harness.serve import RemoteSession, _b64decode, _b64encode, remote_tools, serve_config


def test_b64_roundtrip_unicode():
    text = "你好 world\nline2"
    assert _b64decode(_b64encode(text)) == text


def test_b64_empty():
    assert _b64decode("") == ""
    assert _b64decode(None) == ""


class QueueWS:
    """In-memory WebSocket: send() auto-replies to requests; pump reads inbound."""

    def __init__(self):
        self.inbound = asyncio.Queue()
        self.outbound = []
        self.sent_calls = []

    async def send(self, text):
        msg = json.loads(text)
        self.outbound.append(msg)
        if "id" in msg:  # a request -> auto-reply
            self.sent_calls.append(msg)
            reply = {"type": "result", "id": msg["id"]}
            t = msg["type"]
            if t == "exec":
                reply.update(ok=True, exit_code=0, output="OK", truncated=False, timed_out=False)
            elif t == "read_file":
                reply.update(ok=True, content_b64=_b64encode("FILE:" + msg["path"]), size=9)
            elif t == "write_file":
                reply.update(ok=True, size=1)
            self.inbound.put_nowait(json.dumps(reply))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbound.get()


def test_request_response_bridge():
    async def scenario():
        loop = asyncio.get_running_loop()
        ws = QueueWS()
        rs = RemoteSession(ws, loop)
        pump = asyncio.create_task(rs.pump())
        r = await asyncio.to_thread(
            rs.request, {"type": "exec", "command": "echo hi", "timeout_sec": 5}
        )
        pump.cancel()
        return r, ws

    r, ws = asyncio.run(scenario())
    assert r["output"] == "OK"
    assert ws.sent_calls[0]["command"] == "echo hi"
    assert ws.sent_calls[0]["id"] == "1"


def test_cancel_marks_and_unblocks_pending():
    async def scenario():
        loop = asyncio.get_running_loop()
        ws = QueueWS()
        rs = RemoteSession(ws, loop)
        pump = asyncio.create_task(rs.pump())
        # A request with no auto-reply (simulate an in-flight tool).
        ws.inbound.put_nowait(json.dumps({"type": "result", "id": "1", "ok": True}))
        # cancel arrives before the reply
        ws2 = QueueWS()
        rs2 = RemoteSession(ws2, loop)
        pump2 = asyncio.create_task(rs2.pump())
        fut = asyncio.Future()
        rs2._pending["1"] = fut
        ws2.inbound.put_nowait(json.dumps({"type": "cancel", "reason": "timeout"}))
        await asyncio.sleep(0.05)
        out = fut.result()
        cancelled = rs2.cancelled.is_set()
        pump.cancel(); pump2.cancel()
        return out, cancelled

    out, cancelled = asyncio.run(scenario())
    assert cancelled
    assert out["ok"] is False and "cancelled" in out["error"]


def test_remote_tools_bash_read_write():
    async def scenario():
        loop = asyncio.get_running_loop()
        ws = QueueWS()
        rs = RemoteSession(ws, loop)
        pump = asyncio.create_task(rs.pump())
        _, impls = remote_tools(rs)
        # Tools block on rs.request(), so they must run off the event loop
        # thread — same as the real worker.
        def run_tools():
            return {
                "bash": impls["bash"]("pwd", 5),
                "read_file": impls["read_file"]("a.txt"),
                "write_file": impls["write_file"]("a.txt", "data"),
            }
        out = await asyncio.to_thread(run_tools)
        pump.cancel()
        return out, ws.sent_calls

    out, calls = asyncio.run(scenario())
    assert out["bash"] == "OK"
    assert out["read_file"] == "FILE:a.txt"
    assert out["write_file"] == "wrote 4 chars to a.txt"
    kinds = [c["type"] for c in calls]
    assert kinds == ["exec", "read_file", "write_file"]
    # write_file payload must be base64
    assert _b64decode(calls[2]["content_b64"]) == "data"


def test_remote_tools_error_mapping():
    class FailRS:
        cancelled = None

        def request(self, payload, timeout=None):
            return {"ok": False, "error": "bad path"}

    _, impls = remote_tools(FailRS())
    assert "read error" in impls["read_file"]("x")
    assert "write error" in impls["write_file"]("x", "y")
    assert "exec error" in impls["bash"]("ls")


def test_serve_config_model_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "env-model")
    monkeypatch.setenv("MODEL_BASE_URL", "http://env/v1")
    monkeypatch.setenv("MODEL_API_KEY", "envkey")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    cfg = serve_config()
    assert cfg.model == "env-model"
    assert cfg.base_url == "http://env/v1"
    assert cfg.api_key == "envkey"


def test_serve_config_openai_fallback(monkeypatch):
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://oa/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "oakey")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    cfg = serve_config()
    assert cfg.base_url == "http://oa/v1"
    assert cfg.api_key == "oakey"
