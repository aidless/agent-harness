"""WebSocket serve layer: run this harness as a PLE-protocol (v1) server.

The evaluator opens one WebSocket per task and sends a ``task`` frame. From
then on the agent loop runs locally while its tools (bash / read_file /
write_file) are executed **inside the task container** by sending
``exec`` / ``read_file`` / ``write_file`` requests back over the socket. When
the loop finishes we send ``done`` with the final message.

This is deliberately a small client: requests are issued one at a time (the
protocol allows up to 8 in flight), and ``cancel`` marks the session so every
further tool call fails fast and the loop ends with what it has.

Run:
    PORT=8765 python -m agent_harness.serve

The model endpoint comes from the evaluator's environment:
    MODEL_NAME      the model to call
    MODEL_BASE_URL / OPENAI_BASE_URL   OpenAI-compatible base URL (with /v1)
    MODEL_API_KEY  / OPENAI_API_KEY    bearer key
Falling back to ``AGENT_*`` variables and the Config defaults.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from concurrent.futures import Future

try:  # websockets is optional: only the serve mode needs it, not the CLI
    import websockets

    ConnectionClosed = websockets.ConnectionClosed
except ImportError:  # pragma: no cover - exercised only without the extra
    websockets = None

    class ConnectionClosed(Exception):  # never actually raised; keeps except valid
        pass

from .agent import Agent
from .config import Config
from .llm import LLM, LLMError
from .tools import make_tools, ToolError
from .trace import Tracer

MAX_FRAME = 16 * 2**20  # protocol: every frame is UTF-8 JSON, at most 16 MiB
REQUEST_TIMEOUT = 180.0  # per tool request, seconds


# ----------------------------------------------------------------- helpers
def _b64decode(value: str | None) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode("utf-8", errors="replace")


def _b64encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def serve_config() -> Config:
    """Build the Config from the evaluator environment, then AGENT_* vars."""
    cfg = Config.load(os.environ.get("AGENT_CONFIG") or None)
    if os.environ.get("MODEL_NAME"):
        cfg.model = os.environ["MODEL_NAME"]
    base = os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base:
        cfg.base_url = base
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        cfg.api_key = key
    return cfg


class RemoteSession:
    """Synchronous request/response bridge for one WebSocket task.

    The agent runs in a worker thread with synchronous tools; those tools block
    on ``request()`` while the main event loop keeps reading frames and
    resolving the matching ``result`` (or ``cancel``).
    """

    def __init__(self, ws, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self._pending: dict[str, Future] = {}
        self._n = 0
        self.cancelled = threading.Event()
        self.llm = None  # set by the worker so cancel can abort an in-flight call

    # -- worker thread side ------------------------------------------------
    def request(self, payload: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
        if self.cancelled.is_set():
            return {"ok": False, "error": "cancelled: the time is up"}
        self._n += 1
        rid = str(self._n)
        fut: Future = Future()
        self._pending[rid] = fut
        try:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps({**payload, "id": rid}, ensure_ascii=False)),
                self.loop,
            )
            return fut.result(timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return {"ok": False, "error": f"request {rid} timed out after {timeout}s"}
        finally:
            self._pending.pop(rid, None)

    # -- event loop side --------------------------------------------------
    async def pump(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "result":
                    fut = self._pending.pop(str(msg.get("id")), None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                elif kind == "cancel":
                    self.cancelled.set()
                    if self.llm is not None:
                        self.llm.abort()  # unblock a hanging model call
                    for fut in list(self._pending.values()):
                        if not fut.done():
                            fut.set_result(
                                {"ok": False, "error": "cancelled: the time is up"}
                            )
                    self._pending.clear()
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            # unblock any tool still waiting
            self.cancelled.set()
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_result({"ok": False, "error": "connection closed"})
            self._pending.clear()


def remote_tools(rs: RemoteSession):
    """Tool impls that execute in the task container; schema matches make_tools."""
    schemas, _ = make_tools(".")  # reuse the local schemas, swap the impls

    def bash(command: str, timeout: int | None = None) -> str:
        r = rs.request(
            {"type": "exec", "command": command, "timeout_sec": int(timeout or 120)},
            timeout=float(timeout or 120) + 30,
        )
        if r.get("ok") is False:
            return f"[exec error: {r.get('error', 'unknown')}]"
        out = r.get("output") or ""
        if r.get("timed_out"):
            out += "\n[timed out]"
        if r.get("truncated"):
            out += "\n[output truncated]"
        return out or f"[empty output; exit={r.get('exit_code')}]"

    def read_file(path: str) -> str:
        r = rs.request({"type": "read_file", "path": path})
        if not r.get("ok"):
            return f"[read error: {r.get('error', 'missing file?')}]"
        content = _b64decode(r.get("content_b64"))
        if r.get("truncated"):
            content += "\n...[truncated by evaluator]"
        return content

    def write_file(path: str, content: str) -> str:
        payload = _b64encode(content)
        if len(payload) > 10 * 1024 * 1024:
            return "[write error: payload over the 10 MB limit]"
        r = rs.request(
            {"type": "write_file", "path": path, "content_b64": payload, "append": False}
        )
        if not r.get("ok"):
            return f"[write error: {r.get('error', 'unknown')}]"
        return f"wrote {len(content)} chars to {path}"

    impls = {"bash": bash, "read_file": read_file, "write_file": write_file}
    return schemas, impls


# ----------------------------------------------------------------- session
async def session(ws) -> None:
    first = await ws.recv()
    try:
        task = json.loads(first)
    except ValueError:
        await ws.send(json.dumps({"type": "error", "message": "first frame is not JSON"}))
        return
    if task.get("type") != "task":
        await ws.send(json.dumps({"type": "error", "message": "expected a task frame"}))
        return

    instruction = task.get("instruction") or ""
    workdir = task.get("workdir") or "/app"
    limits = task.get("limits") or {}
    agent_timeout = limits.get("agent_timeout_sec")

    loop = asyncio.get_running_loop()
    rs = RemoteSession(ws, loop)
    pump_task = asyncio.create_task(rs.pump())

    cfg = serve_config()
    # Keep each model call well inside the task's remaining time.
    if isinstance(agent_timeout, (int, float)) and agent_timeout > 0:
        cfg.timeout = max(30.0, min(cfg.timeout, float(agent_timeout)))

    async def log(text: str) -> None:
        try:
            await ws.send(json.dumps({"type": "log", "text": text}))
        except ConnectionClosed:
            pass

    def on_usage(u: dict) -> None:
        # usage frames are one-way: safe to send from the worker thread.
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps({"type": "usage", **u}, ensure_ascii=False)), loop
            )
        except Exception:
            pass

    def worker() -> dict:
        schemas, impls = remote_tools(rs)
        tracer = Tracer(cfg.trace_path)
        tracer.log("serve_task", workdir=workdir, model=cfg.model, session=task.get("session"))
        try:
            with LLM(cfg, usage_callback=on_usage) as llm:
                rs.llm = llm
                agent = Agent(cfg, llm, schemas, impls, tracer)
                return agent.run(instruction)
        except LLMError as exc:
            return {"answer": "", "error": str(exc), "finished": False}
        except Exception as exc:
            return {"answer": "", "error": f"harness crashed: {exc}", "finished": False}

    try:
        result = await asyncio.to_thread(worker)
    except Exception as exc:  # never leave the evaluator waiting
        result = {"answer": "", "error": f"harness crashed: {exc}", "finished": False}

    error = (result.get("error") or "").strip()
    final = (result.get("answer") or "").strip()
    if not final:
        final = "(the harness ended without a final answer)"

    pump_task.cancel()
    try:
        if error:
            # Spec: error ends the task early; files already written are still graded.
            # Include the best answer so far in the message for the grader.
            await ws.send(json.dumps(
                {"type": "error", "message": f"{final}\n[{error}]"}, ensure_ascii=False
            ))
        else:
            await ws.send(json.dumps(
                {"type": "done", "final_message": final}, ensure_ascii=False
            ))
    except ConnectionClosed:
        pass


async def serve_forever(port: int | None = None) -> None:
    if websockets is None:
        raise RuntimeError(
            "serve mode needs the 'websockets' package: pip install agent-harness[serve]"
        )
    port = port or int(os.environ.get("PORT", "8765"))
    async with websockets.serve(
        session,
        "0.0.0.0",
        port,
        ping_interval=None,  # protocol: no keepalive pings
        max_size=MAX_FRAME,
        close_timeout=10,
    ):
        print(f"agent-harness serve listening on 0.0.0.0:{port}", flush=True)
        await asyncio.Future()


def main() -> None:
    asyncio.run(serve_forever())


if __name__ == "__main__":
    main()
