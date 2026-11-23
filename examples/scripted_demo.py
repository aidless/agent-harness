#!/usr/bin/env python3
"""Prove the multi-step loop with a *scripted* LLM, not a local model.

Local ollama models fail the bugfix demo (they emit pseudo-code, drop
functions, or write syntactically broken files). That is a model-capability
limit, not a harness bug. This script removes the model from the equation:
a tiny in-process HTTP server plays an OpenAI-compatible endpoint and returns
a fixed sequence of assistant messages (tool_calls). The Agent runs its real
loop with the real tools against a real copy of the fixture. We then run
pytest *independently* and judge only that result.

Usage:
    python examples/scripted_demo.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness.agent import Agent  # noqa: E402
from agent_harness.config import Config  # noqa: E402
from agent_harness.llm import LLM  # noqa: E402
from agent_harness.tools import make_tools  # noqa: E402
from agent_harness.trace import Tracer  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "bugfix-project"

# The bug is one line. With edit_file the scripted model fixes exactly that
# line - the docstrings and all other functions are never rewritten.
BUG_LINE = "    return a - b          # BUG: should be a + b"
FIX_LINE = "    return a + b"


def _call(call_id: int, name: str, args: dict) -> dict:
    return {
        "id": f"call_{call_id}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def scripted_replies(pytest_cmd: str) -> list[dict]:
    """The assistant message the fake model returns on each turn."""
    return [
        # turn 1: read the buggy file
        {"role": "assistant", "content": None,
         "tool_calls": [_call(1, "read_file", {"path": "src/calc.py"})]},
        # turn 2: edit only the buggy line, leaving docstrings byte-for-byte intact
        {"role": "assistant", "content": None,
         "tool_calls": [_call(2, "edit_file", {"path": "src/calc.py", "old": BUG_LINE, "new": FIX_LINE})]},
        # turn 3: run the project's tests
        {"role": "assistant", "content": None,
         "tool_calls": [_call(3, "bash", {"command": pytest_cmd})]},
        # turn 4: finish
        {"role": "assistant", "content": "Fixed add() and verified all three tests pass.",
         "tool_calls": []},
    ]


class FakeEndpoint:
    """In-process OpenAI-compatible server driven by a fixed reply script."""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.n = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                self.rfile.read(length)  # drain the request body
                with outer.lock:
                    i = min(outer.n, len(outer.replies) - 1)
                    msg = dict(outer.replies[i])
                    outer.n += 1
                body = json.dumps({
                    "id": f"chatcmpl-{i}",
                    "object": "chat.completion",
                    "model": "scripted",
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100 + i, "completion_tokens": 20,
                              "total_tokens": 120 + i},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="ah-scripted-"))
    try:
        shutil.copytree(FIXTURE, workdir, dirs_exist_ok=True)
        pytest_cmd = f"{sys.executable} -m pytest -q tests"

        with FakeEndpoint(scripted_replies(pytest_cmd)) as ep:
            cfg = Config(
                base_url=ep.url, model="scripted", api_key="x",
                workdir=str(workdir), max_steps=10, temperature=0.0,
            )
            tools, impls = make_tools(workdir)
            with LLM(cfg) as llm:
                agent = Agent(cfg, llm, tools, impls, Tracer())
                result = agent.run("Find the bug in src/calc.py, fix it, and run the tests.")

        print(f"agent: steps={result['steps']} tool_calls={result['tool_calls']} "
              f"finished={result['finished']} error={result['error']!r}")
        print(f"answer: {result['answer']}")

        # Independent verification — never trust the agent's own claim.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=workdir, capture_output=True, text=True,
        )
        print("--- independent pytest ---")
        print(proc.stdout.strip() or proc.stderr.strip())
        ok = result["finished"] and not result["error"] and proc.returncode == 0
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
