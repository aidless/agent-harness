"""Mock evaluator: drive agent_harness.serve over the PLE v1 WebSocket protocol.

This plays the evaluator's role so the serve layer can be tested locally, and
doubles as the reference for "what the server sees":

  1. open one WebSocket per task
  2. send a `task` frame (protocol 1)
  3. serve sends `exec` / `read_file` / `write_file` requests; this mock runs
     them in a local directory standing in for the task container
  4. when serve sends `done`, print the final message and exit

Run (with a serve listening on ws://127.0.0.1:8765):

    python tests/serve_smoke.py "Create hello.txt with 'hi' then read it back"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shlex
import subprocess
from pathlib import Path

import websockets

MAX_OUT = 100_000


def clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUT:
        return text, False
    head = text[: MAX_OUT // 2]
    tail = text[-(MAX_OUT // 2):]
    return f"{head}\n[... {len(text) - MAX_OUT} bytes cut ...]\n{tail}", True


def run_exec(workdir: Path, command: str, timeout_sec: int) -> dict:
    timeout_sec = max(1, min(int(timeout_sec or 120), 600))
    try:
        proc = subprocess.run(
            command, shell=True, cwd=workdir, capture_output=True,
            timeout=timeout_sec,
        )
        out = (proc.stdout or b"") + (proc.stderr or b"")
        text = out.decode("utf-8", errors="replace")
        text, truncated = clip(text)
        return {
            "type": "result", "id": None, "exit_code": proc.returncode,
            "output": text, "truncated": truncated, "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "type": "result", "id": None, "exit_code": 124,
            "output": f"[timed out after {timeout_sec}s]",
            "truncated": False, "timed_out": True,
        }


def resolve(workdir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (workdir / p)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="?", default="List files in the workdir and summarise.")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--workdir", default="/tmp/serve-smoke")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(args.url, max_size=16 * 2**20) as ws:
        await ws.send(json.dumps({
            "type": "task", "protocol": 1, "session": "smoke-test",
            "instruction": args.instruction, "workdir": str(workdir),
            "limits": {"agent_timeout_sec": 600, "exec_timeout_sec": 600,
                       "max_output_bytes": MAX_OUT, "max_write_bytes": 10_000_000},
            "mcp_servers": [],
        }, ensure_ascii=False))

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "exec":
                reply = run_exec(workdir, msg.get("command", ""), msg.get("timeout_sec", 120))
                reply["id"] = msg.get("id")
            elif kind == "read_file":
                p = resolve(workdir, msg.get("path", ""))
                try:
                    data = p.read_bytes()
                    reply = {"type": "result", "id": msg.get("id"), "ok": True,
                             "content_b64": base64.b64encode(data).decode("ascii"),
                             "size": len(data), "truncated": False}
                except OSError as exc:
                    reply = {"type": "result", "id": msg.get("id"), "ok": False,
                             "error": f"cannot read {msg.get('path')}: {exc}"}
            elif kind == "write_file":
                p = resolve(workdir, msg.get("path", ""))
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    data = base64.b64decode(msg.get("content_b64") or "")
                    with p.open("ab" if msg.get("append") else "wb") as fh:
                        fh.write(data)
                    reply = {"type": "result", "id": msg.get("id"), "ok": True, "size": len(data)}
                except OSError as exc:
                    reply = {"type": "result", "id": msg.get("id"), "ok": False,
                             "error": f"cannot write {msg.get('path')}: {exc}"}
            elif kind == "log":
                print(f"[log] {msg.get('text')}")
                continue
            elif kind == "usage":
                print(f"[usage] {json.dumps(msg, ensure_ascii=False)[:200]}")
                continue
            elif kind == "done":
                print("\n=== DONE ===")
                print(msg.get("final_message", ""))
                return 0
            elif kind == "error":
                print(f"\n=== ERROR ===\n{msg.get('message')}")
                return 1
            else:
                continue
            await ws.send(json.dumps(reply, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
