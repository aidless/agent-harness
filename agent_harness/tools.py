"""Built-in tools: bash, read_file, write_file, scoped to a work directory."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


class ToolError(RuntimeError):
    pass


def _resolve(workdir: Path, path: str) -> Path:
    """Resolve a user path against workdir, refusing to escape it."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = workdir / p
    p = p.resolve()
    root = workdir.resolve()
    if p != root and root not in p.parents:
        raise ToolError(f"path escapes workdir: {path}")
    return p


def make_tools(workdir: str | Path, timeout: int = 120) -> tuple[list[dict], dict[str, Callable]]:
    root = Path(workdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def bash(command: str) -> str:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=root, capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        return out or f"[empty output; exit={proc.returncode}]"

    def read_file(path: str) -> str:
        p = _resolve(root, path)
        if not p.exists():
            return f"[file not found: {path}]"
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(path: str, content: str) -> str:
        p = _resolve(root, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p.relative_to(root)}"

    impls: dict[str, Callable] = {
        "bash": bash,
        "read_file": read_file,
        "write_file": write_file,
    }

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the work directory. Returns stdout+stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file inside the work directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write text to a file inside the work directory (creates parent dirs).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]
    return schemas, impls
