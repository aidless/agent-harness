"""CLI entry point for agent-harness.

Usage:
    python -m agent_harness task "Do something" [--config config.yaml] [--workdir DIR]
    python -m agent_harness task --file task.txt
    echo "task" | python -m agent_harness task -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent
from .config import Config
from .llm import LLM
from .tools import make_tools
from .trace import Tracer


def read_task(args) -> str:
    if args.file:
        if args.file == "-":
            return sys.stdin.read().strip()
        return Path(args.file).read_text(encoding="utf-8").strip()
    return " ".join(args.task).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-harness", description="A minimal tool-calling agent harness."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("task", help="Run one task through the agent loop")
    run.add_argument("task", nargs="*", help="Task text (or use --file)")
    run.add_argument("--file", "-f", help="Read task from a file, or - for stdin")
    run.add_argument("--config", "-c", help="YAML config file")
    run.add_argument("--workdir", "-w", help="Override the tool work directory")
    run.add_argument("--model", help="Override the model name")
    run.add_argument("--base-url", help="Override the API base URL")
    run.add_argument("--max-steps", type=int, help="Override max agent steps")
    run.add_argument("--trace", help="Write a JSONL trace to this path")
    run.add_argument("--json", action="store_true", help="Print full result as JSON")

    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.workdir:
        cfg.workdir = args.workdir
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.trace:
        cfg.trace_path = args.trace

    task = read_task(args)
    if not task:
        print("error: empty task", file=sys.stderr)
        return 2

    schemas, impls = make_tools(cfg.workdir)
    tracer = Tracer(cfg.trace_path)
    tracer.log("start", task=task[:500], model=cfg.model, workdir=cfg.workdir)

    with LLM(cfg) as llm:
        agent = Agent(cfg, llm, schemas, impls, tracer)
        result = agent.run(task)

    if args.json:
        print(json.dumps({
            "answer": result["answer"], "steps": result["steps"],
            "tool_calls": result["tool_calls"], "finished": result["finished"],
            "error": result["error"],
        }, ensure_ascii=False, indent=2))
    else:
        if result["error"]:
            print(f"[harness] {result['error']}", file=sys.stderr)
        print(result["answer"] or "(no answer)")

    return 0 if result["finished"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
