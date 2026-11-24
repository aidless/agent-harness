"""Batch task runner: score a task list and summarize quality + cost.

A tasks file is JSONL, one task per line:
    {"id": "t1", "task": "...", "workdir": "/tmp/t1", "verify_command": "pytest -q"}
Only ``id`` and ``task`` are required; ``workdir`` / ``verify_command`` /
``max_steps`` override the base config for that task.

Outputs under ``--out``:
    <id>.json      per-task record (answer, score, steps, tokens, seconds, error)
    summary.json   pass rate, mean steps/tokens/seconds, per-task table

CLI (usually reached via ``agent-harness bench``):
    python -m agent_harness bench --tasks tasks.jsonl --out results/ \
        --model qwen3:8b --base-url http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from .agent import Agent
from .config import Config
from .llm import LLM
from .tools import make_tools
from .trace import Tracer


def load_tasks(path: str | Path) -> list[dict]:
    """Read a JSONL task file. Blank lines and ``#`` comments are skipped."""
    tasks: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if "id" not in obj or "task" not in obj:
            raise ValueError(f"task line missing id/task: {line[:80]}")
        tasks.append(obj)
    return tasks


def run_task(task: dict, base_cfg: Config, llm_factory=None) -> dict:
    """Run one task and return a record with answer, score, and cost counters."""
    overrides: dict = {}
    if task.get("workdir"):
        overrides["workdir"] = task["workdir"]
    if task.get("verify_command"):
        overrides["verify_command"] = task["verify_command"]
    if task.get("max_steps") is not None:
        overrides["max_steps"] = int(task["max_steps"])
    cfg = replace(base_cfg, **overrides)

    tokens = {"input_tokens": 0, "output_tokens": 0}

    def sink(u: dict) -> None:
        for k in tokens:
            tokens[k] += int(u.get(k, 0) or 0)

    if llm_factory is None:
        llm_factory = lambda c, cb: LLM(c, usage_callback=cb)

    schemas, impls = make_tools(cfg.workdir)
    tracer = Tracer(cfg.trace_path)
    t0 = time.perf_counter()
    with llm_factory(cfg, sink) as llm:
        agent = Agent(cfg, llm, schemas, impls, tracer)
        result = agent.run(task["task"])
    seconds = time.perf_counter() - t0

    return {
        "id": task["id"],
        "answer": result["answer"],
        "score": 1.0 if result["finished"] and not result.get("error") else 0.0,
        "finished": result["finished"],
        "steps": result["steps"],
        "tool_calls": result["tool_calls"],
        "verify_rounds": result["verify_rounds"],
        "error": result["error"],
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "seconds": round(seconds, 3),
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate per-task records into pass rate + mean cost."""
    n = len(records)
    if not n:
        return {"tasks": 0}
    passed = sum(1 for r in records if r["score"] >= 1.0)
    return {
        "tasks": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "mean_steps": round(sum(r["steps"] for r in records) / n, 2),
        "mean_input_tokens": round(sum(r["input_tokens"] for r in records) / n, 1),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in records) / n, 1),
        "mean_seconds": round(sum(r["seconds"] for r in records) / n, 2),
        "per_task": [
            {
                "id": r["id"], "score": r["score"], "steps": r["steps"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "seconds": r["seconds"], "error": r["error"] or "",
            }
            for r in records
        ],
    }


def run_bench(args) -> int:
    """CLI entry: load tasks, run each, write per-task + summary JSON."""
    cfg = Config.load(args.config)
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.workdir:
        cfg.workdir = args.workdir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    records: list[dict] = []
    for task in tasks:
        rec = run_task(task, cfg)
        (out / f"{task['id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(rec)
        print(
            f"[bench] {task['id']}: score={rec['score']} steps={rec['steps']} "
            f"tokens={rec['input_tokens']}+{rec['output_tokens']} {rec['seconds']}s"
        )

    summary = summarize(records)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[bench] " + json.dumps(
        {k: v for k, v in summary.items() if k != "per_task"}, ensure_ascii=False
    ))
    return 0
