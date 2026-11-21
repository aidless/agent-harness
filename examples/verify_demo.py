#!/usr/bin/env python3
"""Demonstrate the verification gate end to end with a scripted LLM.

The model first declares "done" before fixing anything. The harness runs the
project's real pytest as a verifier, sees it fail, feeds the output back, and
gives the model another turn. On turn 2 the model writes the fix; on turn 3 it
finishes again and the verifier now passes. Nothing here trusts the model's
own claim of success — only pytest's exit code decides.

Usage:
    python examples/verify_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from agent_harness.agent import Agent  # noqa: E402
from agent_harness.config import Config  # noqa: E402
from agent_harness.llm import LLM  # noqa: E402
from agent_harness.tools import make_tools  # noqa: E402
from agent_harness.trace import Tracer  # noqa: E402
from scripted_demo import FIXED_CALC, FIXTURE, FakeEndpoint, _call  # noqa: E402

VERIFY_CMD = f"{sys.executable} -m pytest -q tests"


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="ah-verify-demo-"))
    try:
        shutil.copytree(FIXTURE, workdir, dirs_exist_ok=True)

        replies = [
            # turn 1: premature "done" -- verifier will reject it
            {"role": "assistant", "content": "Looks done to me."},
            # turn 2: actually fix the file (write_file overwrites the whole file)
            {"role": "assistant", "content": None,
             "tool_calls": [_call(1, "write_file",
                                  {"path": "src/calc.py", "content": FIXED_CALC})]},
            # turn 3: finish again -- verifier now passes
            {"role": "assistant", "content": "Fixed add() and verified with pytest."},
        ]

        with FakeEndpoint(replies) as ep:
            cfg = Config(
                base_url=ep.url, model="scripted", api_key="x",
                workdir=str(workdir), max_steps=8, temperature=0.0,
                verify_command=VERIFY_CMD, max_verify_rounds=2,
            )
            tools, impls = make_tools(workdir)
            tracer = Tracer()
            with LLM(cfg) as llm:
                result = Agent(cfg, llm, tools, impls, tracer).run(
                    "Fix the bug in src/calc.py and make the tests pass."
                )

        events = [json.loads(line)["event"] for line in tracer.lines()] if hasattr(tracer, "lines") else []
        print(f"agent: steps={result['steps']} tool_calls={result['tool_calls']} "
              f"verify_rounds={result['verify_rounds']} "
              f"finished={result['finished']} error={result['error']!r}")
        print(f"answer: {result['answer']}")

        # Independent check: the verifier is authoritative, not the agent.
        proc = __import__("subprocess").run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=workdir, capture_output=True, text=True,
        )
        ok = (result["finished"] and not result["error"]
              and result["verify_rounds"] == 2 and proc.returncode == 0)
        print(f"independent pytest: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
