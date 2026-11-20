"""Demo: run a real multi-step task through the harness.

Copies the fixture project into a scratch dir, then asks the agent to make the
test suite pass. Prints the agent's final answer and verifies the result
independently (outside the harness), so the demo shows a real fix, not a claim.

    python examples/run_demo.py \
        --model qwen2.5:7b --base-url http://127.0.0.1:11434/v1 \
        --workdir /tmp/demo-bugfix

Exit code is 0 when the independent pytest run passes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "bugfix-project"

TASK = (
    "The project in the current directory has a failing test suite. "
    "Run `pytest`, find the bug in src/calc.py, fix it with the write_file tool, "
    "then run `pytest` again to confirm everything passes. "
    "Finish with a one-line summary of what was wrong."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--workdir", default="/tmp/demo-bugfix")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch dir first")
    args = ap.parse_args()

    work = Path(args.workdir)
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE, work / "project")
    project = work / "project"

    print(f"[demo] workdir : {project}")
    print(f"[demo] model   : {args.model} @ {args.base_url}")
    print("[demo] task    :", TASK, "\n")

    harness_cmd = [
        sys.executable, "-m", "agent_harness", "task",
        "--workdir", str(project),
        "--model", args.model,
        "--base-url", args.base_url,
        "--max-steps", str(args.max_steps),
        "--json",
        TASK,
    ]
    proc = subprocess.run(
        harness_cmd, cwd=str(HERE.parent), capture_output=True, text=True, timeout=900,
    )
    if proc.stdout:
        print("[demo] harness output:\n" + proc.stdout.strip() + "\n")
    if proc.stderr:
        print("[demo] harness stderr:\n" + proc.stderr.strip() + "\n")

    # Independent verification: run pytest ourselves, not through the agent.
    print("[demo] independent verification: pytest")
    check = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=str(project),
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip())
    ok = check.returncode == 0
    print(f"\n[demo] result: {'PASS' if ok else 'FAIL'} (pytest exit {check.returncode})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
