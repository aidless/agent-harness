"""End-to-end multi-step test driven by a scripted (deterministic) LLM.

This proves the Agent loop itself is correct: it reads a file, overwrites it
with a complete fix, runs the tests, and finishes -- then an *independent*
pytest run confirms the fix. No local model capability is involved, so this
test is stable regardless of which ollama models happen to be installed.
"""

from __future__ import annotations

import shutil
import subprocess
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
from scripted_demo import FIXTURE, FakeEndpoint, scripted_replies  # noqa: E402


def test_scripted_multistep_bugfix_passes_independent_pytest():
    workdir = Path(tempfile.mkdtemp(prefix="ah-test-scripted-"))
    try:
        shutil.copytree(FIXTURE, workdir, dirs_exist_ok=True)
        pytest_cmd = f"{sys.executable} -m pytest -q tests"

        with FakeEndpoint(scripted_replies(pytest_cmd)) as ep:
            cfg = Config(base_url=ep.url, model="scripted", api_key="x",
                         workdir=str(workdir), max_steps=10)
            tools, impls = make_tools(workdir)
            with LLM(cfg) as llm:
                result = Agent(cfg, llm, tools, impls, Tracer()).run(
                    "Find the bug in src/calc.py, fix it, and run the tests.")

        assert result["finished"] is True
        assert result["error"] == ""
        assert result["steps"] == 4
        assert result["tool_calls"] == 3
        assert "add() and verified" in result["answer"]

        # The file must actually be fixed on disk (add returns a + b).
        source = (workdir / "src" / "calc.py").read_text()
        assert "return a + b" in source
        assert "def mul" in source and "def div" in source  # nothing dropped

        # And the project's own tests must pass independently.
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                              cwd=workdir, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "3 passed" in proc.stdout
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
