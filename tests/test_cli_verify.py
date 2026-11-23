"""CLI wiring: --verify must reach Config.verify_command."""

import agent_harness.__main__ as cli


class FakeLLM:
    def __init__(self, cfg): self.cfg = cfg
    def __enter__(self): return self
    def __exit__(self, *a): return False


CAPTURE = {}


class FakeAgent:
    def __init__(self, cfg, llm, schemas, impls, tracer):
        CAPTURE["verify"] = cfg.verify_command
    def run(self, task):
        return {"answer": "ok", "steps": 1, "tool_calls": 0,
                "finished": True, "error": "", "messages": []}


def test_verify_flag_reaches_config(monkeypatch):
    monkeypatch.setattr(cli, "LLM", FakeLLM)
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    assert cli.main(["task", "do it", "--verify", "pytest -q"]) == 0
    assert CAPTURE["verify"] == "pytest -q"


def test_verify_absent_keeps_default(monkeypatch):
    monkeypatch.setattr(cli, "LLM", FakeLLM)
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    assert cli.main(["task", "do it"]) == 0
    assert CAPTURE["verify"] == ""
