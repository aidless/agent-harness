"""Batch bench: task loading, per-task scoring, summary aggregation."""

import json

from agent_harness.bench import load_tasks, run_task, summarize
from agent_harness.config import Config


class FakeLLM:
    """Answers every task with a fixed reply; no tool calls, no network."""

    def __init__(self, cfg, usage_callback=None):
        self.cfg = cfg
        self.usage_callback = usage_callback

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def chat(self, messages, tools=None):
        if self.usage_callback:
            self.usage_callback({"input_tokens": 100, "output_tokens": 20})
        return {"role": "assistant", "content": "done"}

    def abort(self):
        pass


def _factory(cfg, cb):
    return FakeLLM(cfg, cb)


def test_load_tasks_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        '# a comment\n'
        '\n'
        '{"id": "t1", "task": "do a"}\n'
        '{"id": "t2", "task": "do b", "workdir": "/tmp/x"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(f)
    assert [t["id"] for t in tasks] == ["t1", "t2"]
    assert tasks[1]["workdir"] == "/tmp/x"


def test_load_tasks_rejects_line_without_id(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('{"task": "no id"}\n', encoding="utf-8")
    try:
        load_tasks(f)
    except ValueError as exc:
        assert "id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_task_records_score_steps_and_tokens(tmp_path):
    cfg = Config(workdir=str(tmp_path), max_steps=2)
    rec = run_task({"id": "t1", "task": "do it"}, cfg, llm_factory=_factory)
    assert rec["id"] == "t1"
    assert rec["score"] == 1.0          # finished, no error
    assert rec["steps"] == 1
    assert rec["tool_calls"] == 0
    assert rec["input_tokens"] == 100
    assert rec["output_tokens"] == 20
    assert rec["seconds"] >= 0.0


def test_run_task_applies_per_task_overrides(tmp_path):
    cfg = Config(workdir=str(tmp_path), max_steps=9)
    wd = tmp_path / "sub"
    wd.mkdir()
    seen = {}

    class SpyLLM(FakeLLM):
        def __init__(self, cfg, usage_callback=None):
            super().__init__(cfg, usage_callback)
            seen["workdir"] = cfg.workdir
            seen["max_steps"] = cfg.max_steps

    rec = run_task(
        {"id": "t", "task": "x", "workdir": str(wd), "max_steps": 3},
        cfg, llm_factory=lambda c, cb: SpyLLM(c, cb),
    )
    assert seen["workdir"] == str(wd)
    assert seen["max_steps"] == 3
    assert rec["id"] == "t"


def test_summarize_pass_rate_and_means():
    recs = [
        {"id": "a", "score": 1.0, "steps": 2, "input_tokens": 10,
         "output_tokens": 2, "seconds": 1.0, "error": ""},
        {"id": "b", "score": 0.0, "steps": 4, "input_tokens": 30,
         "output_tokens": 6, "seconds": 3.0, "error": "max_steps"},
    ]
    s = summarize(recs)
    assert s["tasks"] == 2
    assert s["passed"] == 1
    assert s["pass_rate"] == 0.5
    assert s["mean_steps"] == 3.0
    assert s["mean_input_tokens"] == 20.0
    assert s["mean_seconds"] == 2.0
    assert [r["id"] for r in s["per_task"]] == ["a", "b"]


def test_summarize_empty_is_safe():
    assert summarize([]) == {"tasks": 0}
