"""Checkpoint/resume: persist the loop so a long task survives a restart."""

import json

from agent_harness.agent import Agent
from agent_harness.config import Config
from agent_harness.trace import Tracer


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls <= len(self.replies):
            return self.replies[self.calls - 1]
        return {"role": "assistant", "content": "fallback"}


def call(name, args, cid):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_no_checkpoint_file_when_disabled(tmp_path):
    path = tmp_path / "ck.json"
    cfg = Config(max_steps=5)  # checkpoint_path defaults to ""
    agent = Agent(cfg, FakeLLM([{"role": "assistant", "content": "done"}]),
                  [], {}, Tracer())
    agent.run("task")
    assert not path.exists()


def test_checkpoint_written_after_each_step(tmp_path):
    path = tmp_path / "ck.json"

    def impl(x):
        return f"out:{x}"

    cfg = Config(max_steps=5, checkpoint_path=str(path))
    agent = Agent(
        cfg,
        FakeLLM([
            {"role": "assistant", "tool_calls": [call("t", {"x": 1}, "c1")]},
            {"role": "assistant", "content": "finished"},
        ]),
        [], {"t": impl}, Tracer(),
    )
    r = agent.run("task")
    assert r["finished"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["finished"] is True
    assert data["task"] == "task"
    assert data["steps"] == 2 and data["tool_calls"] == 1
    # the tool result survives in the persisted conversation
    assert any(m.get("role") == "tool" and m.get("content") == "out:1"
               for m in data["messages"])


def test_resume_continues_without_rerunning_earlier_steps(tmp_path):
    path = tmp_path / "ck.json"
    seen = []

    def impl(x):
        seen.append(x)
        return f"out:{x}"

    # First run is cut short by max_steps=1: one tool call executed, then out of steps.
    cfg1 = Config(max_steps=1, checkpoint_path=str(path))
    a1 = Agent(cfg1, FakeLLM([
        {"role": "assistant", "tool_calls": [call("t", {"x": 1}, "c1")]},
    ]), [], {"t": impl}, Tracer())
    r1 = a1.run("task")
    assert not r1["finished"] and "max_steps" in r1["error"]
    assert seen == [1]

    # Resume: must pick up at step 2, not replay the tool call, and finish.
    cfg2 = Config(max_steps=5, checkpoint_path=str(path), resume=True)
    llm2 = FakeLLM([{"role": "assistant", "content": "finished"}])
    a2 = Agent(cfg2, llm2, [], {"t": impl}, Tracer())
    r2 = a2.run("task")
    assert r2["finished"] and r2["answer"] == "finished"
    assert r2["steps"] == 2
    assert r2["tool_calls"] == 1          # counter carried over, not doubled
    assert seen == [1]                    # tool not executed twice
    assert llm2.calls == 1                # only the new step hit the LLM


def test_resume_of_finished_run_makes_no_llm_call(tmp_path):
    path = tmp_path / "ck.json"
    cfg1 = Config(max_steps=5, checkpoint_path=str(path))
    a1 = Agent(cfg1, FakeLLM([{"role": "assistant", "content": "done"}]), [], {}, Tracer())
    a1.run("task")

    llm2 = FakeLLM([{"role": "assistant", "content": "should not be used"}])
    cfg2 = Config(max_steps=5, checkpoint_path=str(path), resume=True)
    a2 = Agent(cfg2, llm2, [], {}, Tracer())
    r2 = a2.run("task")
    assert r2["finished"] and r2["answer"] == "done"
    assert llm2.calls == 0


def test_corrupt_checkpoint_starts_fresh(tmp_path):
    path = tmp_path / "ck.json"
    path.write_text("{not json", encoding="utf-8")
    llm = FakeLLM([{"role": "assistant", "content": "fresh"}])
    cfg = Config(max_steps=5, checkpoint_path=str(path), resume=True)
    agent = Agent(cfg, llm, [], {}, Tracer())
    r = agent.run("task")
    assert r["finished"] and r["answer"] == "fresh"
    assert llm.calls == 1