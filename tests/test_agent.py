"""Agent loop: final answer, tool execution, dedup guard, per-step limit, errors."""

import json

from agent_harness.agent import Agent
from agent_harness.config import Config
from agent_harness.trace import Tracer


class FakeLLM:
    """Returns scripted assistant messages, one per chat() call."""

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


def make_agent(replies, impls=None, max_steps=5):
    cfg = Config(max_steps=max_steps)
    impls = impls or {}
    return Agent(cfg, FakeLLM(replies), [], impls, Tracer()), cfg


def test_final_answer_without_tools():
    agent, _ = make_agent([{"role": "assistant", "content": "done"}])
    r = agent.run("task")
    assert r["finished"] and r["answer"] == "done"
    assert r["steps"] == 1 and r["tool_calls"] == 0


def test_one_tool_call_then_final():
    seen = []

    def impl(x):
        seen.append(x)
        return f"out:{x}"

    agent, _ = make_agent(
        [
            {"role": "assistant", "tool_calls": [call("t", {"x": 1}, "c1")]},
            {"role": "assistant", "content": "finished"},
        ],
        {"t": impl},
    )
    r = agent.run("task")
    assert seen == [1]
    assert r["finished"] and r["tool_calls"] == 1 and r["steps"] == 2
    # tool result was fed back into the conversation
    assert any(m.get("role") == "tool" and m["content"] == "out:1" for m in r["messages"])


def test_duplicate_calls_executed_once():
    n = []

    def impl(x):
        n.append(x)
        return "ok"

    # 46 identical calls, like qwen2.5:7b at temperature=0
    calls = [call("t", {"x": 1}, f"c{i}") for i in range(46)]
    agent, _ = make_agent(
        [
            {"role": "assistant", "tool_calls": calls},
            {"role": "assistant", "content": "done"},
        ],
        {"t": impl},
    )
    r = agent.run("task")
    assert n == [1]  # executed exactly once
    assert r["tool_calls"] == 46  # counted all
    # every call id got a reply message (protocol validity)
    tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 46
    assert sum(m["content"] == "[duplicate call skipped]" for m in tool_msgs) == 45


def test_per_step_execution_limit_eight():
    executed = []

    def impl(i):
        executed.append(i)
        return "ok"

    calls = [call("t", {"i": i}, f"c{i}") for i in range(20)]  # all distinct
    agent, _ = make_agent(
        [{"role": "assistant", "tool_calls": calls}, {"role": "assistant", "content": "done"}],
        {"t": impl},
    )
    r = agent.run("task")
    assert executed == list(range(8))
    tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
    assert sum(m["content"] == "[per-step tool-call limit reached]" for m in tool_msgs) == 12


def test_unknown_tool_and_tool_error():
    from agent_harness.tools import ToolError

    def boom():
        raise ToolError("bad")

    agent, _ = make_agent(
        [
            {"role": "assistant", "tool_calls": [
                call("missing", {}, "c1"), call("boom", {}, "c2"),
            ]},
            {"role": "assistant", "content": "done"},
        ],
        {"boom": boom},
    )
    r = agent.run("task")
    contents = [m["content"] for m in r["messages"] if m.get("role") == "tool"]
    assert any("unknown tool" in c for c in contents)
    assert any("tool error" in c and "bad" in c for c in contents)


def test_malformed_args_become_empty_dict():
    seen = []

    def impl(**kwargs):
        seen.append(kwargs)
        return "ok"

    bad = {"id": "c1", "type": "function",
           "function": {"name": "t", "arguments": "{not json"}}
    agent, _ = make_agent(
        [{"role": "assistant", "tool_calls": [bad]}, {"role": "assistant", "content": "done"}],
        {"t": impl},
    )
    agent.run("task")
    assert seen == [{}]


def test_max_steps_sets_error():
    calls = [call("t", {}, f"c{i}") for i in range(1)]
    agent, _ = make_agent(
        [{"role": "assistant", "tool_calls": calls}] * 3,
        {"t": lambda: "ok"},
        max_steps=3,
    )
    r = agent.run("task")
    assert not r["finished"]
    assert "max_steps" in r["error"]
    assert r["steps"] == 3


def test_llm_error_is_captured():
    from agent_harness.llm import LLMError

    class BadLLM:
        def chat(self, messages, tools=None):
            raise LLMError("boom")

    cfg = Config()
    r = Agent(cfg, BadLLM(), [], {}, Tracer()).run("task")
    assert not r["finished"] and "boom" in r["error"]


def test_count_tokens_config_wires_sizer():
    from agent_harness.agent import Agent
    cfg = Config(count_tokens=True)
    a = Agent(cfg, FakeLLM([{"role": "assistant", "content": "done"}]), [], {}, Tracer())
    assert a.sizer is not None  # tiktoken sizer installed
    assert a.run("task")["finished"]  # still works end to end


def test_count_tokens_false_keeps_char_sizer():
    _, _ = make_agent([])
    cfg = Config(count_tokens=False)
    a = Agent(cfg, FakeLLM([{"role": "assistant", "content": "done"}]), [], {}, Tracer())
    assert a.sizer is None
