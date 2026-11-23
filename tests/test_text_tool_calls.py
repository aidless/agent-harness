"""Compat: models that emit tool calls as JSON text in content (e.g. Ollama/Qwen)."""

import json

from agent_harness.agent import Agent
from agent_harness.config import Config
from agent_harness.trace import Tracer


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen = []

    def chat(self, messages, tools=None):
        self.calls += 1
        self.seen.append([m.get("role") for m in messages])
        if self.calls <= len(self.replies):
            return self.replies[self.calls - 1]
        return {"role": "assistant", "content": "fallback"}


def make(replies, impls=None):
    return Agent(Config(max_steps=6), FakeLLM(replies), [], impls or {}, Tracer())


def as_call(name, arguments):
    """A content string shaped like a tool call."""
    return json.dumps({"name": name, "arguments": arguments})


def test_parses_single_line_json_call():
    a = make([])
    call = a._text_tool_calls({"role": "assistant", "content": as_call("bash", {"command": "ls"})})
    assert len(call) == 1
    assert call[0]["function"]["name"] == "bash"
    assert json.loads(call[0]["function"]["arguments"]) == {"command": "ls"}


def test_parses_multiline_json_calls():
    content = as_call("bash", {"command": "pytest"}) + chr(10) + as_call("read_file", {"path": "a.py"})
    calls = make([])._text_tool_calls({"role": "assistant", "content": content})
    assert [c["function"]["name"] for c in calls] == ["bash", "read_file"]


def test_plain_text_is_not_a_call():
    for content in ["The bug is in add(): it subtracts.", "I will now run pytest.", ""]:
        assert make([])._text_tool_calls({"role": "assistant", "content": content}) == []


def test_malformed_or_prose_wrapped_is_not_a_call():
    bad = [
        '{"name": "bash"}',
        '{"arguments": {"command": "ls"}}',
        '{"name": 5, "arguments": {}}',
        '{"name": "bash", "arguments": 3}',
        'Here you go: {"name": "bash", "arguments": {"command": "ls"}}',
        '{"name": "bash", "arguments": {"command": "ls"}}' + chr(10) + 'thanks',
    ]
    for content in bad:
        assert make([])._text_tool_calls({"role": "assistant", "content": content}) == [], content


def test_arguments_given_as_string_is_kept():
    content = as_call("bash", json.dumps({"command": "ls"}))
    calls = make([])._text_tool_calls({"role": "assistant", "content": content})
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


def test_end_to_end_executes_text_call_then_finishes():
    ran = []

    def bash(command, timeout=None):
        ran.append(command)
        return "executed:" + command

    agent = make(
        [
            {"role": "assistant", "content": as_call("bash", {"command": "pytest"})},
            {"role": "assistant", "content": "all tests pass"},
        ],
        {"bash": bash},
    )
    r = agent.run("fix it")
    assert ran == ["pytest"]
    assert r["tool_calls"] == 1
    assert r["finished"] and r["answer"] == "all tests pass"
    assert "tool" in [m.get("role") for m in r["messages"]]


def test_malformed_json_line_is_skipped_but_prose_aborts():
    good = as_call("bash", {"command": "pytest"})
    broken = good + "}"  # a real model typo: one extra closing brace
    content = broken + chr(10) + good
    calls = make([])._text_tool_calls({"role": "assistant", "content": content})
    assert [c["function"]["name"] for c in calls] == ["bash"]
    assert make([])._text_tool_calls({"role": "assistant", "content": good + chr(10) + "thanks"}) == []
