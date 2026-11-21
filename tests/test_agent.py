"""Agent loop: final answer, tool execution, dedup guard, per-step limit, errors."""

import json
from pathlib import Path

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


# ---- P0: verification gate -------------------------------------------------
def _agent_with_verify(replies, verify_command, max_verify_rounds=2, workdir=None):
    import tempfile
    from agent_harness.tools import make_tools

    workdir = workdir or tempfile.mkdtemp(prefix="ah-verify-")
    cfg = Config(max_steps=6, verify_command=verify_command,
                 max_verify_rounds=max_verify_rounds, workdir=workdir)
    _, impls = make_tools(workdir)
    return Agent(cfg, FakeLLM(replies), [], impls, Tracer()), cfg


def test_verify_passes_when_command_succeeds():
    agent, _ = _agent_with_verify(
        [{"role": "assistant", "content": "done"}], "true"
    )
    r = agent.run("task")
    assert r["finished"] and r["answer"] == "done"
    assert r["verify_rounds"] == 1


def test_verify_failure_feeds_back_then_succeeds():
    # First "done" is rejected; second attempt is accepted. The verifier
    # command reads a file the model writes via a tool on round 2.
    agent, _ = _agent_with_verify(
        [
            {"role": "assistant", "content": "done too early"},
            {"role": "assistant", "content": "now really done"},
        ],
        "test -f marker.txt",
    )
    # Create the marker on the SECOND call (FakeLLM increments `calls` inside
    # base_chat, so before call 2 the counter reads 1).
    base_chat = agent.llm.chat
    def chat_with_side_effect(messages, tools=None):
        if agent.llm.calls == 1:
            (Path(agent.cfg.workdir) / "marker.txt").write_text("ok")
        return base_chat(messages, tools)
    agent.llm.chat = chat_with_side_effect

    r = agent.run("task")
    assert r["finished"] and r["answer"] == "now really done"
    assert r["verify_rounds"] == 2
    assert any("[verification failed]" in m.get("content", "")
               for m in r["messages"] if m.get("role") == "user")


def test_verify_exhausts_rounds_sets_error():
    agent, _ = _agent_with_verify(
        [{"role": "assistant", "content": "done"}] * 5,
        "false",
        max_verify_rounds=2,
    )
    r = agent.run("task")
    assert not r["finished"]
    assert "verification failed after 2" in r["error"]
    assert r["verify_rounds"] == 3  # first + 2 retries


def test_max_steps_with_passing_verifier_counts_as_success():
    # The model never finishes, but the verifier passes: work is done.
    agent, _ = _agent_with_verify(
        [{"role": "assistant", "content": "working"}] * 3, "true"
    )
    agent.cfg.max_steps = 3
    r = agent.run("task")
    assert r["finished"] and "max_steps" not in r["error"]


# ---- P0: context compaction ------------------------------------------------
def test_should_compact_and_compact_fold():
    from agent_harness.context import compact, should_compact, SUMMARY_PREFIX

    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "user", "content": "NEW"},
    ]
    assert should_compact(msgs, 120, 200) is True
    folded = compact(msgs, 120, 200, lambda old: f"summary of {len(old)} messages",
                     keep_recent=1)
    assert folded[0]["content"] == "SYS"
    assert any(SUMMARY_PREFIX in m.get("content", "") for m in folded)
    assert folded[-1]["content"] == "NEW"
    assert len(folded) < len(msgs)


def test_compact_noop_when_within_budget():
    from agent_harness.context import compact

    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert compact(msgs, 1000, 100, lambda old: "x") == msgs


def test_agent_compact_calls_summarizer_on_long_history():
    from agent_harness.context import SUMMARY_PREFIX

    summarized = []
    class SummarizingLLM(FakeLLM):
        def chat(self, messages, tools=None):
            if messages and messages[0].get("role") == "system" and \
               "Summarize this earlier part" in messages[0].get("content", ""):
                summarized.append(len(messages))
                return {"role": "assistant", "content": "notes: added file X"}
            return super().chat(messages, tools)

    # Two tool-call rounds grow the conversation past compact_after_messages,
    # then a final answer. The long tool results push it over the tiny budget.
    long_out = "x" * 60
    replies = [
        {"role": "assistant", "tool_calls": [
            call("t", {"x": long_out}, "c1")]},
        {"role": "assistant", "tool_calls": [
            call("t", {"x": long_out}, "c2")]},
        {"role": "assistant", "content": "final"},
    ]
    cfg = Config(max_steps=5, compact=True, compact_after_messages=4,
                 context_window=90, max_tool_result_chars=100)
    llm = SummarizingLLM(replies)
    agent = Agent(cfg, llm, [], {"t": lambda x: x}, Tracer())
    r = agent.run("Do lots of work")
    assert summarized  # summarizer was actually invoked
    assert r["finished"]
    # With a tiny budget the final fit() may drop the summary again; what this
    # test proves is that the LLM-backed summarizer path actually ran. The
    # output shape of compact() itself is checked in test_compact_fold above.


def test_agent_compact_failure_falls_back_to_plain_fit():
    class BrokenSummaryLLM(FakeLLM):
        def chat(self, messages, tools=None):
            if messages and messages[0].get("content", "").startswith("Summarize"):
                from agent_harness.llm import LLMError
                raise LLMError("summarizer down")
            return super().chat(messages, tools)

    long_out = "x" * 60
    replies = [
        {"role": "assistant", "tool_calls": [call("t", {"x": long_out}, "c1")]},
        {"role": "assistant", "tool_calls": [call("t", {"x": long_out}, "c2")]},
        {"role": "assistant", "content": "ok"},
    ]
    cfg = Config(max_steps=5, compact=True, compact_after_messages=4,
                 context_window=90, max_tool_result_chars=100)
    agent = Agent(cfg, BrokenSummaryLLM(replies), [], {"t": lambda x: x}, Tracer())
    r = agent.run("task")
    assert r["finished"] and r["answer"] == "ok"  # no crash, plain fit used

