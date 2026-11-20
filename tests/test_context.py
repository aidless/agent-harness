"""Context management: truncation of tool results, dropping old messages."""

from agent_harness.context import fit, tool_result


def test_fit_noop_within_budget():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert fit(msgs, 100, 50) == msgs


def test_fit_truncates_oversized_tool_results_first():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 1000},
        {"role": "user", "content": "U"},
    ]
    out = fit(msgs, 5000, 50)
    assert out[1]["content"].endswith("...[truncated]")
    assert len(out[1]["content"]) < 100
    # input untouched (deepcopy)
    assert len(msgs[1]["content"]) == 1000


def test_fit_keeps_system_and_drops_oldest():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a" * 40},
        {"role": "assistant", "content": "b" * 40},
        {"role": "user", "content": "c" * 40},
    ]
    out = fit(msgs, 100, 50)
    assert out[0]["content"] == "SYS"
    assert out[-1]["content"] == "c" * 40
    assert len(out) < len(msgs)


def test_fit_huge_latest_message_is_kept_truncated():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "z" * 5000},
    ]
    out = fit(msgs, 100, 50)
    assert len(out) == 2
    assert len(out[-1]["content"]) <= 100


def test_fit_empty():
    assert fit([], 100, 50) == []


def test_tool_result_truncates_and_keeps_call_id():
    call = {"id": "call_42", "name": "bash"}
    out = tool_result(call, "y" * 200, 20)
    assert out["role"] == "tool"
    assert out["tool_call_id"] == "call_42"
    assert len(out["content"]) < 40
    assert out["content"].endswith("...[truncated]")


def test_tool_result_short_content_untouched():
    out = tool_result({"id": "1"}, "ok", 20)
    assert out["content"] == "ok"


def test_default_fit_behavior_unchanged():
    """fit() without a sizer still counts characters (backward compatible)."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "a" * 100},
        {"role": "user", "content": "b" * 10},
    ]
    out = fit(msgs, 60, 50)
    assert out[0]["content"] == "S"
    assert out[-1]["content"] == "b" * 10  # newest kept by char budget
    assert len(out) == 2


def test_token_sizer_counts_tokens():
    from agent_harness.context import token_sizer
    sizer = token_sizer()
    assert sizer("hello world") >= 1
    assert sizer("") == 0
    assert sizer("你好世界") > 0
    # special-token-like text must not raise
    assert sizer("<|endoftext|>") >= 1


def test_fit_with_token_sizer():
    from agent_harness.context import token_sizer
    sizer = token_sizer()
    # Chinese chars cost >1 token each, so a token budget drops more messages
    # than the same number as a char budget.
    msgs = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "这是很长的一段中文文本" * 5},
        {"role": "user", "content": "短"},
    ]
    out = fit(msgs, 10, 50, sizer=sizer)
    assert out[0]["content"] == "系统"
    assert out[-1]["content"] == "短"
    # the huge message was dropped
    assert all(m["content"] != "这是很长的一段中文文本" * 5 for m in out)
