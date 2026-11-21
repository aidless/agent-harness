"""Context management: keep the conversation inside a character budget.

Strategy is deliberately simple: system prompt is always kept; the newest
messages are kept whole; older tool-result messages are truncated first, then
dropped if still over budget. This avoids the SDK-specific token counting that
breaks on non-OpenAI servers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


def _char_size(text: str) -> int:
    return len(text)


def token_sizer(encoding: str = "cl100k_base") -> Callable[[str], int]:
    """Return an exact token counter (needs tiktoken installed).

    The harness works with no optional deps by default; callers opt in by
    passing this sizer to ``fit``. ``disallowed_special=()`` lets arbitrary
    tool output through without tiktoken rejecting special-token strings.
    """
    import tiktoken  # optional dependency, imported lazily
    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def fit(
    messages: list[dict],
    budget: int,
    max_result_chars: int,
    sizer: Callable[[str], int] | None = None,
) -> list[dict]:
    """Fit messages into ``budget`` units counted by ``sizer`` (default: chars).

    Truncating an oversized tool result is still done by characters — that is
    a coarse safety cut, not a budget decision. Only the keep/drop decision is
    measured in the chosen unit, so enabling token counting never changes the
    shape of the output, just where the boundary lands.
    """
    size = sizer or _char_size

    def _size(msg: dict) -> int:
        return size(str(msg.get("content", "")) + str(msg.get("reasoning", "")))

    msgs = deepcopy(messages)
    if not msgs:
        return msgs

    # Truncate oversized tool outputs everywhere first.
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            if len(m["content"]) > max_result_chars:
                m["content"] = m["content"][:max_result_chars] + "\n...[truncated]"

    if sum(_size(m) for m in msgs) <= budget:
        return msgs

    # Keep system + newest exchanges; drop oldest non-system messages.
    head = [msgs[0]] if msgs[0].get("role") == "system" else []
    body = msgs[len(head):]
    kept: list[dict] = []
    used = sum(_size(m) for m in head)
    for m in reversed(body):
        cost = _size(m)
        if used + cost > budget:
            break
        kept.append(m)
        used += cost
    kept.reverse()

    # If even the newest message alone busts the budget, keep it truncated.
    if not kept and body:
        last = deepcopy(body[-1])
        if isinstance(last.get("content"), str):
            last["content"] = last["content"][:budget]
        kept = [last]

    return head + kept


def tool_result(msg: dict, content: str, max_chars: int) -> dict[str, Any]:
    """Build the tool-result message, truncating oversized output."""
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]"
    return {"role": "tool", "tool_call_id": msg["id"], "content": content}


SUMMARY_PREFIX = "[conversation summary]"


def should_compact(
    messages: list[dict],
    budget: int,
    max_result_chars: int,
    sizer: Callable[[str], int] | None = None,
) -> bool:
    """True when ``fit`` would drop at least one message (history is being lost)."""
    return len(fit(messages, budget, max_result_chars, sizer)) < len(messages)


def compact(
    messages: list[dict],
    budget: int,
    max_result_chars: int,
    summarize: Callable[[list[dict]], str],
    sizer: Callable[[str], int] | None = None,
    keep_recent: int = 6,
) -> list[dict]:
    """Two-level compaction: fold the oldest history into one model summary.

    ``fit`` only truncates/drops — the dropped part is gone forever. Here the
    messages that ``fit`` would discard are first handed to ``summarize`` (an
    LLM-backed callback) and replaced by a single short note, so facts like
    file paths, failed attempts and open work survive a long run.

    System prompt and the newest ``keep_recent`` messages stay verbatim. If the
    summary itself is too large, a final ``fit`` keeps the window safe. On any
    summarizer failure the caller should fall back to plain ``fit``.
    """
    fitted = fit(messages, budget, max_result_chars, sizer)
    if len(fitted) >= len(messages):
        return list(messages)  # nothing would be lost

    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    body = messages[len(head):]
    if len(body) <= keep_recent:
        # Too little history to summarize: leave the list intact so the caller
        # can keep accumulating. Returning `fitted` here would silently drop
        # messages and prevent compaction from ever reaching its threshold.
        return list(messages)
    recent = body[-keep_recent:]
    old = body[:-keep_recent]
    if not old:
        return list(messages)

    note = summarize(old) or ""
    summary = {
        "role": "user",
        "content": f"{SUMMARY_PREFIX}:\n{note.strip()}",
    }
    return fit(head + [summary] + recent, budget, max_result_chars, sizer)
