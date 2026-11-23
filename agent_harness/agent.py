"""The agent loop: LLM -> tool calls -> results -> ... -> final answer."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .context import compact, fit, should_compact, tool_result, token_sizer
from .llm import LLM, LLMError
from .tools import ToolError
from .trace import Tracer

# The verifier is a shell command; append a sentinel so its exit code survives
# bash-tool output (which otherwise merges stdout/stderr into one string).
_VERIFY_SENTINEL = "__AH_VERIFY_EXIT__"
_VERIFY_RE = re.compile(rf"^{_VERIFY_SENTINEL}=(\d+)\s*$", re.MULTILINE)

_SUMMARY_PROMPT = (
    "Summarize this earlier part of an agent run so work can continue without "
    "the raw messages. Keep exact file paths, commands, decisions, failures and "
    "what remains to be done. Drop verbatim logs. Under 300 words."
)


class Agent:
    def __init__(self, config, llm: LLM, tools, impls, tracer: Tracer):
        self.cfg = config
        self.llm = llm
        self.tools = tools
        self.impls = impls
        self.tracer = tracer
        # Optional exact-token window. Default stays character-based so the
        # harness has no hard dependency on tiktoken.
        self.sizer = token_sizer() if getattr(config, "count_tokens", False) else None

    def _summarize(self, old_messages: list[dict]) -> str:
        """LLM-backed callback for context.compact (one small, tool-free call)."""
        lines = []
        for m in old_messages:
            role = m.get("role", "?")
            content = str(m.get("content", ""))
            if len(content) > 1500:
                content = content[:1500] + "...[truncated for summary]"
            lines.append(f"[{role}] {content}")
        reply = self.llm.chat([
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ])
        return (reply.get("content") or "")[:4000]

    def _maybe_compact(self, messages: list[dict]) -> None:
        """Fold history into a summary before fit() would silently drop it."""
        if not getattr(self.cfg, "compact", False):
            return
        after = getattr(self.cfg, "compact_after_messages", 12)
        if len(messages) < after:
            return
        if not should_compact(
            messages, self.cfg.context_window, self.cfg.max_tool_result_chars, self.sizer
        ):
            return
        before = len(messages)
        try:
            folded = compact(
                messages, self.cfg.context_window, self.cfg.max_tool_result_chars,
                self._summarize, sizer=self.sizer,
                keep_recent=getattr(self.cfg, "compact_keep_recent", 4),
            )
        except Exception as exc:
            self.tracer.log("compact_failed", error=str(exc)[:300])
            return  # plain fit() still keeps the window valid
        messages[:] = folded
        self.tracer.log("compact", before=before, after=len(messages))

    def _verify(self) -> tuple[bool | None, str]:
        """Run the configured verifier in the task workspace.

        Returns ``(None, "")`` when no verifier is configured, otherwise
        ``(passed, output)``. The command's exit code is captured via a shell
        sentinel so it survives the bash tool's merged-output format.
        """
        command = (getattr(self.cfg, "verify_command", "") or "").strip()
        if not command:
            return None, ""
        wrapped = f"{command} ; echo {_VERIFY_SENTINEL}=$?"
        output = self._execute("bash", {"command": wrapped})
        match = _VERIFY_RE.search(output)
        if match is None:
            return False, output + "\n[verifier produced no exit-code sentinel]"
        passed = match.group(1) == "0"
        clean = _VERIFY_RE.sub("", output).strip()
        return passed, clean

    def _checkpoint(self, task: str, messages: list[dict], result: dict,
                    step: int) -> None:
        """Persist loop state to checkpoint_path (atomic write). Best effort."""
        path = getattr(self.cfg, "checkpoint_path", "") or ""
        if not path:
            return
        state = {
            "task": task,
            "steps": step,
            "messages": messages,
            "answer": result.get("answer", ""),
            "tool_calls": result.get("tool_calls", 0),
            "verify_rounds": result.get("verify_rounds", 0),
            "finished": result.get("finished", False),
            "error": result.get("error", ""),
        }
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)  # atomic: a crash never leaves a half-written file
        except OSError as exc:
            self.tracer.log("checkpoint_failed", error=str(exc)[:300])

    def _restore(self, task: str) -> tuple[list[dict], dict, int] | None:
        """Load a checkpoint for this exact task. Corrupt/mismatched => None."""
        path = getattr(self.cfg, "checkpoint_path", "") or ""
        if not path or not getattr(self.cfg, "resume", False):
            return None
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("task") != task:
                return None
            messages = state["messages"]
            if not isinstance(messages, list):
                return None
            steps = int(state.get("steps", 0))
            result = {
                "answer": state.get("answer", ""),
                "steps": steps,
                "tool_calls": int(state.get("tool_calls", 0)),
                "verify_rounds": int(state.get("verify_rounds", 0)),
                "finished": bool(state.get("finished", False)),
                "error": state.get("error", ""),
                "messages": messages,
            }
            return messages, result, steps
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _text_tool_calls(self, message: dict) -> list[dict]:
        """Tool calls some endpoints emit as JSON text in ``content``.

        Ollama's OpenAI-compatible endpoint, with templates that do not render
        the tools schema, answers ``{"name": ..., "arguments": {...}}`` in the
        content instead of filling ``tool_calls``. Parse that shape strictly so
        ordinary prose is never mistaken for a call: the whole content must be
        one JSON object, or one JSON object per non-empty line.
        """
        content = (message.get("content") or "").strip()
        if not content or '"name"' not in content:
            return []
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        calls: list[dict] = []
        for line in lines:
            # A line that looks like a call but has a typo (a real model habit:
            # one extra closing brace) is skipped; the healthy calls still run.
            looks_like_call = line.startswith("{") and '"name"' in line and '"arguments"' in line
            try:
                obj = json.loads(line)
            except ValueError:
                if looks_like_call:
                    continue
                return []  # prose, not a tool-call batch
            if not isinstance(obj, dict):
                return []
            name = obj.get("name")
            args = obj.get("arguments")
            if not isinstance(name, str) or not name.strip():
                return []
            if not isinstance(args, (dict, str)):
                return []
            raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            calls.append(
                {
                    "id": f"text_{len(calls) + 1}",
                    "type": "function",
                    "function": {"name": name, "arguments": raw},
                }
            )
        return calls

    def run(self, task: str) -> dict:
        restored = self._restore(task)
        if restored is not None and restored[1]["finished"]:
            return restored[1]  # already done: do not call the LLM again
        if restored is not None:
            messages, result, start = restored
            result["error"] = ""  # a resumed run drops the prior stop reason
        else:
            messages: list[dict] = [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": task},
            ]
            result = {
                "answer": "",
                "steps": 0,
                "tool_calls": 0,
                "verify_rounds": 0,
                "finished": False,
                "error": "",
                "messages": messages,
            }
            start = 0
        max_verify_rounds = getattr(self.cfg, "max_verify_rounds", 0)
        try:
            for step in range(start + 1, self.cfg.max_steps + 1):
                result["steps"] = step
                self._maybe_compact(messages)
                window = fit(
                    messages, self.cfg.context_window,
                    self.cfg.max_tool_result_chars, sizer=self.sizer,
                )
                assistant = self.llm.chat(window, self.tools)
                calls = assistant.get("tool_calls") or []
                if not calls:
                    calls = self._text_tool_calls(assistant)
                    if calls:
                        assistant["tool_calls"] = calls
                        self.tracer.log("text_tool_calls", step=step, count=len(calls))
                messages.append(assistant)
                self.tracer.log("llm_reply", step=step, message=assistant)
                if not calls:
                    self._finish_with_verify(result, messages, step, max_verify_rounds)
                    if result["finished"] or result["error"]:
                        break
                    # verification failed but rounds remain: feedback was appended
                    continue

                # Some local models emit a whole batch of identical
                # tool_calls (reproduced: qwen2.5:7b returned 46 copies at
                # temperature 0). Execute each unique call once, but still
                # answer every call id so the protocol stays valid.
                seen: set[tuple[str, str]] = set()
                executed = 0
                for call in calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments") or "{}"
                    args = self._parse_args(raw_args)
                    key = (name, raw_args)
                    if key in seen:
                        output = "[duplicate call skipped]"
                    elif executed >= 8:
                        output = "[per-step tool-call limit reached]"
                    else:
                        seen.add(key)
                        executed += 1
                        output = self._execute(name, args)
                    result["tool_calls"] += 1
                    messages.append(tool_result(call, output, self.cfg.max_tool_result_chars))
                    self.tracer.log("tool_call", step=step, name=name, args=args,
                                    output=output[:1000])
                # Persist after each step so a crash mid-task loses at most one step.
                self._checkpoint(task, messages, result, step)
            else:
                # Out of steps: a passing verifier still counts as success.
                passed, output = self._verify()
                if passed:
                    result["finished"] = True
                    self.tracer.log("verify", step=result["steps"], passed=True,
                                    at_max_steps=True)
                else:
                    result["error"] = f"max_steps ({self.cfg.max_steps}) reached"
                    self.tracer.log("max_steps", step=self.cfg.max_steps)
        except LLMError as exc:
            result["error"] = f"LLM error: {exc}"
            self.tracer.log("error", error=result["error"])
        # Final state (finished / error / max_steps) must also be durable.
        self._checkpoint(task, messages, result, result["steps"])
        return result

    def _finish_with_verify(self, result: dict, messages: list[dict],
                            step: int, max_rounds: int) -> None:
        """Accept a final answer, or feed verification failure back for a retry."""
        answer = result["answer"] or last_assistant_content(messages)
        passed, output = self._verify()
        if passed is None:  # no verifier configured
            result["answer"] = answer
            result["finished"] = True
            self.tracer.log("final", step=step, answer=answer[:500])
            return

        self.tracer.log("verify", step=step, passed=passed,
                        round=result["verify_rounds"] + 1, output=output[:500])
        result["verify_rounds"] += 1
        if passed:
            result["answer"] = answer
            result["finished"] = True
            self.tracer.log("final", step=step, answer=answer[:500])
            return

        rounds = result["verify_rounds"]
        if rounds > max_rounds:
            result["error"] = (
                f"verification failed after {max_rounds} round(s): {output[:300]}"
            )
            return

        feedback = (
            "[verification failed] The verifier command failed. Read its output, "
            "fix the work in the workspace, then finish again.\n"
            f"--- verifier output ---\n{output[:4000]}"
        )
        messages.append({"role": "user", "content": feedback})
        self.tracer.log("verify_feedback", round=rounds)

    @staticmethod
    def _parse_args(raw_args: str) -> dict:
        try:
            return json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return {}

    def _execute(self, name: str, args: dict) -> str:
        impl = self.impls.get(name)
        if impl is None:
            return f"[unknown tool: {name}]"
        try:
            return str(impl(**args))
        except (ToolError, TypeError, OSError) as exc:
            return f"[tool error: {exc}]"


def last_assistant_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])
    return ""
