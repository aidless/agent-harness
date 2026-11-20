"""The agent loop: LLM -> tool calls -> results -> ... -> final answer."""

from __future__ import annotations

import json
from typing import Any

from .context import fit, tool_result, token_sizer
from .llm import LLM, LLMError
from .tools import ToolError
from .trace import Tracer


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

    def run(self, task: str) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": task},
        ]
        result = {
            "answer": "",
            "steps": 0,
            "tool_calls": 0,
            "finished": False,
            "error": "",
            "messages": messages,
        }
        try:
            for step in range(1, self.cfg.max_steps + 1):
                result["steps"] = step
                window = fit(
                    messages, self.cfg.context_window,
                    self.cfg.max_tool_result_chars, sizer=self.sizer,
                )
                assistant = self.llm.chat(window, self.tools)
                messages.append(assistant)
                self.tracer.log("llm_reply", step=step, message=assistant)

                calls = assistant.get("tool_calls") or []
                if not calls:
                    result["answer"] = assistant.get("content") or ""
                    result["finished"] = True
                    self.tracer.log("final", step=step, answer=result["answer"][:500])
                    break

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
            else:
                result["error"] = f"max_steps ({self.cfg.max_steps}) reached"
                self.tracer.log("max_steps", step=self.cfg.max_steps)
        except LLMError as exc:
            result["error"] = f"LLM error: {exc}"
            self.tracer.log("error", error=result["error"])
        return result

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
