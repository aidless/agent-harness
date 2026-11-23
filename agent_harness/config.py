"""Configuration for the agent harness.

Load order: dataclass defaults -> YAML file -> AGENT_* environment variables.
Environment always wins, so a run can be tuned without editing a file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. You have tools that run commands and manage "
    "files in a workspace. Use them when they help. "
    "read_file returns the whole file; write_file OVERWRITES the whole file, "
    "dropping anything you do not include. Before editing an existing file, "
    "read it first, then write back the COMPLETE corrected content, not just "
    "the changed lines."
)


@dataclass
class Config:
    base_url: str = "http://172.17.0.1:11435/v1"
    model: str = "qwen3:8b"
    api_key: str = "local"

    max_steps: int = 20
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 300.0

    # transient HTTP (429/5xx) and network errors: exponential backoff with jitter
    max_retries: int = 5           # number of retries after the first attempt
    retry_base_delay: float = 1.0  # seconds before the first retry
    retry_max_delay: float = 30.0  # cap for any single backoff / Retry-After

    context_window: int = 32768      # budget for kept messages (chars, or tokens if count_tokens)
    max_tool_result_chars: int = 12000  # truncate a single oversized tool result
    count_tokens: bool = False       # opt-in: count the window in exact tokens (needs tiktoken)

    # Verification gate: the agent is not trusted to say "done". When set,
    # verify_command runs in workdir before a final answer is accepted; a
    # non-zero exit feeds its output back to the model for another attempt.
    verify_command: str = ""         # empty => no gate
    verify_timeout: float = 120.0
    max_verify_rounds: int = 2

    # Two-level context compaction: summarize old messages instead of just
    # dropping them. Opt-in because it costs one extra LLM call per compaction.
    compact: bool = False
    compact_keep_recent: int = 4
    compact_after_messages: int = 12

    workdir: str = "."
    trace_path: str = ""             # empty => no trace file
    # Checkpoint/resume: persist the loop (messages + counters) so a long task
    # can survive a process restart. Off by default.
    checkpoint_path: str = ""        # empty => no checkpoint
    checkpoint_dir: str = ""         # serve mode: one checkpoint per session id
    resume: bool = False             # continue from checkpoint_path when it exists
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # ---- loading -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config must be a mapping: {path}")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def apply_env(self) -> "Config":
        """Override fields from AGENT_<FIELD> environment variables."""
        for f in fields(self):
            key = f"AGENT_{f.name.upper()}"
            if key in os.environ:
                raw = os.environ[key]
                cur = getattr(self, f.name)
                if isinstance(cur, bool):
                    val = raw.lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    val = int(raw)
                elif isinstance(cur, float):
                    val = float(raw)
                else:
                    val = raw
                setattr(self, f.name, val)
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls.from_yaml(path) if path else cls()
        return cfg.apply_env()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
