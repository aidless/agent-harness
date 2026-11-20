"""Append-only JSONL trace: every LLM reply and tool call, in order."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: str | Path = ""):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **data}
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["event"]] = counts.get(r["event"], 0) + 1
        return {"steps": len(self.records), "events": counts}
