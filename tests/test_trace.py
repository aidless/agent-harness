"""Tracer: in-memory records, JSONL persistence, summary."""

import json

from agent_harness.trace import Tracer


def test_log_records_in_order():
    t = Tracer()
    t.log("first", x=1)
    t.log("second", x=2)
    assert [r["event"] for r in t.records] == ["first", "second"]
    assert all("t" in r for r in t.records)


def test_summary_counts_events():
    t = Tracer()
    t.log("a"); t.log("a"); t.log("b")
    assert t.summary() == {"steps": 3, "events": {"a": 2, "b": 1}}


def test_writes_jsonl(tmp_path):
    p = tmp_path / "sub" / "trace.jsonl"
    t = Tracer(p)
    t.log("evt", msg="你好")
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "evt" and rec["msg"] == "你好"


def test_tracer_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "dir" / "t.jsonl"
    Tracer(p).log("x")
    assert p.exists()


def test_empty_tracer_no_file():
    t = Tracer()
    t.log("x")
    assert t.path is None
