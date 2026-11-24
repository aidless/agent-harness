"""CLI wiring: the bench subcommand dispatches to agent_harness.bench."""

import agent_harness.__main__ as cli


def test_bench_subcommand_dispatches(monkeypatch):
    calls = {}

    def fake_run_bench(args):
        calls["tasks"] = args.tasks
        calls["out"] = args.out
        calls["model"] = args.model
        return 0

    monkeypatch.setattr(
        "agent_harness.bench.run_bench", fake_run_bench
    )
    rc = cli.main([
        "bench", "--tasks", "t.jsonl", "--out", "out/",
        "--model", "qwen3:8b",
    ])
    assert rc == 0
    assert calls == {"tasks": "t.jsonl", "out": "out/", "model": "qwen3:8b"}
