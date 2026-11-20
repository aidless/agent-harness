"""Config: defaults, YAML filtering, env overrides, load priority."""

import pytest

from agent_harness.config import Config, DEFAULT_SYSTEM_PROMPT


def test_defaults():
    c = Config()
    assert c.model == "qwen3:8b"
    assert c.max_steps == 20
    assert c.temperature == 0.0
    assert c.system_prompt == DEFAULT_SYSTEM_PROMPT
    # The prompt must avoid the phrase that makes qwen2.5:7b stop calling tools.
    assert "Verify your work" not in c.system_prompt


def test_from_yaml_filters_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: test-model\nmax_steps: 3\nnot_a_field: 1\n", encoding="utf-8")
    c = Config.from_yaml(p)
    assert c.model == "test-model"
    assert c.max_steps == 3
    assert not hasattr(c, "not_a_field")


def test_from_yaml_rejects_non_mapping(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        Config.from_yaml(p)


def test_apply_env_types(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "env-model")
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.5")
    c = Config().apply_env()
    assert c.model == "env-model"
    assert c.max_steps == 7 and isinstance(c.max_steps, int)
    assert c.temperature == 0.5 and isinstance(c.temperature, float)


def test_env_wins_over_yaml(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("model: yaml-model\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MODEL", "env-model")
    assert Config.load(p).model == "env-model"


def test_to_dict_roundtrip():
    d = Config(model="x").to_dict()
    assert d["model"] == "x"
    assert set(d) == {f for f in Config().__dict__}
