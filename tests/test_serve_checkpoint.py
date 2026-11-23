"""Serve layer: per-session checkpoint so a self-hosted deployment can resume."""

import os

from agent_harness.config import Config
from agent_harness.serve import session_checkpoint


def test_no_dir_leaves_checkpoint_off():
    cfg = Config()
    session_checkpoint(cfg, {"session": "abc", "instruction": "do it"})
    assert cfg.checkpoint_path == "" and cfg.resume is False


def test_dir_derives_per_session_path():
    cfg = Config(checkpoint_dir="/tmp/ck")
    session_checkpoint(cfg, {"session": "task-42", "instruction": "do it"})
    assert cfg.checkpoint_path == os.path.join("/tmp/ck", "task-42.json")
    assert cfg.resume is True


def test_unsafe_session_id_is_sanitised():
    cfg = Config(checkpoint_dir="/tmp/ck")
    session_checkpoint(cfg, {"session": "../../etc/passwd"})
    name = os.path.basename(cfg.checkpoint_path)
    assert "/" not in name and "\\" not in name
    assert name.endswith(".json")


def test_missing_session_id_leaves_checkpoint_off():
    cfg = Config(checkpoint_dir="/tmp/ck")
    session_checkpoint(cfg, {"instruction": "do it"})
    assert cfg.checkpoint_path == ""
