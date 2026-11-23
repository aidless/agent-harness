"""Tools: path scoping, real file IO, shell execution, schemas."""

import pytest

from agent_harness.tools import ToolError, _resolve, make_tools


def test_resolve_relative_stays_inside(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    p = _resolve(root, "sub/a.txt")
    assert p == (root / "sub" / "a.txt").resolve()


def test_resolve_rejects_escape(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    with pytest.raises(ToolError, match="escapes workdir"):
        _resolve(root, "../secret.txt")


def test_resolve_rejects_absolute_outside(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    with pytest.raises(ToolError, match="escapes workdir"):
        _resolve(root, str(tmp_path / "outside.txt"))


def test_resolve_allows_workdir_itself(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    assert _resolve(root, ".") == root.resolve()


def test_write_read_roundtrip(tmp_path):
    _, impls = make_tools(tmp_path / "w")
    assert "wrote" in impls["write_file"]("a.txt", "hello 世界")
    assert impls["read_file"]("a.txt") == "hello 世界"


def test_read_missing_file_message(tmp_path):
    _, impls = make_tools(tmp_path / "w")
    assert "not found" in impls["read_file"]("nope.txt")


def test_write_creates_parent_dirs(tmp_path):
    _, impls = make_tools(tmp_path / "w")
    impls["write_file"]("deep/nested/f.txt", "x")
    assert (tmp_path / "w" / "deep" / "nested" / "f.txt").read_text() == "x"


def test_bash_runs_in_workdir(tmp_path):
    _, impls = make_tools(tmp_path / "w")
    out = impls["bash"]("pwd")
    assert str((tmp_path / "w").resolve()) in out


def test_bash_captures_stderr(tmp_path):
    _, impls = make_tools(tmp_path / "w")
    out = impls["bash"]("echo boom >&2")
    assert "boom" in out


def test_bash_timeout(tmp_path):
    _, impls = make_tools(tmp_path / "w", timeout=1)
    out = impls["bash"]("sleep 5")
    assert "timeout" in out


def test_schemas_have_four_tools_with_required_params(tmp_path):
    schemas, impls = make_tools(tmp_path / "w")
    assert set(impls) == {"bash", "read_file", "write_file", "edit_file"}
    names = {s["function"]["name"] for s in schemas}
    assert names == set(impls)
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["required"]
