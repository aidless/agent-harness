"""edit_file: exact-substring replace that leaves the rest of a file untouched."""

import pytest

from agent_harness.tools import ToolError, _resolve, make_tools


def _impls(tmp_path):
    return make_tools(tmp_path / "w")[1]


def test_edit_replaces_exact_substring(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "return a - b\n")
    out = impls["edit_file"]("a.txt", "return a - b", "return a + b")
    assert "edited" in out
    assert impls["read_file"]("a.txt") == "return a + b\n"


def test_edit_only_touches_matched_region(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "line1\nline2\nline3\n")
    impls["edit_file"]("a.txt", "line2", "LINE2")
    assert impls["read_file"]("a.txt") == "line1\nLINE2\nline3\n"


def test_edit_missing_old_reports_error(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "hello\n")
    out = impls["edit_file"]("a.txt", "absent", "x")
    assert "not found" in out
    assert impls["read_file"]("a.txt") == "hello\n"


def test_edit_multiple_matches_needs_replace_all(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "a\nx\na\n")
    out = impls["edit_file"]("a.txt", "a", "b")
    assert "match" in out
    assert impls["read_file"]("a.txt") == "a\nx\na\n"


def test_edit_replace_all_replaces_every_match(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "a\nx\na\n")
    impls["edit_file"]("a.txt", "a", "b", replace_all=True)
    assert impls["read_file"]("a.txt") == "b\nx\nb\n"


def test_edit_empty_old_rejected(tmp_path):
    impls = _impls(tmp_path)
    impls["write_file"]("a.txt", "data\n")
    out = impls["edit_file"]("a.txt", "", "x")
    assert "empty" in out
    assert impls["read_file"]("a.txt") == "data\n"


def test_edit_missing_file_reports_not_found(tmp_path):
    impls = _impls(tmp_path)
    out = impls["edit_file"]("nope.txt", "a", "b")
    assert "not found" in out


def test_edit_preserves_docstring_in_bugfix_scenario(tmp_path):
    """The real failure: fix add() without corrupting the triple-quoted docstring."""
    impls = _impls(tmp_path)
    original = (
        '"""Small arithmetic helpers, with a bug to find."""\n'
        "\n\n"
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a - b          # BUG\n"
    )
    impls["write_file"]("calc.py", original)
    impls["edit_file"]("calc.py", "return a - b          # BUG", "return a + b")
    fixed = impls["read_file"]("calc.py")
    assert '"""Small arithmetic helpers, with a bug to find."""\n' in fixed
    assert '"""Return the sum of a and b."""\n' in fixed
    assert "return a + b" in fixed
    compile(fixed, "calc.py", "exec")  # must still be valid Python


def test_schemas_include_edit_file(tmp_path):
    schemas, impls = make_tools(tmp_path / "w")
    names = {s["function"]["name"] for s in schemas}
    assert "edit_file" in names
    assert set(impls) == {"bash", "read_file", "write_file", "edit_file"}
    schema = next(s for s in schemas if s["function"]["name"] == "edit_file")
    props = schema["function"]["parameters"]["properties"]
    assert {"path", "old", "new"} <= set(props)
    assert set(schema["function"]["parameters"]["required"]) == {"path", "old", "new"}
