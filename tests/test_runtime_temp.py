from pathlib import Path

from runtime_temp import clear_runtime_temp, runtime_temp_root


def test_clear_runtime_temp_removes_only_temp_contents(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    temp_root = runtime_temp_root(project_root)
    nested = temp_root / "stale-download" / "nested"
    nested.mkdir(parents=True)
    (nested / "artifact.part").write_bytes(b"partial")
    (temp_root / "stale-file.tmp").write_bytes(b"temporary")
    runtime_sibling = project_root / "runtime" / "keep.json"
    runtime_sibling.write_text("keep", encoding="utf-8")

    result = clear_runtime_temp(project_root)

    assert result == temp_root
    assert temp_root.is_dir()
    assert list(temp_root.iterdir()) == []
    assert runtime_sibling.read_text(encoding="utf-8") == "keep"


def test_runtime_temp_root_is_inside_project_runtime_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"

    result = runtime_temp_root(project_root)

    assert result == (project_root / "runtime" / "temp").resolve()
    assert result.is_dir()
