"""Installation layout shared by the existing engine and future application UIs.

Source and bundled defaults live in src; user settings and writable content live
beside it in config and data. A project_root argument denotes the source folder.
"""
from __future__ import annotations

from pathlib import Path
import shutil


SOURCE_DIR = Path(__file__).resolve().parent
INSTALL_DIR = SOURCE_DIR.parent
CONFIG_DIR = INSTALL_DIR / "config"
DATA_DIR = INSTALL_DIR / "data"


def install_root(project_root: str | Path = SOURCE_DIR) -> Path:
    root = Path(project_root).resolve()
    return root.parent if root.name == "src" else root


def data_root(project_root: str | Path = SOURCE_DIR) -> Path:
    return install_root(project_root) / "data"


def comfy_input_root(project_root: str | Path = SOURCE_DIR) -> Path:
    """The managed Comfy installation always owns this input directory."""
    return data_root(project_root) / "comfy" / "input"


def config_root(project_root: str | Path = SOURCE_DIR) -> Path:
    return install_root(project_root) / "config"


def resolve_install_path(value: str, project_root: str | Path = SOURCE_DIR) -> Path:
    """Resolve portable resource paths independently of the working directory."""
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else install_root(project_root) / path


def initialize_layout(project_root: str | Path = SOURCE_DIR) -> None:
    """Create writable roots and seed missing defaults without replacing edits."""
    source = Path(project_root).resolve()
    config_root(source).mkdir(parents=True, exist_ok=True)
    destination = data_root(source)
    destination.mkdir(parents=True, exist_ok=True)
    for bundled_name, data_name in (
        ("prompts", "prompts"),
        ("auto_complete", "auto_complete"),
        ("defaults/illustration", "asset_data"),
    ):
        bundled = source / bundled_name
        if not bundled.is_dir():
            continue
        for original in bundled.rglob("*"):
            if not original.is_file() or "__pycache__" in original.parts:
                continue
            target = destination / data_name / original.relative_to(bundled)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as output, original.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            except FileExistsError:
                pass
    from data_backups import backup_presets
    backup_presets(source)


if __name__ == "__main__":
    initialize_layout()
