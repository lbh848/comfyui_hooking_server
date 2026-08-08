import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.bot_lora_mode import list_bot_trained_steps
from modes.lora_mode import list_trained_steps
from modes.style_lora_mode import list_style_trained_steps


STEP_NAME = "48661245-step00000100"


def _write_step_files(session_dir: Path) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / f"{STEP_NAME}.json").write_text(
        json.dumps(
            {
                "lora_file": f"{STEP_NAME}.safetensors",
                "previews": [f"{STEP_NAME}-1.jpg"],
                "avr_loss": 0.125,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (session_dir / f"{STEP_NAME}.metadata.json").write_text(
        json.dumps(
            {
                "file_name": f"{STEP_NAME}.safetensors",
                "metadata_source": "civitai",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("session_dir_parts", "list_steps", "args"),
    [
        (
            ("Character", "Lora", "Entry", "Session"),
            list_trained_steps,
            ("Character", "Entry", "Session"),
        ),
        (
            ("Bot", "Lora", "Project", "Character", "Session"),
            list_bot_trained_steps,
            ("Bot", "Project", "Character", "Session"),
        ),
        (
            ("anima", "Project", "Session"),
            list_style_trained_steps,
            ("anima", "Project", "Session"),
        ),
    ],
    ids=["asset", "bot", "style"],
)
def test_trained_steps_exclude_companion_metadata_json(
    tmp_path: Path,
    session_dir_parts: tuple[str, ...],
    list_steps,
    args: tuple[str, ...],
) -> None:
    _write_step_files(tmp_path.joinpath(*session_dir_parts))

    steps = list_steps(str(tmp_path), *args)

    assert [step["name"] for step in steps] == [STEP_NAME]
    assert steps[0]["safetensors"] == f"{STEP_NAME}.safetensors"
