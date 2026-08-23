import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "comfy"
    / "custom_nodes"
    / "comfyui-soya-custom-nodes"
    / "soya_optional_image_by_name.py"
)
SPEC = importlib.util.spec_from_file_location("soya_optional_image_by_name", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SoyaOptionalImageByName_mdsoya = MODULE.SoyaOptionalImageByName_mdsoya


def test_select_image_returns_the_named_single_image() -> None:
    node = SoyaOptionalImageByName_mdsoya()
    batch = torch.stack(
        [
            torch.zeros((2, 2, 3)),
            torch.ones((2, 2, 3)),
            torch.full((2, 2, 3), 2.0),
        ]
    )

    (selected,) = node.select_image(
        [batch],
        ["[1].png", "[2].png", "[3].png"],
        ["[2]"],
    )

    assert selected.shape == (1, 2, 2, 3)
    assert torch.equal(selected, batch[1:2])


def test_select_image_returns_none_for_an_absent_optional_slot(capsys) -> None:
    node = SoyaOptionalImageByName_mdsoya()
    batch = torch.zeros((1, 2, 2, 3))

    (selected,) = node.select_image([batch], ["[1].png"], ["[3]"])

    assert selected is None
    assert "선택 슬롯 생략" in capsys.readouterr().out
