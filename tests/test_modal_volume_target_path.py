"""매니페스트 상대 경로 → Volume 절대 경로 변환.

Volume 은 models 폴더 자체를 마운트하므로 선두 "models/" 를 벗겨야 한다. LoRA 는
별도 Volume 이라 종류까지 갈라야 한다 — 틀리면 파일은 올라가지만 ComfyUI 가 LoRA
목록에서 못 찾는다. 둘 다 조용히 어긋나 생성 시점에야 드러난다.
"""

from pathlib import Path

import pytest

from modal_backend.modal_app import volume_target_path

MODELS = Path("/vol/models").resolve()
LORAS = Path("/vol/loras").resolve()


def _target(relative):
    return volume_target_path(MODELS, LORAS, relative)


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("models/vae/x.safetensors", "vae/x.safetensors"),
        ("Models/vae/x.safetensors", "vae/x.safetensors"),
        ("vae/x.safetensors", "vae/x.safetensors"),
        ("models/x.safetensors", "x.safetensors"),
    ],
)
def test_leading_models_segment_is_stripped_once(relative, expected):
    assert _target(relative) == (MODELS / expected, "model")


def test_only_the_first_segment_is_stripped():
    """중간의 models 는 실제 폴더 이름일 수 있다."""
    assert _target("models/models/x.pt") == (MODELS / "models/x.pt", "model")


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("models/loras/char/a.safetensors", "char/a.safetensors"),
        ("loras/a.safetensors", "a.safetensors"),
        ("models/LoRAs/a.safetensors", "a.safetensors"),
    ],
)
def test_loras_go_to_their_own_volume(relative, expected):
    assert _target(relative) == (LORAS / expected, "lora")


def test_lora_prefix_alone_is_rejected():
    with pytest.raises(ValueError):
        _target("models/loras")


@pytest.mark.parametrize("relative", ["", "models", "models/"])
def test_empty_path_is_rejected(relative):
    with pytest.raises(ValueError):
        _target(relative)


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.safetensors",
        "models/../../etc/passwd",
        "/etc/passwd",
        "models/loras/../../escape.safetensors",
    ],
)
def test_paths_escaping_the_volume_are_rejected(relative):
    """매니페스트는 외부 데이터다. Volume 밖으로 쓰게 두면 안 된다."""
    with pytest.raises(ValueError):
        _target(relative)
