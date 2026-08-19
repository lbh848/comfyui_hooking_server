"""매니페스트 상대 경로 → models Volume 절대 경로 변환.

Volume 은 models 폴더 자체를 마운트하므로 선두 "models/" 를 벗겨야 한다. 이걸
틀리면 파일이 조용히 엉뚱한 자리에 쌓이고, 생성 시점에야 없다는 걸 알게 된다.
경로가 Volume 밖으로 나가는 것도 여기서 막는다.
"""

from pathlib import Path

import pytest

from modal_backend.modal_app import volume_target_path

ROOT = Path("/vol/models").resolve()


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
    assert volume_target_path(ROOT, relative) == ROOT / expected


def test_only_the_first_segment_is_stripped():
    """중간의 models 는 실제 폴더 이름일 수 있다."""
    assert volume_target_path(ROOT, "models/models/x.pt") == ROOT / "models/x.pt"


@pytest.mark.parametrize("relative", ["", "models", "models/"])
def test_empty_path_is_rejected(relative):
    with pytest.raises(ValueError):
        volume_target_path(ROOT, relative)


@pytest.mark.parametrize(
    "relative",
    ["../outside.safetensors", "models/../../etc/passwd", "/etc/passwd"],
)
def test_paths_escaping_the_volume_are_rejected(relative):
    """매니페스트는 외부 데이터다. 볼륨 밖으로 쓰게 두면 안 된다."""
    with pytest.raises(ValueError):
        volume_target_path(ROOT, relative)
