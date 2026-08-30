"""원격 LoRA 결과는 lora_load_path 아래에만 저장된다.

원격 artifact 의 상대 경로는 `SOYA_INSTANCE_LORA/...` 처럼 언제나 그 루트 아래를
가리킨다. 종류별 로드 경로가 밖으로 나가 있으면 저장은 성공하는데 피커는 다른
폴더를 읽어, 결과가 사라진 것처럼 보인다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_backend.service import ModalService


def _artifact(tmp_path: Path) -> list[dict]:
    source = tmp_path / "downloaded.safetensors"
    source.write_bytes(b"lora")
    return [
        {
            "path": str(source),
            "relative_path": "SOYA_INSTANCE_LORA/anima/probe/lora.safetensors",
            "remote_path": "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/anima/probe/lora.safetensors",
        }
    ]


def _service(tmp_path: Path) -> ModalService:
    return ModalService(tmp_path, lambda: {"modal_enabled": True})


def test_store_rejects_scoped_root_outside_lora_load_path(tmp_path: Path) -> None:
    root = tmp_path / "loras" / "SOYA_CHAR_LORA"
    config = {
        "lora_load_path": str(root),
        "instance_lora_load_path": str(tmp_path / "elsewhere" / "SOYA_INSTANCE_LORA"),
    }

    with pytest.raises(ValueError) as excinfo:
        _service(tmp_path)._store_modal_artifacts(_artifact(tmp_path), config)

    assert "instance_lora_load_path" in str(excinfo.value)


def test_store_accepts_nested_scoped_roots(tmp_path: Path) -> None:
    root = tmp_path / "loras" / "SOYA_CHAR_LORA"
    config = {
        "lora_load_path": str(root),
        "bot_lora_load_path": str(root / "SOYA_BOT_LORA"),
        "instance_lora_load_path": str(root / "SOYA_INSTANCE_LORA"),
        "style_lora_load_path": str(root / "SOYA_STYLE_LORA"),
    }

    stored = _service(tmp_path)._store_modal_artifacts(_artifact(tmp_path), config)

    assert len(stored) == 1
    assert Path(stored[0]["local_path"]).is_file()
    assert stored[0]["status"] == "stored"


def test_store_allows_empty_scoped_roots_that_fall_back(tmp_path: Path) -> None:
    # 비어 있으면 lora_load_path 아래로 폴백한다 — 설치기 기본값이 아닌 구성도 있다.
    root = tmp_path / "loras" / "SOYA_CHAR_LORA"
    config = {"lora_load_path": str(root), "instance_lora_load_path": ""}

    stored = _service(tmp_path)._store_modal_artifacts(_artifact(tmp_path), config)

    assert len(stored) == 1
