"""원격 LoRA 학습 스테이징 폴더 정리 계약.

워커는 자기 쪽 입력을 정리하지만 로컬에는 학습 이미지 사본이 작업마다 새 폴더로
쌓이기만 했다. 정리를 넣되 **로컬 실행이 공유하는 폴더는 절대 지우지 않는다**는
것이 이 테스트의 핵심이다.
"""

from __future__ import annotations

import os
from pathlib import Path

from queue_manager import _cleanup_remote_training_staging


def _staged(root: Path, relative: str) -> Path:
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    (target / "1_train.png").write_bytes(b"x")
    return target


def test_removes_remote_staging_dir_and_prunes_empty_parents(tmp_path: Path) -> None:
    export_dir = _staged(tmp_path, "soya_lora/modal_jobs/item-1/anima")

    _cleanup_remote_training_staging(str(export_dir))

    assert not export_dir.exists(), "원격 스테이징 폴더가 지워져야 한다"
    assert not (tmp_path / "soya_lora" / "modal_jobs").exists(), (
        "비게 된 modal_jobs 상위 폴더도 함께 정리돼야 한다"
    )


def test_never_prunes_above_modal_jobs(tmp_path: Path) -> None:
    """`soya_lora` 는 설치기가 만드는 공용 입력 폴더다.

    실제로 지워 봤다: 비었다고 위로 계속 올라가면 `soya_lora` 는 물론 `input`
    까지 사라진다. 그러면 다음 학습의 export 가 폴더 없는 상태에서 시작한다.
    아래로 지우는 것만 검사하고 **위를 남기는 것**을 검사하지 않아서 놓쳤다.
    """
    shared = tmp_path / "input" / "soya_lora"
    export_dir = _staged(tmp_path, "input/soya_lora/modal_jobs/item-9/anima")

    _cleanup_remote_training_staging(str(export_dir))

    assert not (shared / "modal_jobs").exists(), "modal_jobs 까지는 지워야 한다"
    assert shared.is_dir(), "공용 입력 폴더 soya_lora 는 남아야 한다"
    assert (tmp_path / "input").is_dir(), "comfy/input 은 절대 지우면 안 된다"


def test_keeps_sibling_jobs_when_pruning(tmp_path: Path) -> None:
    keep = _staged(tmp_path, "soya_lora/modal_jobs/item-2/anima")
    drop = _staged(tmp_path, "soya_lora/modal_jobs/item-1/anima")

    _cleanup_remote_training_staging(str(drop))

    assert not drop.exists()
    assert keep.exists(), "다른 작업의 스테이징까지 지우면 안 된다"
    assert (tmp_path / "soya_lora" / "modal_jobs").is_dir()


def test_never_removes_the_shared_local_folder(tmp_path: Path) -> None:
    # 로컬 실행 경로에는 modal_jobs 조각이 붙지 않는다. 그 폴더를 지우면
    # 다음 로컬 학습의 입력이 사라진다.
    local_dir = _staged(tmp_path, "soya_lora")

    _cleanup_remote_training_staging(str(local_dir))

    assert local_dir.exists(), "로컬 공용 학습 폴더는 지우면 안 된다"
    assert (local_dir / "1_train.png").exists()


def test_missing_or_empty_path_is_not_an_error(tmp_path: Path) -> None:
    _cleanup_remote_training_staging("")
    _cleanup_remote_training_staging(str(tmp_path / "soya_lora" / "modal_jobs" / "gone"))


def test_cleanup_failure_does_not_propagate(tmp_path: Path, monkeypatch) -> None:
    # 정리 실패가 학습 결과를 버리게 두지 않는다 — finally 안에서 호출되기 때문이다.
    export_dir = _staged(tmp_path, "soya_lora/modal_jobs/item-3")

    import queue_manager

    def boom(_path):
        raise PermissionError("정리 실패")

    monkeypatch.setattr(queue_manager.shutil, "rmtree", boom)
    _cleanup_remote_training_staging(str(export_dir))

    assert export_dir.exists()


def test_windows_style_separator_is_recognized(tmp_path: Path) -> None:
    # 경로 판정이 os.sep 분해에 기대므로, 정규화 전 형태로 들어와도 동작해야 한다.
    export_dir = _staged(tmp_path, "soya_lora/modal_jobs/item-4")
    noisy = str(export_dir).replace(os.sep, os.sep + os.sep)

    _cleanup_remote_training_staging(noisy)

    assert not export_dir.exists()
