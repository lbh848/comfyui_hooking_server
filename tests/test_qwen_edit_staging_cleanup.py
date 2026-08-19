"""Qwen Edit 스테이징 폴더가 실행 뒤 디스크에 남던 문제 (F8).

MACOS_SETUP_LOG.md §4-18 의 '사소한 관찰': cleanup_staged_request() 는 메모리
dict 만 비웠고, comfy/input/qwen_edit/<job> 은 다음 실행의 _reset_shared_input_dir
가 지울 때까지 남았다. 무한 증가는 아니지만(최대 직전 1건) 지울 이유는 분명하다.

정리 실패가 작업 실패가 되면 안 되므로, 경로가 이상하면 조용히 건너뛴다.
"""

import os
from pathlib import Path

from modes.qwen_edit_mode import QWEN_EDIT_INPUT_SUBDIR, QwenEditMode


def _mode() -> QwenEditMode:
    mode = object.__new__(QwenEditMode)
    mode._pending_inputs = {}
    return mode


def _staged(tmp_path: Path, job_id: str = "job-1") -> tuple[dict, Path]:
    input_dir = tmp_path / "input"
    folder = input_dir / QWEN_EDIT_INPUT_SUBDIR / job_id
    folder.mkdir(parents=True)
    (folder / "source.png").write_bytes(b"src")
    (folder / "mask.png").write_bytes(b"mask")
    return {"comfy_input_dir": str(input_dir)}, folder


def test_cleanup_removes_the_staged_folder(tmp_path):
    mode = _mode()
    mode._pending_inputs["job-1"] = {"source": b"src", "mask": b"mask"}
    config, folder = _staged(tmp_path)
    assert folder.is_dir()

    mode.cleanup_staged_request({"job_id": "job-1"}, config)

    assert not folder.exists()
    assert "job-1" not in mode._pending_inputs


def test_cleanup_without_config_keeps_previous_behaviour(tmp_path):
    """설정이 없으면 메모리만 비운다 — 지울 경로를 모르기 때문이다."""
    mode = _mode()
    mode._pending_inputs["job-1"] = {"source": b"src", "mask": b"mask"}
    _config, folder = _staged(tmp_path)

    mode.cleanup_staged_request({"job_id": "job-1"})

    assert folder.is_dir()
    assert "job-1" not in mode._pending_inputs


def test_cleanup_leaves_sibling_jobs_alone(tmp_path):
    mode = _mode()
    config, folder = _staged(tmp_path, "job-1")
    _config2, other = _staged(tmp_path, "job-2")

    mode.cleanup_staged_request({"job_id": "job-1"}, config)

    assert not folder.exists()
    assert other.is_dir()


def test_cleanup_refuses_paths_outside_the_input_folder(tmp_path):
    """지우는 쪽이 더 위험하다 — 배치 때와 같은 경로 검증을 다시 한다."""
    mode = _mode()
    input_dir = tmp_path / "input"
    (input_dir / QWEN_EDIT_INPUT_SUBDIR).mkdir(parents=True)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"keep")

    mode.cleanup_staged_request(
        {"job_id": "../../precious"}, {"comfy_input_dir": str(input_dir)}
    )

    assert outside.is_dir()
    assert (outside / "keep.txt").is_file()


def test_cleanup_survives_a_missing_folder(tmp_path):
    """이미 없는 것을 지우려 해도 예외가 나가면 안 된다."""
    mode = _mode()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    mode.cleanup_staged_request(
        {"job_id": "never-staged"}, {"comfy_input_dir": str(input_dir)}
    )


def test_queue_manager_passes_config_to_cleanup():
    source = (Path(__file__).resolve().parents[1] / "queue_manager.py").read_text(
        encoding="utf-8"
    )
    assert source.count("cleanup_staged_request(") == 2
    assert "self.get_config() if self.get_config else None" in source
