"""학습은 생성보다 오래 걸리므로 상한을 따로 요구한다.

기본값(3_300초)을 그대로 쓰면 원격 첫 컨테이너는 sd-scripts 런타임 부트스트랩까지
그 안에 끝내야 하고, 초과하면 "학습 실패"가 아니라 조용한 타임아웃으로 나타난다.
"""

from __future__ import annotations

import queue_manager
from modal_backend.settings import MAX_WORKFLOW_TIMEOUT_SECONDS


def test_training_asks_for_more_than_the_generation_default():
    # 생성 기본값 3_300 을 넘어서야 의미가 있다.
    assert queue_manager.LORA_TRAINING_TIMEOUT_SECONDS > 3_300


def test_training_request_fits_under_the_shared_ceiling():
    # 상한보다 크면 클램프에 잘려 상수가 거짓말이 된다.
    assert queue_manager.LORA_TRAINING_TIMEOUT_SECONDS <= MAX_WORKFLOW_TIMEOUT_SECONDS


def test_worker_timeout_exceeds_the_workflow_ceiling():
    """워커가 먼저 죽으면 학습이 끝나 있어도 artifact 를 돌려받지 못한다."""
    from pathlib import Path

    source = Path(queue_manager.__file__).resolve().parents[0]
    app = (source / "modal_backend" / "modal_app.py").read_text(encoding="utf-8")

    assert "timeout=MAX_WORKFLOW_TIMEOUT_SECONDS + 600," in app, (
        "워커 함수 timeout 은 워크플로우 상한보다 커야 한다"
    )


def test_generation_default_is_left_alone():
    """상한만 올리고 기본값은 그대로다 — 짧은 생성 작업의 동작을 바꾸지 않는다."""
    import inspect

    from modal_backend.service import ModalService

    sig = inspect.signature(ModalService.run_workflow)
    assert sig.parameters["timeout_seconds"].default == 3_300
