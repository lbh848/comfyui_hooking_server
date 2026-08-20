"""로컬 모델 결손 점검과 클라우드 전용 판정 회귀 테스트.

배경: 설치기가 cloud_direct 에서 원격 위임분을 로컬에 받지 않게 되면서(C1),
작업 배분을 원격 → 로컬로 되돌리면 "그 작업이 쓰는 모델이 로컬에 없는" 상태가
성립하게 됐다. 그 실패는 ComfyUI 안에서 `... not in []` 로만 나타나 원인을 알기
어렵다(MODEL_SYNC_DIRECTION.md §5 가 '조용한 불일치'로 지목한 바로 그 종류).

FIX_AND_TEST_PLAN.md §4 (F3·F4) 참고.
"""

import json
from pathlib import Path

from comfy_allocation import (
    COMFY_TASK_KEYS,
    MODAL_SUPPORTED_COMFY_TASK_KEYS,
    cloud_only_assessment,
)
from comfy_installer.model_scope import (
    configured_binding_ids,
    local_model_gaps,
    tasks_needing_model,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "comfy_installer" / "resources" / "install_manifest.json").read_text(
        encoding="utf-8"
    )
)
WORKFLOWS = MANIFEST["workflows"]
MODELS = list(MANIFEST["models"])


def _all_remote() -> dict:
    result = {key: 1 for key in COMFY_TASK_KEYS}
    for key in MODAL_SUPPORTED_COMFY_TASK_KEYS:
        result[key] = "modal"
    return result


def _one_task_local() -> dict:
    """전부 원격이되 face_extract 만 로컬.

    실제 구성은 로컬 모델이 0개라 "빠진 모델을 보고한다" 를 확인할 대상이 없다.
    보고 자체가 죽어도 통과해 버리므로, 모델을 요구하는 작업 하나를 일부러
    로컬에 둔다.
    """
    return {**_all_remote(), "face_extract": 1}


def _config_with_every_binding(tmp_path: Path) -> dict:
    """모든 바인딩에 경로가 채워진 설정 (= 전체 워크플로우를 설치한 상태)."""
    config: dict = {}
    for binding in sorted(
        {
            str(entry["id"])
            for entries in WORKFLOWS["release_dependencies"].values()
            for entry in entries
        }
    ):
        target = config
        parts = binding.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = str(tmp_path / f"{parts[-1]}.json")
    return config


def test_no_gap_when_every_needed_model_is_present(tmp_path):
    """있는 모델을 없다고 하면 경고가 잡음이 된다."""
    config = _config_with_every_binding(tmp_path)
    comfy_root = tmp_path / "comfy"
    for model in MODELS:
        target = comfy_root / str(model["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    gaps = local_model_gaps(
        models=MODELS,
        workflows=WORKFLOWS,
        allocations=_all_remote(),
        config=config,
        comfy_root=comfy_root,
    )
    assert gaps == ()


def test_gap_reported_when_a_local_task_model_is_missing(tmp_path):
    config = _config_with_every_binding(tmp_path)
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    gaps = local_model_gaps(
        models=MODELS,
        workflows=WORKFLOWS,
        allocations=_one_task_local(),
        config=config,
        comfy_root=comfy_root,
    )
    ids = {gap["id"] for gap in gaps}
    # 로컬로 돌린 작업이 쓰는 모델이 빠졌다고 보고돼야 한다.
    assert "face-yolov8m" in ids
    assert "anime-sharp-v4-upscaler" in ids
    # 원격 배분 작업의 모델은 로컬에 없어도 정상이다 — 보고하면 안 된다.
    assert "qwen-image-edit-rapid-v19" not in ids


def test_remote_allocated_models_are_not_reported(tmp_path):
    """원격에서 도는 작업의 모델이 로컬에 없는 것은 cloud_direct 의 정상 상태다."""
    config = _config_with_every_binding(tmp_path)
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    remote_gaps = {
        gap["id"]
        for gap in local_model_gaps(
            models=MODELS,
            workflows=WORKFLOWS,
            allocations=_all_remote(),
            config=config,
            comfy_root=comfy_root,
        )
    }
    all_local_gaps = {
        gap["id"]
        for gap in local_model_gaps(
            models=MODELS,
            workflows=WORKFLOWS,
            allocations={key: 1 for key in COMFY_TASK_KEYS},
            config=config,
            comfy_root=comfy_root,
        )
    }
    assert remote_gaps < all_local_gaps


def test_flipping_a_task_to_local_surfaces_its_models(tmp_path):
    """배분을 되돌리면 그 작업의 모델이 결손으로 잡혀야 한다 — F4 의 요지."""
    config = _config_with_every_binding(tmp_path)
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    base = _all_remote()
    before = {
        gap["id"]
        for gap in local_model_gaps(
            models=MODELS,
            workflows=WORKFLOWS,
            allocations=base,
            config=config,
            comfy_root=comfy_root,
        )
    }
    after = {
        gap["id"]
        for gap in local_model_gaps(
            models=MODELS,
            workflows=WORKFLOWS,
            allocations={**base, "qwen_edit": 1},
            config=config,
            comfy_root=comfy_root,
        )
    }
    assert "qwen-image-edit-rapid-v19" in after - before


def test_uninstalled_workflows_do_not_produce_false_alarms(tmp_path):
    """설치하지 않은 워크플로우의 모델을 '없다'고 하면 오경보다."""
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    gaps = local_model_gaps(
        models=MODELS,
        workflows=WORKFLOWS,
        allocations={key: 1 for key in COMFY_TASK_KEYS},
        config={},  # 바인딩이 하나도 채워지지 않은 상태
        comfy_root=comfy_root,
    )
    assert gaps == ()
    assert configured_binding_ids(WORKFLOWS, {}) == frozenset()


def test_tasks_needing_model_names_the_culprit():
    """안내 문구가 '어느 작업 때문인지'를 말해야 조치할 수 있다."""
    tasks = tasks_needing_model(
        WORKFLOWS, {**_all_remote(), "qwen_edit": 1}, "qwen-image-edit-rapid-v19"
    )
    assert tasks == ("qwen_edit",)


def test_cloud_only_needs_both_conditions():
    """GPU 부재만으로도, 전부 원격만으로도 클라우드 전용이라 할 수 없다."""
    remote = _all_remote()
    assert cloud_only_assessment(nvidia_available=False, allocations=remote)["cloud_only"]
    assert not cloud_only_assessment(
        nvidia_available=True, allocations=remote
    )["cloud_only"]
    mixed = {**remote, "illustration": 1}
    assert not cloud_only_assessment(
        nvidia_available=False, allocations=mixed
    )["cloud_only"]


def test_cloud_only_reports_which_tasks_block_it():
    mixed = {**_all_remote(), "illustration": 1}
    result = cloud_only_assessment(nvidia_available=False, allocations=mixed)
    assert result["locally_assigned_remote_capable_tasks"] == ["illustration"]
    assert result["remote_capable_assigned_remote"] == result[
        "remote_capable_total"
    ] - 1


def test_cloud_only_detection_does_not_mutate_settings():
    """감지가 설정을 바꾸면 사용자가 모르는 사이에 과금·다운로드가 달라진다."""
    allocations = _all_remote()
    snapshot = json.dumps(allocations, sort_keys=True)
    result = cloud_only_assessment(nvidia_available=False, allocations=allocations)
    assert json.dumps(allocations, sort_keys=True) == snapshot
    # 판정 결과에 '바꾼다'가 아니라 '권고한다'만 담긴다.
    assert "modal_model_source" not in result
    assert set(result) == {
        "cloud_only",
        "nvidia_available",
        "remote_capable_total",
        "remote_capable_assigned_remote",
        "locally_assigned_remote_capable_tasks",
    }


def test_installer_preflight_exposes_the_advice():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "model_acquisition" in html
    assert "recommend_cloud_direct" in html
