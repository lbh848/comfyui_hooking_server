"""클라우드 전용 설치 모드의 기본값.

이 모드가 뜻하는 바는 하나다: **원격에 보낼 수 있는 작업은 전부 원격에 보내고,
모델도 저장소에서 원격 볼륨으로 직접 받는다.** 사용자가 설치할 때 한 번 고르면
설정 화면을 돌아다니며 12개 작업을 손으로 바꾸지 않아도 된다.
"""

import json
from pathlib import Path

import pytest

from comfy_allocation import (
    CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS,
    CLOUD_ONLY_DEFAULT_MODEL_SOURCE,
    COMFY_TASK_KEYS,
    DEFAULT_COMFY_TASK_ALLOCATIONS,
    MODAL_SUPPORTED_COMFY_TASK_KEYS,
    cloud_only_assessment,
    default_comfy_task_allocations,
    is_remote_allocation,
    normalize_comfy_task_allocations,
)
from comfy_installer.configurator import apply_installed_config
from comfy_installer.install_modes import (
    INSTALL_MODE_CLOUD_ONLY,
    INSTALL_MODE_NVIDIA_COMPATIBILITY,
    INSTALL_MODE_STANDARD,
    effective_gpu_profile,
    normalize_install_mode,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 기본 배분 ────────────────────────────────────────────────────────────────


def test_cloud_only_defaults_send_every_remote_capable_task_remote():
    for key in MODAL_SUPPORTED_COMFY_TASK_KEYS:
        assert is_remote_allocation(
            CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS[key]
        ), key


def test_cloud_only_defaults_leave_unsupported_tasks_local():
    """보낼 수 없는 것까지 보내지 않는다.

    원격 미지원 작업을 원격으로 적어 두면 정규화가 거부하거나(검증) 조용히
    로컬로 되돌아간다. 어느 쪽이든 사용자가 고른 의미와 달라진다.
    """

    unsupported = set(COMFY_TASK_KEYS) - set(MODAL_SUPPORTED_COMFY_TASK_KEYS)
    assert unsupported, "원격 미지원 작업이 하나도 없으면 이 테스트는 의미가 없다"
    for key in unsupported:
        assert not is_remote_allocation(
            CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS[key]
        ), key


def test_cloud_only_defaults_track_the_supported_set_automatically():
    """표를 손으로 관리하지 않는다.

    MODAL_SUPPORTED_COMFY_TASK_KEYS 에 키가 하나 늘면(face_extract 실측 등)
    기본값도 따라와야 한다. 두 곳을 손으로 맞추는 구조면 반드시 어긋난다.
    """

    expected = {
        key: ("modal" if key in MODAL_SUPPORTED_COMFY_TASK_KEYS else 1)
        for key in COMFY_TASK_KEYS
    }
    assert CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS == expected


def test_cloud_only_defaults_pass_allocation_validation():
    normalized = normalize_comfy_task_allocations(
        dict(CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS)
    )
    assert normalized == CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS


def test_cloud_only_defaults_actually_satisfy_the_cloud_only_assessment():
    """기본값과 판정이 서로 맞는지 확인한다.

    cloud_only_assessment 는 배분이 전부 원격일 때 참이 된다. 기본값이 그
    조건을 못 채우면 "클라우드 전용으로 설치했는데 클라우드 전용이 아니라고
    나오는" 모순이 된다.
    """

    verdict = cloud_only_assessment(
        nvidia_available=False,
        allocations=dict(CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS),
    )
    assert verdict["cloud_only"] is True
    assert verdict["locally_assigned_remote_capable_tasks"] == []


def test_default_allocations_helper_is_opt_in():
    assert default_comfy_task_allocations() == DEFAULT_COMFY_TASK_ALLOCATIONS
    assert (
        default_comfy_task_allocations(cloud_only=True)
        == CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS
    )


def test_default_allocations_helper_returns_a_copy():
    """호출자가 받은 dict 를 고쳐도 모듈 상수가 오염되지 않아야 한다."""

    mutated = default_comfy_task_allocations(cloud_only=True)
    mutated["illustration"] = 1
    assert is_remote_allocation(
        CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS["illustration"]
    )


# ── 설치 모드 ────────────────────────────────────────────────────────────────


def test_cloud_only_is_a_supported_install_mode():
    assert normalize_install_mode(INSTALL_MODE_CLOUD_ONLY) == INSTALL_MODE_CLOUD_ONLY


def test_cloud_only_profile_drops_local_acceleration_packages():
    profile = {
        "kind": "nvidia",
        "sageattention_required": True,
        "sageattention": {"package": "sageattention"},
        "triton_package": "triton-windows",
    }
    effective = effective_gpu_profile(profile, INSTALL_MODE_CLOUD_ONLY)
    assert effective["sageattention_required"] is False
    assert "sageattention" not in effective
    assert "triton_package" not in effective


def test_cloud_only_profile_sets_no_compute_capability_floor():
    """호환 설치와 다른 점.

    로컬 GPU 가 아예 없는 머신이 이 모드의 주 대상이다. 최소 compute
    capability 를 두면 정상 구성이 프리플라이트에서 막힌다.
    """

    profile = {"kind": "nvidia", "sageattention_required": True}
    cloud_only = effective_gpu_profile(profile, INSTALL_MODE_CLOUD_ONLY)
    compatibility = effective_gpu_profile(profile, INSTALL_MODE_NVIDIA_COMPATIBILITY)
    assert "minimum_compute_capability" not in cloud_only
    assert "minimum_compute_capability" in compatibility


def test_cloud_only_profile_applies_without_a_local_gpu():
    """kind 가 nvidia 가 아니어도 적용돼야 한다 (Apple Silicon·CPU)."""

    effective = effective_gpu_profile(
        {"kind": "apple", "sageattention_required": True},
        INSTALL_MODE_CLOUD_ONLY,
    )
    assert effective["sageattention_required"] is False
    assert effective["install_mode"] == INSTALL_MODE_CLOUD_ONLY


# ── config.json 적용 ─────────────────────────────────────────────────────────


def _apply(tmp_path: Path, *, install_mode: str, original: dict | None = None) -> dict:
    config_path = tmp_path / "config.json"
    comfy = tmp_path / "comfy"
    workflows = comfy / "user" / "default" / "workflows"
    workflows.mkdir(parents=True)
    workflow = workflows / "first.json"
    _write_json(workflow, {"1": {"class_type": "KSampler"}})
    _write_json(config_path, original if original is not None else {"unrelated": "keep"})

    apply_installed_config(
        config_path=config_path,
        requirements_dir=tmp_path / "backup",
        comfy_root=comfy,
        workflow_bindings={"asset_workflow_source_path": str(workflow)},
        required_bindings=["asset_workflow_source_path"],
        install_mode=install_mode,
    )
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_cloud_only_install_writes_remote_allocations(tmp_path):
    config = _apply(tmp_path, install_mode=INSTALL_MODE_CLOUD_ONLY)
    assert config["comfy_task_allocations"] == CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS
    assert config["modal_model_source"] == CLOUD_ONLY_DEFAULT_MODEL_SOURCE
    assert config["modal_enabled"] is True


def test_cloud_only_install_overrides_existing_local_allocations(tmp_path):
    """모드를 골랐으면 기존 값이 있어도 적용돼야 한다.

    '값이 없을 때만 채운다' 로 두면 서버를 한 번이라도 띄운 머신에서는
    config.json 에 이미 전부-로컬 기본값이 적혀 있어 아무 일도 일어나지 않는다.
    설치는 구성을 **선언하는** 행위라 여기서는 덮는 것이 맞다.
    """

    original = {
        "comfy_task_allocations": dict(DEFAULT_COMFY_TASK_ALLOCATIONS),
        "modal_model_source": "local_first",
        "modal_enabled": False,
    }
    config = _apply(
        tmp_path, install_mode=INSTALL_MODE_CLOUD_ONLY, original=original
    )
    assert config["comfy_task_allocations"] == CLOUD_ONLY_DEFAULT_COMFY_TASK_ALLOCATIONS
    assert config["modal_model_source"] == "cloud_direct"


@pytest.mark.parametrize(
    "mode", [INSTALL_MODE_STANDARD, INSTALL_MODE_NVIDIA_COMPATIBILITY]
)
def test_other_install_modes_do_not_touch_allocations(tmp_path, mode):
    config = _apply(tmp_path, install_mode=mode)
    assert "comfy_task_allocations" not in config
    assert "modal_model_source" not in config
    assert "modal_enabled" not in config


def test_install_mode_defaults_to_standard_for_existing_callers(tmp_path):
    """기존 호출자(워크플로우 업데이트 경로)는 인자를 넘기지 않는다."""

    config_path = tmp_path / "config.json"
    comfy = tmp_path / "comfy"
    workflows = comfy / "user" / "default" / "workflows"
    workflows.mkdir(parents=True)
    workflow = workflows / "first.json"
    _write_json(workflow, {"1": {"class_type": "KSampler"}})
    _write_json(config_path, {"modal_model_source": "local_first"})

    apply_installed_config(
        config_path=config_path,
        requirements_dir=tmp_path / "backup",
        comfy_root=comfy,
        workflow_bindings={"asset_workflow_source_path": str(workflow)},
        required_bindings=["asset_workflow_source_path"],
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["modal_model_source"] == "local_first"
