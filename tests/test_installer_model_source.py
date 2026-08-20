"""설치기가 모델 취득 경로(local_first / cloud_direct)를 따르는지 검증한다.

배경: cloud_direct 는 워커가 저장소에서 Modal 볼륨으로 모델을 직접 받게 하지만,
설치기는 그 설정을 보지 않아 여전히 매니페스트 전체(117.7 GiB)를 로컬로 받았다.
클라우드 전용 머신에서는 그 바이트를 쓰지도 않는다.

규칙: **원격이 아닌 대상에 배분된 작업이 쓰는 모델만 로컬로 받는다.**
플랫폼이 아니라 배분이 기준이다 — NVIDIA 없는 Windows 도 같은 처지이기 때문이다.

자세한 배경은 MODEL_SYNC_DIRECTION.md §4.7 (C1), FIX_AND_TEST_PLAN.md §2 참고.
"""

import json
from pathlib import Path

import pytest

from comfy_allocation import (
    COMFY_TASK_KEYS,
    COMFY_TASK_WORKFLOW_BINDINGS,
    UNMAPPED_WORKFLOW_BINDINGS,
    local_comfy_task_keys,
    local_required_binding_ids,
)
from comfy_installer.manifest import load_install_manifest
from comfy_installer.model_scope import (
    MODEL_SOURCE_CLOUD_DIRECT,
    MODEL_SOURCE_LOCAL_FIRST,
    local_model_ids,
    manifest_binding_ids,
    scope_models,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "comfy_installer" / "resources" / "install_manifest.json").read_text(
        encoding="utf-8"
    )
)
WORKFLOWS = MANIFEST["workflows"]
MODELS = list(MANIFEST["models"])

# Modal 이 지원하지 않아 반드시 로컬에서 도는 작업들.
#
# tag_analysis 와 utility_debug 는 원격 회수 경로가 생겨 여기서 빠졌다(둘 다
# Modal 에서 실측 완료). face_extract 는 원격 분기를 구현했지만 이 머신에
# 인스턴스 LoRA 데이터가 없어 **실측하지 못해** 아직 로컬 전용으로 둔다.
# outfit 은 워크플로우가 배포되지 않아 애초에 바인딩이 없다.
LOCAL_ONLY_TASKS = ("outfit", "face_extract")

# 로컬 전용 작업이 줄면 로컬에 받아야 할 모델도 줄어든다. 예전에는 6개 6.42 GiB
# 였고(MACOS_LOCAL_COMFYUI.md §5.9 의 수기 선택과 일치했다), tag_analysis·
# utility_debug 가 원격으로 옮겨간 지금은 face_extract 바인딩의 2개만 남는다.
# outfit 은 바인딩이 없어 기여하지 않는다.
EXPECTED_LOCAL_MODEL_IDS = {
    "anime-sharp-v4-upscaler",
    "face-yolov8m",
}


def _allocations(remote_tasks=(), local_tasks=()) -> dict:
    result = {key: 1 for key in COMFY_TASK_KEYS}
    for key in remote_tasks:
        result[key] = "modal"
    for key in local_tasks:
        result[key] = 1
    return result


def _all_remote_supported() -> dict:
    """Modal 이 지원하는 8종을 전부 원격으로 — 이 맥의 실제 구성이다."""
    from comfy_allocation import MODAL_SUPPORTED_COMFY_TASK_KEYS

    return _allocations(remote_tasks=MODAL_SUPPORTED_COMFY_TASK_KEYS)


def test_local_first_downloads_every_selected_model():
    """기존 사용자의 동작이 1바이트도 달라지면 안 된다 — 항등이어야 한다."""
    scope = scope_models(
        MODELS,
        workflows=WORKFLOWS,
        allocations=_all_remote_supported(),
        model_source=MODEL_SOURCE_LOCAL_FIRST,
    )
    assert [m["id"] for m in scope.keep] == [m["id"] for m in MODELS]
    assert scope.skipped == ()
    assert not scope.filtered


def test_every_manifest_binding_is_mapped_or_declared_unmapped():
    """표가 매니페스트와 어긋나면 모델이 조용히 누락된다."""
    declared: set[str] = set()
    for bindings in COMFY_TASK_WORKFLOW_BINDINGS.values():
        declared.update(bindings)
    manifest_ids = manifest_binding_ids(WORKFLOWS)

    uncovered = manifest_ids - declared - UNMAPPED_WORKFLOW_BINDINGS
    assert not uncovered, (
        "매니페스트 바인딩이 어느 작업에도 매이지 않았습니다. "
        "COMFY_TASK_WORKFLOW_BINDINGS 또는 UNMAPPED_WORKFLOW_BINDINGS 에 "
        f"추가하세요: {sorted(uncovered)}"
    )
    stale = declared - manifest_ids
    assert not stale, f"매니페스트에 없는 바인딩을 선언했습니다: {sorted(stale)}"


def test_every_task_key_has_a_binding_entry():
    """작업이 추가됐는데 표에 없으면 그 작업의 모델은 로컬 집합에서 빠진다."""
    missing = set(COMFY_TASK_KEYS) - set(COMFY_TASK_WORKFLOW_BINDINGS)
    assert not missing, f"바인딩 선언이 없는 작업: {sorted(missing)}"


def test_cloud_direct_keeps_only_locally_allocated_task_models():
    """이 맥의 구성(Modal 지원 8종 원격, 나머지 로컬) → 경량 6개만 남아야 한다."""
    scope = scope_models(
        MODELS,
        workflows=WORKFLOWS,
        allocations=_all_remote_supported(),
        model_source=MODEL_SOURCE_CLOUD_DIRECT,
    )
    assert {m["id"] for m in scope.keep} == EXPECTED_LOCAL_MODEL_IDS
    assert scope.filtered
    # 0.078 GiB — face_extract 바인딩 2개뿐이다. 예전 6.42 GiB 에서 줄어든 것은
    # tag_analysis·utility_debug 가 원격으로 옮겨갔기 때문이다. 숫자가 크게
    # 움직이면 배분 규칙이 바뀐 것이다.
    assert 0.0 < scope.keep_bytes / 1024**3 < 0.5
    assert scope.skipped_bytes > 100 * 1024**3


def test_cloud_direct_never_skips_locally_executed_task_models():
    """Modal 미지원 4종은 어떤 설정에서도 로컬에서 돈다 — 그 모델은 남아야 한다.

    utility_debug 가 만드는 cache.pt 가 없으면 등록 캐릭터 삽화가 전부 막힌다(G1).
    "클라우드 전용 = 아무것도 안 받는다" 가 아니라는 것이 이 테스트의 요지다.
    """
    needed = local_model_ids(WORKFLOWS, _all_remote_supported())
    assert needed == EXPECTED_LOCAL_MODEL_IDS
    assert needed, "로컬 실행 작업이 남아 있는데 로컬 모델이 0개일 수 없다."

    # 로컬 실행 작업이 하나도 없는 극단 구성에서만 0개가 된다.
    every_task_remote = {key: "modal" for key in COMFY_TASK_KEYS}
    assert local_comfy_task_keys(every_task_remote) == ()
    assert local_model_ids(WORKFLOWS, every_task_remote) == frozenset()


def test_local_allocation_of_one_task_pulls_its_models_back():
    """작업 하나를 로컬로 돌리면 그 작업의 모델이 다시 로컬 대상이 된다."""
    base = _all_remote_supported()
    with_local_qwen = {**base, "qwen_edit": 1}
    added = local_model_ids(WORKFLOWS, with_local_qwen) - local_model_ids(
        WORKFLOWS, base
    )
    assert added, "qwen_edit 를 로컬로 돌렸는데 추가된 모델이 없다."
    assert "anima-lllite-inpainting-v2" in added


def test_missing_allocation_key_is_treated_as_local():
    """모르는 것을 원격으로 가정하면 필요한 모델을 안 받게 된다."""
    assert set(local_comfy_task_keys({})) == set(COMFY_TASK_KEYS)
    assert local_comfy_task_keys(None) == tuple(COMFY_TASK_KEYS)


@pytest.mark.parametrize("value", ["", None, "nonsense", 123, "MODAL"])
def test_non_remote_values_are_local(value):
    """대소문자 'MODAL' 은 원격, 그 밖의 값은 전부 로컬로 본다."""
    allocations = {key: 1 for key in COMFY_TASK_KEYS}
    allocations["illustration"] = value
    is_local = "illustration" in local_comfy_task_keys(allocations)
    assert is_local is (str(value).strip().lower() not in {"modal", "vast"})


def test_filter_is_platform_independent(monkeypatch):
    """규칙은 배분 기준이다. platform 분기가 들어가면 그건 버그다."""
    import platform

    results = []
    for system in ("Windows", "Darwin", "Linux"):
        monkeypatch.setattr(platform, "system", lambda s=system: s)
        results.append(
            frozenset(
                m["id"]
                for m in scope_models(
                    MODELS,
                    workflows=WORKFLOWS,
                    allocations=_all_remote_supported(),
                    model_source=MODEL_SOURCE_CLOUD_DIRECT,
                ).keep
            )
        )
    assert len(set(results)) == 1
    assert results[0] == EXPECTED_LOCAL_MODEL_IDS


def test_scope_summary_reports_what_was_skipped():
    """조용한 스킵은 버그와 구별되지 않는다 — 개수와 용량이 로그에 남아야 한다."""
    scope = scope_models(
        MODELS,
        workflows=WORKFLOWS,
        allocations=_all_remote_supported(),
        model_source=MODEL_SOURCE_CLOUD_DIRECT,
    )
    summary = scope.summary()
    assert str(len(scope.keep)) in summary
    assert str(len(scope.skipped)) in summary
    assert "GiB" in summary


def test_unknown_model_source_behaves_like_local_first():
    """알 수 없는 값이 조용히 cloud_direct 로 새면 설치 내용이 달라진다."""
    scope = scope_models(
        MODELS,
        workflows=WORKFLOWS,
        allocations=_all_remote_supported(),
        model_source="nonsense",
    )
    assert len(scope.keep) == len(MODELS)
    assert not scope.filtered


def test_binding_lookup_survives_a_broken_manifest():
    """매니페스트가 깨졌을 때 예외 대신 빈 집합이면 전부 스킵된다 — 그러면 안 된다."""
    assert manifest_binding_ids({}) == frozenset()
    assert local_model_ids({}, _all_remote_supported()) == frozenset()
    # 빈 집합이어도 cloud_direct 는 '전부 스킵'이 되므로, 서비스 계층이
    # 매니페스트를 신뢰할 수 있을 때만 이 경로를 타야 한다는 것을 문서화한다.
    scope = scope_models(
        MODELS,
        workflows={},
        allocations=_all_remote_supported(),
        model_source=MODEL_SOURCE_CLOUD_DIRECT,
    )
    assert scope.keep == ()


def test_local_required_bindings_union_shared_workflows():
    """LoRA 학습 3종은 워크플로우를 공유한다 — 하나만 로컬이어도 남아야 한다."""
    base = {key: "modal" for key in COMFY_TASK_KEYS}
    only_bot_local = {**base, "bot_lora_training": 1}
    bindings = local_required_binding_ids(only_bot_local)
    assert "lora_training_workflow_source_paths.anima" in bindings
    assert "style_lora_training_workflow_source_paths.anima" not in bindings


class _FakeManifest:
    def __init__(self):
        self.models = MODELS
        self.workflows = WORKFLOWS


def _service_with_config(tmp_path: Path, config: dict):
    """ComfyInstallerService 를 만들지 않고 스코프 헬퍼만 떼어 검증한다.

    설치기 서비스 생성은 git/마이그레이션까지 건드리므로 여기서는
    _scope_selected_models 가 config.json 을 읽는 계약만 확인한다.
    """
    from comfy_installer.service import ComfyInstallerService

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    service = object.__new__(ComfyInstallerService)
    service.config_path = config_path
    service.manifest = _FakeManifest()
    return service


def test_service_reads_model_source_from_config(tmp_path):
    service = _service_with_config(
        tmp_path,
        {
            "modal_model_source": "cloud_direct",
            "comfy_task_allocations": _all_remote_supported(),
        },
    )
    scope = service._scope_selected_models(MODELS)
    assert scope.model_source == MODEL_SOURCE_CLOUD_DIRECT
    assert {m["id"] for m in scope.keep} == EXPECTED_LOCAL_MODEL_IDS


def test_service_defaults_to_local_first_without_config(tmp_path):
    """설정을 못 읽으면 전부 받는다 — 모르는 채로 건너뛰면 나중에 조용히 실패한다."""
    from comfy_installer.service import ComfyInstallerService

    service = object.__new__(ComfyInstallerService)
    service.config_path = tmp_path / "does-not-exist.json"
    service.manifest = _FakeManifest()
    scope = service._scope_selected_models(MODELS)
    assert scope.model_source == MODEL_SOURCE_LOCAL_FIRST
    assert len(scope.keep) == len(MODELS)


def test_service_defaults_to_local_first_on_broken_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{ not json", encoding="utf-8")
    from comfy_installer.service import ComfyInstallerService

    service = object.__new__(ComfyInstallerService)
    service.config_path = config_path
    service.manifest = _FakeManifest()
    scope = service._scope_selected_models(MODELS)
    assert scope.model_source == MODEL_SOURCE_LOCAL_FIRST
    assert len(scope.keep) == len(MODELS)


def test_preflight_disk_requirement_follows_the_filtered_set(tmp_path, monkeypatch):
    """cloud_direct 에서 매니페스트 전체를 요구하면 멀쩡한 머신이 막힌다.

    30 GiB(런타임+여유) + '실제로 받을 바이트' 여야 한다.
    """
    from comfy_installer import service as service_module

    captured: dict = {}

    def fake_probe(comfy_root, manifest, *, required_bytes, require_disk, install_mode):
        captured["required_bytes"] = required_bytes
        return {
            "gpu_profile": "cpu",
            "disk": {"free": 400 * 1024**3, "required": required_bytes},
            "nvidia": {"available": False, "gpus": []},
        }

    def fake_requirements(*, library_root, release_version, selected_item_ids):
        return {
            "release_version": release_version,
            "selected_item_ids": list(selected_item_ids),
            "model_ids": [str(m["id"]) for m in MODELS],
            "model_bytes": sum(int(m["size"]) for m in MODELS),
        }

    monkeypatch.setattr(service_module, "probe_system", fake_probe)
    monkeypatch.setattr(service_module, "selection_requirements", fake_requirements)
    # 팩이 자기완결형이 된 뒤로 preflight_selection 은 팩 동봉 매니페스트를 읽는다.
    # 여기서 보는 것은 디스크 요구량 산정이지 팩 해제가 아니므로 저장소
    # 매니페스트를 그대로 돌려준다.
    monkeypatch.setattr(
        service_module,
        "release_install_manifest",
        lambda *, library_root, release_version: load_install_manifest(),
    )

    service = _service_with_config(
        tmp_path,
        {
            "modal_model_source": "cloud_direct",
            "comfy_task_allocations": _all_remote_supported(),
        },
    )
    service.comfy_root = tmp_path / "comfy"
    service.workflow_library_root = tmp_path / "library"
    service._state = {"manifest": {}}
    service._lock = __import__("threading").RLock()

    result = service.preflight_selection(
        release_version="v2",
        selected_item_ids=["anything"],
    )

    runtime_and_buffer = 30 * 1024**3
    local_bytes = captured["required_bytes"] - runtime_and_buffer
    assert 0.0 < local_bytes / 1024**3 < 0.5, (
        "cloud_direct 인데 디스크 요구량이 로컬 다운로드분을 넘습니다: "
        f"{local_bytes / 1024**3:.2f} GiB"
    )
    selection = result["selection"]
    assert selection["model_source"] == MODEL_SOURCE_CLOUD_DIRECT
    assert set(selection["local_model_ids"]) == EXPECTED_LOCAL_MODEL_IDS
    assert selection["remote_model_count"] == len(MODELS) - len(
        EXPECTED_LOCAL_MODEL_IDS
    )


def test_preflight_disk_requirement_unchanged_for_local_first(tmp_path, monkeypatch):
    from comfy_installer import service as service_module

    captured: dict = {}

    def fake_probe(comfy_root, manifest, *, required_bytes, require_disk, install_mode):
        captured["required_bytes"] = required_bytes
        return {
            "gpu_profile": "cpu",
            "disk": {"free": 400 * 1024**3, "required": required_bytes},
            "nvidia": {"available": False, "gpus": []},
        }

    def fake_requirements(*, library_root, release_version, selected_item_ids):
        return {
            "release_version": release_version,
            "selected_item_ids": list(selected_item_ids),
            "model_ids": [str(m["id"]) for m in MODELS],
            "model_bytes": sum(int(m["size"]) for m in MODELS),
        }

    monkeypatch.setattr(service_module, "probe_system", fake_probe)
    monkeypatch.setattr(service_module, "selection_requirements", fake_requirements)
    # 팩이 자기완결형이 된 뒤로 preflight_selection 은 팩 동봉 매니페스트를 읽는다.
    # 여기서 보는 것은 디스크 요구량 산정이지 팩 해제가 아니므로 저장소
    # 매니페스트를 그대로 돌려준다.
    monkeypatch.setattr(
        service_module,
        "release_install_manifest",
        lambda *, library_root, release_version: load_install_manifest(),
    )

    service = _service_with_config(
        tmp_path,
        {"comfy_task_allocations": _all_remote_supported()},
    )
    service.comfy_root = tmp_path / "comfy"
    service.workflow_library_root = tmp_path / "library"
    service._state = {"manifest": {}}
    service._lock = __import__("threading").RLock()

    service.preflight_selection(release_version="v2", selected_item_ids=["anything"])
    expected = 30 * 1024**3 + sum(int(m["size"]) for m in MODELS)
    assert captured["required_bytes"] == expected


def test_installer_ui_shows_the_local_remote_split():
    """설치기 화면이 전체 선택 용량만 보여주면 사용자는 그만큼 받는 줄 안다."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "selection.model_source === 'cloud_direct'" in html
    assert "local_model_bytes" in html
    assert "remote_model_bytes" in html
    # 규칙을 JS 에서 다시 계산하면 서버와 어긋난다 — 서버 값을 그대로 쓴다.
    assert "COMFY_TASK_WORKFLOW_BINDINGS" not in html


def test_civitai_key_is_not_required_for_remote_only_models(tmp_path):
    """워커가 Secret 으로 받는 인증 모델까지 로컬 키를 요구하면 안 된다."""
    service = _service_with_config(
        tmp_path,
        {
            "modal_model_source": "cloud_direct",
            "comfy_task_allocations": {key: "modal" for key in COMFY_TASK_KEYS},
        },
    )
    scope = service._scope_selected_models(MODELS)
    assert scope.keep == ()
    assert any(m.get("auth") == "civitai" for m in scope.skipped)


def test_cloud_only_install_mode_scopes_before_config_is_written(tmp_path):
    """설치 순서가 만드는 함정을 막는다.

    설치는 모델을 먼저 받고(`models` 단계) 설정을 나중에 적용한다(`config` 단계).
    클라우드 전용 설치가 config.json 에만 반영되면, 정작 다운로드를 정하는
    시점에는 아직 옛 설정(local_first · 전부 로컬)이 보여서 매니페스트 전체를
    로컬로 받아 놓고 그 다음에 "클라우드 전용" 이라고 적게 된다. 이 모드의
    이득이 통째로 사라지는 조용한 실패다.

    그래서 여기서는 config.json 이 **정반대로** 적혀 있는 상태를 만들어 두고,
    선언된 설치 모드만으로 스코프가 좁혀지는지 본다.
    """

    from comfy_allocation import MODAL_SUPPORTED_COMFY_TASK_KEYS
    from comfy_installer.install_modes import INSTALL_MODE_CLOUD_ONLY

    service = _service_with_config(
        tmp_path,
        {
            "modal_model_source": "local_first",
            "comfy_task_allocations": _allocations(
                local_tasks=sorted(MODAL_SUPPORTED_COMFY_TASK_KEYS)
            ),
        },
    )
    scope = service._scope_selected_models(
        MODELS, install_mode=INSTALL_MODE_CLOUD_ONLY
    )
    assert scope.model_source == MODEL_SOURCE_CLOUD_DIRECT
    assert {m["id"] for m in scope.keep} == EXPECTED_LOCAL_MODEL_IDS


def test_standard_install_mode_still_obeys_config(tmp_path):
    """클라우드 전용이 아닌 설치는 예전과 똑같이 config.json 을 따른다."""

    from comfy_installer.install_modes import INSTALL_MODE_STANDARD

    service = _service_with_config(
        tmp_path,
        {
            "modal_model_source": "local_first",
            "comfy_task_allocations": _all_remote_supported(),
        },
    )
    scope = service._scope_selected_models(
        MODELS, install_mode=INSTALL_MODE_STANDARD
    )
    assert scope.model_source == MODEL_SOURCE_LOCAL_FIRST
    assert {m["id"] for m in scope.keep} == {m["id"] for m in MODELS}
