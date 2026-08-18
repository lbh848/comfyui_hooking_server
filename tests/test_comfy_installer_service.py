from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import comfy_installer.service as service_module
from comfy_installer.e2e import ComfyE2EError
from comfy_installer.crypto import ExtractedWorkflowPack
from comfy_installer.manifest import load_install_manifest
from comfy_installer.operations import uv_python_path
from comfy_installer.service import (
    ComfyInstallerService,
    _E2E_PHASES,
    _INSTALL_PHASES,
    _UPDATE_PHASES,
    _hooking_server_restart_required,
)
from comfy_installer.workflow_library import (
    DISTRIBUTION_LIBRARY_DIRNAME,
    LEGACY_USER_WORKFLOW_DIRNAME,
    USER_WORKFLOW_DIRNAME,
    WorkflowSelection,
)


def test_install_and_update_do_not_embed_workflow_e2e() -> None:
    install_phases = [phase for phase, _label in _INSTALL_PHASES]
    update_phases = [phase for phase, _label in _UPDATE_PHASES]
    e2e_phases = [phase for phase, _label in _E2E_PHASES]

    assert install_phases.index("models") < install_phases.index("repatch")
    assert install_phases.index("repatch") < install_phases.index("startup")
    assert "e2e_static" not in install_phases
    assert "e2e_runtime" not in install_phases
    assert "e2e_static" not in update_phases
    assert "e2e_runtime" not in update_phases
    assert e2e_phases.index("e2e_static") < e2e_phases.index("e2e_runtime")
    assert e2e_phases.index("e2e_runtime") < e2e_phases.index(
        "e2e_video_runtime"
    )


def test_hooking_server_restart_required_after_pull() -> None:
    assert _hooking_server_restart_required(
        startup_head="a" * 40,
        update_result={"changed": True, "after": "b" * 40},
    )


def test_hooking_server_restart_required_for_stale_process() -> None:
    assert _hooking_server_restart_required(
        startup_head="a" * 40,
        update_result={"changed": False, "after": "b" * 40},
    )


def test_hooking_server_restart_not_required_for_current_process() -> None:
    assert not _hooking_server_restart_required(
        startup_head="a" * 40,
        update_result={"changed": False, "after": "A" * 40},
    )


def test_e2e_catalog_uses_distribution_originals_without_runtime_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    python = uv_python_path(service.comfy_root / ".venv")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    release_root = (
        service.workflow_library_root
        / DISTRIBUTION_LIBRARY_DIRNAME
        / "v1"
    )
    release_root.mkdir(parents=True)
    workflow = release_root / "debug.json"
    workflow.write_text('{"nodes": []}\n', encoding="utf-8")
    state = {
        "schema_version": 2,
        "release_version": "v1",
        "items": [
            {
                "id": "debug_workflow_source_path",
                "name": "디버그",
                "filename": workflow.name,
                "sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
                "bindings": ["debug_workflow_source_path"],
                "model_ids": [],
            }
        ],
    }
    state_path = release_root / ".soya-pack.json"
    state_path.write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )

    try:
        catalog = service.e2e_workflow_catalog()
        assert catalog["release_version"] == "v1"
        assert catalog["source_kind"] == (
            "read_only_distribution_original"
        )
        assert [item["id"] for item in catalog["items"]] == [
            "debug_workflow_source_path"
        ]
        assert not (
            service.comfy_root / ".installer-state" / "runtime-receipt.json"
        ).exists()

        monkeypatch.setattr(
            service,
            "_start_operation",
            lambda **kwargs: kwargs,
        )
        started = service.start_e2e(
            release_version="v1",
            selected_item_ids=["debug_workflow_source_path"],
        )
        assert started["operation"] == "e2e"
        assert started["kwargs"] == {
            "release_version": "v1",
            "selected_item_ids": ["debug_workflow_source_path"],
        }
    finally:
        for path in release_root.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode | stat.S_IWRITE)


def test_video_e2e_uses_only_comfy_three_profile_arguments(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "comfy_launch_profiles": {
                    "1": {"vram_mode": "novram"},
                    "2": {"cuda_device": 2},
                    "3": {
                        "cuda_device": 1,
                        "vram_mode": "lowvram",
                        "disable_dynamic_vram": True,
                        "fast": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )

    assert service._video_e2e_extra_args() == (
        "--cuda-device",
        "1",
        "--lowvram",
        "--disable-dynamic-vram",
        "--fast",
    )


def test_runtime_validations_split_h3_from_legacy(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    legacy_validation = SimpleNamespace(
        binding_keys=("illustration_workflow_source_paths.v1",),
        filename="legacy.json",
    )
    video_validation = SimpleNamespace(
        binding_keys=("video_workflow_source_paths.i2v",),
        filename="h3.json",
    )

    legacy, video = service._partition_runtime_validations(
        [video_validation, legacy_validation]
    )

    assert legacy == [legacy_validation]
    assert video == [video_validation]


def test_video_runtime_order_is_standard_then_fast() -> None:
    validations = [
        SimpleNamespace(
            binding_keys=("video_workflow_source_paths.first_last_fast",),
            filename="first-last-fast.json",
        ),
        SimpleNamespace(
            binding_keys=("video_workflow_source_paths.i2v_fast",),
            filename="i2v-fast.json",
        ),
        SimpleNamespace(
            binding_keys=("video_workflow_source_paths.first_last",),
            filename="first-last.json",
        ),
        SimpleNamespace(
            binding_keys=("video_workflow_source_paths.i2v",),
            filename="i2v.json",
        ),
    ]

    ordered = sorted(validations, key=ComfyInstallerService._runtime_order)

    assert [item.filename for item in ordered] == [
        "i2v.json",
        "first-last.json",
        "i2v-fast.json",
        "first-last-fast.json",
    ]


@pytest.mark.parametrize(
    ("binding_key", "sample_steps", "sample_width", "sample_height"),
    [
        ("video_workflow_source_paths.i2v", 8, 960, 544),
        ("video_workflow_source_paths.i2v_fast", 4, 1344, 768),
    ],
)
def test_runtime_e2e_applies_h3_manifest_defaults(
    tmp_path: Path,
    monkeypatch,
    binding_key: str,
    sample_steps: int,
    sample_width: int,
    sample_height: int,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    validation = SimpleNamespace(
        binding_keys=(binding_key,),
        filename="h3.json",
        prompt={},
        workflow={"nodes": []},
    )
    captured: dict = {}
    promotion_calls: list[dict] = []

    def fake_make_e2e_prompt(value, **kwargs):
        captured["validation"] = value
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(service_module, "make_e2e_prompt", fake_make_e2e_prompt)
    monkeypatch.setattr(
        service_module,
        "execute_prompt",
        lambda **_kwargs: {
            "filename": "h3.json",
            "prompt_id": "prompt-h3",
            "status": "success",
            "outputs": [],
            "output_data": {},
        },
    )
    monkeypatch.setattr(
        service_module,
        "promote_generated_fixture",
        lambda **kwargs: promotion_calls.append(kwargs),
    )

    results = service._run_runtime_e2e(
        process=SimpleNamespace(base_url="http://127.0.0.1:12345"),
        validations=[validation],
        fixtures={
            "training": "fixture/training/sample.png",
            "face_source": "fixture/fallback/face.webp",
        },
    )

    assert len(results) == 1
    assert captured == {
        "validation": validation,
        "sample_steps": sample_steps,
        "sample_width": sample_width,
        "sample_height": sample_height,
    }
    assert promotion_calls == []


def test_runtime_e2e_keeps_fixture_promotion_for_legacy_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    validation = SimpleNamespace(
        binding_keys=("illustration_workflow_source_paths.v1",),
        filename="legacy.json",
        prompt={},
        workflow={"nodes": []},
    )
    promotion_calls: list[dict] = []

    monkeypatch.setattr(service_module, "make_e2e_prompt", lambda _value: {})
    monkeypatch.setattr(
        service_module,
        "execute_prompt",
        lambda **_kwargs: {
            "filename": "legacy.json",
            "prompt_id": "prompt-legacy",
            "status": "success",
            "outputs": [],
            "output_data": {},
        },
    )
    monkeypatch.setattr(
        service_module,
        "promote_generated_fixture",
        lambda **kwargs: promotion_calls.append(kwargs),
    )

    results = service._run_runtime_e2e(
        process=SimpleNamespace(base_url="http://127.0.0.1:12345"),
        validations=[validation],
        fixtures={
            "training": "fixture/training/sample.png",
            "face_source": "fixture/fallback/face.webp",
        },
    )

    assert len(results) == 1
    assert len(promotion_calls) == 1


@pytest.mark.parametrize("release_version", ["v1", "v2"])
def test_pack_validation_uses_release_specific_bindings(
    tmp_path: Path, release_version: str
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    bindings: dict[str, str] = {}
    release = service.manifest.workflows["release_dependencies"][release_version]
    for index, entry in enumerate(release):
        path = str(tmp_path / f"workflow-{index}.json")
        for binding in entry["bindings"]:
            bindings[binding] = path
    extracted = ExtractedWorkflowPack(
        target_dir=tmp_path,
        workflow_bindings=bindings,
        workflow_hashes={},
        pack_sha256="0" * 64,
        release_version=release_version,
    )

    service._validate_extracted_pack(extracted)


def test_service_status_never_contains_credentials(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    payload = json.dumps(service.status(), ensure_ascii=False)
    assert "civitai_key" not in payload
    assert "workflow_key" not in payload
    assert service.comfy_root == tmp_path / "comfy"


def test_service_startup_migrates_existing_workflow_paths_to_ascii(
    tmp_path: Path,
) -> None:
    legacy_root = (
        tmp_path
        / "comfy"
        / "user"
        / "default"
        / "workflows"
        / LEGACY_USER_WORKFLOW_DIRNAME
    )
    legacy_root.mkdir(parents=True)
    legacy_workflow = legacy_root / "main.json"
    legacy_workflow.write_text('{"legacy":true}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"comfy_workflow_source_path": str(legacy_workflow)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    requirements = tmp_path / "요구사항"

    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=requirements,
    )

    updated = json.loads(config.read_text(encoding="utf-8"))
    migrated = Path(updated["comfy_workflow_source_path"])
    assert migrated.parent.name == USER_WORKFLOW_DIRNAME
    assert migrated.read_text(encoding="utf-8") == '{"legacy":true}'
    assert not legacy_root.exists()
    migration = service.status()["workflow_path_migration"]
    assert migration["config"]["updated"] is True
    legacy_backup = Path(migration["user"]["legacy_archive"]["backup"])
    assert (legacy_backup / "main.json").read_text(
        encoding="utf-8"
    ) == '{"legacy":true}'
    deployment_backups = (
        tmp_path / "comfy" / ".installer-state" / "backups" / "config"
    )
    assert len(
        list(
            deployment_backups.glob(
                "config_before_workflow_ascii_migration_*.json"
            )
        )
    ) == 1
    assert not requirements.exists()


def test_server_runs_workflow_path_migration_before_loading_config() -> None:
    source = Path("server.py").read_text(encoding="utf-8")

    assert source.index("WORKFLOW_PATH_MIGRATION = migrate_legacy_workflow_layout(") < (
        source.index("app_config = load_config()")
    )


def test_common_settings_persist_absolute_workflow_base_dir() -> None:
    server_source = Path("server.py").read_text(encoding="utf-8")
    frontend_source = Path("frontend/index.html").read_text(encoding="utf-8")

    assert '"workflow_base_dir": ""' in server_source
    assert (
        "currentConfig.workflow_base_dir || pathParts.baseDir"
        in frontend_source
    )
    assert (
        "workflow_base_dir: document.getElementById('setting-workflow-base-dir')"
        in frontend_source
    )


def test_frontend_keeps_empty_lora_paths_empty_and_rebases_every_workflow() -> None:
    frontend_source = Path("frontend/index.html").read_text(encoding="utf-8")
    update_start = frontend_source.index("function updateWorkflowPath()")
    update_end = frontend_source.index("function combineWorkflowPath", update_start)
    update_source = frontend_source[update_start:update_end]

    for updater in (
        "updateQwenEditWorkflowPath();",
        "updateAnimaInpaintingWorkflowPath();",
        "updateAssetTagAnalysisWorkflowPath();",
        "updateDebugWorkflowPath();",
    ):
        assert updater in update_source
    assert "if (!normalizedBase) return '';" in frontend_source
    assert "function stripManagedLoraPath(path)" in frontend_source
    assert ".trim().replace(/\\\\?$/, '\\\\SOYA_CHAR_LORA')" not in frontend_source


def test_default_config_and_runtime_normalization_do_not_invent_workflow_names() -> None:
    import copy

    import server
    import workflow_profiles

    config = copy.deepcopy(server.DEFAULT_CONFIG)
    workflow_profiles.normalize_workflow_config(config)

    assert config["workflow_base_dir"] == ""
    assert config["comfy_workflow_source_path"] == ""
    assert set(config["illustration_workflow_source_paths"].values()) == {""}
    source_paths = {
        key: value
        for key, value in config.items()
        if key.endswith("_workflow_source_path")
    }
    assert set(source_paths.values()) == {""}
    assert set(config["lora_training_workflow_source_paths"].values()) == {""}
    assert set(config["style_lora_training_workflow_source_paths"].values()) == {""}


def test_library_covers_every_distributed_config_workflow_binding() -> None:
    import json

    import server
    from comfy_installer.workflow_library import latest_release_version

    library_root = Path("comfy_workflow_library")
    release = latest_release_version(library_root)
    manifest_path = library_root / "SOYA_DISTRIBUTION" / release / ".soya-pack.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    library_bindings = {
        str(binding)
        for item in manifest["items"]
        for binding in item["bindings"]
    }

    config_bindings = set()
    for key, value in server.DEFAULT_CONFIG.items():
        if key == "comfy_workflow_source_path" or key.endswith(
            "_workflow_source_path"
        ):
            config_bindings.add(key)
        elif key.endswith("_workflow_source_paths") and isinstance(value, dict):
            config_bindings.update(f"{key}.{child_key}" for child_key in value)

    assert library_bindings - config_bindings == set()
    assert config_bindings - library_bindings == {"outfit_workflow_source_path"}


def test_installed_compatibility_mode_is_reused_for_updates(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    state_root = tmp_path / "comfy" / ".installer-state"
    state_root.mkdir(parents=True)
    (state_root / "install-result-20260730_120000_000001.json").write_text(
        json.dumps(
            {
                "operation": "install",
                "install_mode": "nvidia_compatibility",
                "python": {"profile": "nvidia-cu128"},
            }
        ),
        encoding="utf-8",
    )

    assert service._installed_install_mode() == "nvidia_compatibility"


def test_v4_migration_backs_up_copies_and_retargets_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    embedded_comfy = tmp_path / "comfy"
    old_lora = (
        old_comfy
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "character.safetensors"
    )
    old_lora.parent.mkdir(parents=True)
    old_lora.write_bytes(b"lora")
    (embedded_comfy / ".git").mkdir(parents=True)
    original = {
        "comfy_input_dir": str(old_comfy / "input"),
        "nested": {
            "lora": str(
                old_comfy / "models" / "loras" / "SOYA_CHAR_LORA"
            )
        },
    }
    config.write_text(
        json.dumps(original, ensure_ascii=False),
        encoding="utf-8",
    )
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=requirements,
    )

    result = service.migrate_from_existing_comfy(old_comfy)

    backup = Path(result["config"]["backup_path"])
    assert backup.parent == (
        embedded_comfy / ".installer-state" / "backups" / "config"
    )
    assert not requirements.exists()
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    updated = json.loads(config.read_text(encoding="utf-8"))
    assert updated["comfy_input_dir"] == str(embedded_comfy / "input")
    assert updated["workflow_base_dir"] == str(
        embedded_comfy
        / "user"
        / "default"
        / "workflows"
        / USER_WORKFLOW_DIRNAME
    )
    assert updated["nested"]["lora"] == str(
        embedded_comfy / "models" / "loras" / "SOYA_CHAR_LORA"
    )
    assert (
        embedded_comfy
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "character.safetensors"
    ).read_bytes() == b"lora"
    assert result["config"]["updated_paths"] == [
        "$.workflow_base_dir",
        "$.comfy_input_dir",
        "$.nested.lora",
    ]
    assert result["config"]["already_retargeted"] is False

    repeated = service.migrate_from_existing_comfy(old_comfy)

    assert repeated["copied"] == []
    assert len(repeated["skipped"]) == 1
    assert repeated["config"]["updated_paths"] == []
    assert repeated["config"]["already_retargeted"] is True


def test_config_only_retarget_uses_installer_backup_directory(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    embedded_comfy = tmp_path / "comfy"
    (embedded_comfy / ".git").mkdir(parents=True)
    external_workflow = (
        tmp_path
        / "external-comfy"
        / "user"
        / "default"
        / "workflows"
        / "main.json"
    )
    config.write_text(
        json.dumps(
            {
                "comfy_input_dir": "",
                "comfy_workflow_source_path": str(external_workflow),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=requirements,
    )

    result = service.retarget_config_to_embedded()

    backup = Path(result["config"]["backup_path"])
    assert backup.parent == service.config_backup_dir
    assert not requirements.exists()
    updated = json.loads(config.read_text(encoding="utf-8"))
    assert updated["comfy_input_dir"] == str(embedded_comfy / "input")
    assert updated["workflow_base_dir"] == str(
        embedded_comfy
        / "user"
        / "default"
        / "workflows"
        / USER_WORKFLOW_DIRNAME
    )
    assert updated["comfy_workflow_source_path"] == str(
        embedded_comfy / "user" / "default" / "workflows" / "main.json"
    )


def test_config_only_retarget_fills_missing_paths_from_library_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    embedded_comfy = tmp_path / "comfy"
    workflow_root = embedded_comfy / "user" / "default" / "workflows" / "SOYA_USER"
    (embedded_comfy / ".git").mkdir(parents=True)
    workflow_root.mkdir(parents=True)
    default_tag = workflow_root / "tag__v4.json"
    default_debug = workflow_root / "debug__v4.json"
    default_qwen = workflow_root / "배포_qwen_edit_v1__v4.json"
    default_tag.write_text("{}\n", encoding="utf-8")
    default_debug.write_text("{}\n", encoding="utf-8")
    default_qwen.write_text("{}\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "comfy_input_dir": str(embedded_comfy / "input"),
                "tag_analysis_workflow_source_path": "",
                "qwen_edit_workflow_source_path": str(
                    tmp_path
                    / "mode_workflow"
                    / "배포_qwen_edit_v1_변환전.json"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_import_defaults(**_kwargs):
        return WorkflowSelection(
            release_version="v4",
            selected_item_ids=("tag", "debug", "qwen"),
            workflow_bindings={
                "tag_analysis_workflow_source_path": str(default_tag),
                "debug_workflow_source_path": str(default_debug),
                "qwen_edit_workflow_source_path": str(default_qwen),
            },
            model_ids=(),
            user_files=(str(default_tag), str(default_debug), str(default_qwen)),
        )

    monkeypatch.setattr(
        service_module,
        "import_default_user_copies",
        fake_import_defaults,
    )
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "요구사항",
    )

    result = service.retarget_config_to_embedded()

    updated = json.loads(config.read_text(encoding="utf-8"))
    assert updated["tag_analysis_workflow_source_path"] == str(default_tag)
    assert updated["debug_workflow_source_path"] == str(default_debug)
    assert updated["qwen_edit_workflow_source_path"] == str(default_qwen)
    assert set(result["config"]["updated_paths"]) == {
        "$.workflow_base_dir",
        "$.tag_analysis_workflow_source_path",
        "$.debug_workflow_source_path",
        "$.qwen_edit_workflow_source_path",
    }


def test_v4_migration_runs_in_background_and_publishes_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    worker_started = Event()
    allow_finish = Event()

    def fake_migration(_old_comfy_root, **kwargs):
        kwargs["set_phase"]("migration_scan")
        kwargs["progress"](
            {
                "event": "migration_copy",
                "engine": "robocopy",
                "current": 1,
                "total": 2,
                "overall_downloaded": 10,
                "overall_total": 20,
                "bytes_per_second": 5,
                "eta_seconds": 2,
            }
        )
        worker_started.set()
        assert allow_finish.wait(timeout=5)
        return {
            "copy_engine": "robocopy",
            "pending_bytes": 20,
            "copied": ["one", "two"],
            "skipped": ["existing"],
            "missing": [],
            "failures": [],
            "config": {
                "updated_paths": [],
                "missing_targets": [],
                "already_retargeted": True,
            },
        }

    monkeypatch.setattr(service, "_perform_migration", fake_migration)

    started = service.start_migration(str(tmp_path / "old-comfy"))
    assert worker_started.wait(timeout=5)
    running = service.status()
    assert started["operation"] == "migrate"
    assert running["state"] == "running"
    assert running["progress"]["engine"] == "robocopy"
    assert running["progress"]["eta_seconds"] == 2

    allow_finish.set()
    assert service._thread is not None
    service._thread.join(timeout=5)
    finished = service.status()
    assert finished["state"] == "succeeded"
    assert finished["operation"] == "migrate"
    assert finished["result"]["copy_engine"] == "robocopy"
    assert finished["result"]["copied"] == ["one", "two"]


def test_comfy_warning_is_visible_in_status_without_console_echo(
    tmp_path: Path,
    capsys,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )

    service._log_comfy(
        "[Comfy][WARNING] 'lora key not loaded' 경고 누적 100건"
    )

    assert capsys.readouterr().out == ""
    entry = service.status()["logs"][-1]
    assert entry["level"] == "warning"
    assert "누적 100건" in entry["message"]


def test_runtime_e2e_records_failure_and_continues_remaining_workflows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    validations = [
        SimpleNamespace(
            binding_keys=(f"fixture.{name}",),
            filename=f"{name}.json",
            prompt={},
            workflow={"nodes": []},
        )
        for name in ("a", "b", "c")
    ]
    calls: list[str] = []

    def fake_execute_prompt(**kwargs):
        filename = kwargs["filename"]
        calls.append(filename)
        if filename == "a.json":
            raise ComfyE2EError("fixture execution error")
        return {
            "filename": filename,
            "prompt_id": f"prompt-{filename}",
            "status": "success",
            "outputs": [],
            "output_data": {},
        }

    monkeypatch.setattr(
        service_module,
        "make_e2e_prompt",
        lambda _validation: {},
    )
    monkeypatch.setattr(
        service_module,
        "execute_prompt",
        fake_execute_prompt,
    )
    monkeypatch.setattr(
        service_module,
        "promote_generated_fixture",
        lambda **_kwargs: None,
    )
    process = SimpleNamespace(
        base_url="http://127.0.0.1:12345",
        process=SimpleNamespace(poll=lambda: None),
    )

    with pytest.raises(ComfyE2EError, match="성공 2/3, 실패 1/3"):
        service._run_runtime_e2e(
            process=process,
            validations=validations,
            fixtures={
                "training": "fixture/training/sample.png",
                "face_source": "fixture/fallback/face.webp",
            },
        )

    assert calls == ["a.json", "b.json", "c.json"]
    progress = service.status()["progress"]
    assert progress["current"] == 3
    assert progress["succeeded"] == 2
    assert progress["failed"] == 1
    assert progress["failed_filenames"] == ["a.json"]


def test_runtime_e2e_can_isolate_each_workflow_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    validations = [
        SimpleNamespace(
            binding_keys=(f"fixture.{name}",),
            filename=f"{name}.json",
            prompt={},
            workflow={"nodes": []},
        )
        for name in ("a", "b", "c")
    ]
    stopped: list[str] = []

    def process(name: str):
        return SimpleNamespace(
            base_url=f"http://{name}",
            process=SimpleNamespace(poll=lambda: None),
            stop=lambda: stopped.append(name),
        )

    created = iter((process("second"), process("third")))
    calls: list[str] = []

    monkeypatch.setattr(service_module, "make_e2e_prompt", lambda _value: {})
    monkeypatch.setattr(
        service_module,
        "execute_prompt",
        lambda **kwargs: {
            "filename": kwargs["filename"],
            "prompt_id": f"prompt-{kwargs['filename']}",
            "status": "success",
            "outputs": [],
            "output_data": calls.append(kwargs["base_url"]),
        },
    )
    monkeypatch.setattr(
        service_module,
        "promote_generated_fixture",
        lambda **_kwargs: None,
    )

    results = service._run_runtime_e2e(
        process=process("first"),
        process_factory=lambda: next(created),
        validations=validations,
        fixtures={
            "training": "fixture/training/sample.png",
            "face_source": "fixture/fallback/face.webp",
        },
    )

    assert len(results) == 3
    assert all("duration_seconds" in result for result in results)
    assert calls == ["http://first", "http://second", "http://third"]
    assert stopped == ["first", "second", "third"]


def test_manifest_has_fully_pinned_windows_runtime_and_assets() -> None:
    manifest = load_install_manifest()
    assert manifest.data["schema_version"] == 2
    assert manifest.comfy["version"] == "0.31.0"
    assert manifest.comfy["ref"] == (
        "62b3c94bd45154f6486c7abf1b9efcacee96ea69"
    )
    assert manifest.python["version"] == "3.12.11"
    assert manifest.python["compatibility_packages"] == [
        "numpy==1.26.4",
        "scipy==1.14.1",
        "tifffile==2024.9.20",
        "opencv-python==4.10.0.84",
        "opencv-python-headless==4.10.0.84",
        "opencv-contrib-python==4.10.0.84",
    ]
    nvidia = {
        profile["id"]: profile
        for profile in manifest.python["gpu_profiles"]
        if profile["kind"] == "nvidia"
    }
    assert set(nvidia) == {"nvidia-cu128", "nvidia-cu130"}
    assert nvidia["nvidia-cu128"]["minimum_driver_version"] == "570.65"
    assert nvidia["nvidia-cu128"]["minimum_compute_capability"] == "8.0"
    assert nvidia["nvidia-cu128"]["torch_cuda"] == "12.8"
    assert nvidia["nvidia-cu128"]["sageattention"]["size"] == 12252026
    assert nvidia["nvidia-cu128"]["sageattention"]["sha256"] == (
        "b8b3134d00dfbdae5c10cc34cc8508891d9420adaa182502fa30a496428531ed"
    )
    assert nvidia["nvidia-cu130"]["minimum_driver_version"] == "580.00"
    assert nvidia["nvidia-cu130"]["minimum_compute_capability"] == "8.0"
    assert nvidia["nvidia-cu130"]["torch_cuda"] == "13.0"
    assert nvidia["nvidia-cu130"]["sageattention"]["size"] == 12321863
    assert all(
        len(profile["sageattention"]["sha256"]) == 64
        for profile in nvidia.values()
    )
    assert len(manifest.custom_nodes) == 15
    assert all(
        node["name"] != "ComfyUI-Manager"
        for node in manifest.custom_nodes
    )
    lora_manager = next(
        node
        for node in manifest.custom_nodes
        if node["name"] == "comfyui-lora-manager"
    )
    assert lora_manager["repository"] == (
        "https://github.com/willmiao/ComfyUI-Lora-Manager.git"
    )
    assert lora_manager["ref"] == (
        "0d8805cdee93d1a7347a813f4ec271ba6bcb55f5"
    )
    spectrum = next(
        node
        for node in manifest.custom_nodes
        if node["name"] == "comfyui-spectrum-ksampler"
    )
    assert spectrum["ref"] == "c806917566ee1c149575cd90da9e4c2e543de019"
    tracking_main_names = {
        node["name"]
        for node in manifest.custom_nodes
        if node.get("tracking_branch") == "main"
    }
    assert tracking_main_names == {
        "comfyui-instant-lora_v_soya",
        "comfyui-soya-custom-nodes",
        "comfyui-workflow-to-api-converter-endpoint",
    }
    assert all(
        "ref" not in node
        for node in manifest.custom_nodes
        if node["name"] in tracking_main_names
    )
    assert manifest.latest_workflow_release == "v2"
    latest_release = manifest.workflows["release_dependencies"][
        manifest.latest_workflow_release
    ]
    assert manifest.latest_workflow_count == len(latest_release)
    assert len({item["id"] for item in latest_release}) == len(latest_release)
    latest_bindings = {
        binding
        for item in latest_release
        for binding in item["bindings"]
    }
    assert latest_bindings == set(manifest.workflows["required_bindings"]).union(
        manifest.workflows["optional_bindings"]
    )
    h3 = manifest.validation_profiles["minimax_h3"]
    assert set(h3["workflow_bindings"]).issubset(latest_bindings)
    assert set(h3["fast_workflow_bindings"]).issubset(
        h3["workflow_bindings"]
    )
    assert set(h3["model_ids"]).issubset(
        {model["id"] for model in manifest.models}
    )
    for defaults_key in ("defaults", "fast_defaults"):
        defaults = h3[defaults_key]
        assert all(
            isinstance(defaults[key], int) and defaults[key] > 0
            for key in ("width", "height", "steps")
        )


def test_manifest_allows_release_content_to_change_without_python_constants(
    tmp_path: Path,
) -> None:
    current = load_install_manifest()
    data = json.loads(json.dumps(current.data, ensure_ascii=False))
    latest_entries = data["workflows"]["release_dependencies"]["v2"]
    data["workflows"]["release_dependencies"] = {
        "v7": latest_entries,
        "v10": latest_entries,
    }
    data["workflows"]["excluded_filenames"] = ["future-exclusion.json"]
    h3 = data["validation_profiles"]["minimax_h3"]
    h3["workflow_bindings"] = [h3["workflow_bindings"][0]]
    h3["fast_workflow_bindings"] = []
    h3.pop("fast_defaults")
    h3["model_ids"] = [h3["model_ids"][0]]
    h3["defaults"]["width"] = 1024
    h3["defaults"]["height"] = 576
    h3["defaults"]["steps"] = 6
    path = tmp_path / "install-manifest.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = load_install_manifest(path)

    assert loaded.latest_workflow_release == "v10"
    assert loaded.latest_workflow_count == len(latest_entries)
    assert loaded.validation_profiles["minimax_h3"]["defaults"] == h3["defaults"]
