import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("release", ["v2", "v3", "v17"])
@pytest.mark.parametrize("copy", ["original", "installed", "collision", "renamed", "edited"])
def test_cloud_uses_selected_pack_metadata(tmp_path, release, copy):
    from modal_backend.manifest import model_entries_for_workflow_files
    payload = b'{"nodes": []}'
    model = {"id": "pack-model", "relative_path": "models/checkpoints/pack.safetensors", "url": "https://example.invalid/pack", "sha256": "a" * 64}
    added = {"id": "extra", "relative_path": "models/loras/extra.safetensors"}
    pack = {"items": [{"filename": "workflow.json", "sha256": hashlib.sha256(payload).hexdigest(), "bindings": ["workflow_path"], "model_ids": ["pack-model"]}], "install_manifest": {"models": [model, added]}}
    write_json(tmp_path / "comfy_workflow_library" / "SOYA_DISTRIBUTION" / release / ".soya-pack.json", pack)
    # A second installed release must not override the selected release's metadata.
    write_json(tmp_path / "comfy_workflow_library" / "SOYA_DISTRIBUTION" / "v1" / ".soya-pack.json", {"items": [{"filename": "workflow.json", "model_ids": ["wrong"]}], "install_manifest": {"models": [{"id": "wrong"}]}})
    write_json(tmp_path / "comfy_installer" / "resources" / "install_manifest.json", {"models": [{"id": "pack-model", "url": "old"}], "workflows": {"items": []}})
    name = {"original": "workflow.json", "installed": f"workflow__{release}.json", "collision": f"workflow__{release}_2.json", "renamed": "renamed.json", "edited": f"workflow__{release}.json"}[copy]
    user = tmp_path / "comfy" / "user" / "default" / "workflows" / "SOYA_USER"
    user.mkdir(parents=True)
    (user / name).write_bytes(payload if copy != "edited" else b'{"1": {"inputs": {"lora": "extra.safetensors"}}}')
    entries = model_entries_for_workflow_files(tmp_path, [name])
    assert model in entries
    assert {e["id"] for e in entries} == ({"pack-model", "extra"} if copy == "edited" else {"pack-model"})


def test_items_metadata_controls_model_scope():
    from comfy_installer.model_scope import binding_model_ids, manifest_binding_ids
    workflows = {"items": [{"id": "item", "bindings": ["workflow_path"], "model_ids": ["current"]}], "release_dependencies": {"v1": [{"id": "workflow_path", "model_ids": ["stale"]}]}}
    assert manifest_binding_ids(workflows) == {"workflow_path"}
    assert binding_model_ids(workflows, ["workflow_path"]) == {"current"}
    assert not binding_model_ids(workflows, ["different_path"])


@pytest.mark.asyncio
async def test_cloud_service_sends_authoritative_model_entries(tmp_path, monkeypatch):
    from modal_backend import service as service_module
    from modal_backend.settings import ModalSettings
    entry = {"id": "new-pack-model", "url": "https://example.invalid/new", "relative_path": "models/checkpoints/new.bin", "sha256": "a" * 64}
    monkeypatch.setattr(service_module, "model_entries_for_workflow_files", lambda *args: [entry])
    service = service_module.ModalService(tmp_path, lambda: {})
    calls = []

    async def run_command(*args, **kwargs):
        calls.append(kwargs["stdin_payload"])
        return 0, json.dumps({"ok": True, "result": {"results": [{"id": entry["id"], "state": "already_present"}]}}), ""

    monkeypatch.setattr(service, "_run_command", run_command)
    await service._sync_models_direct(ModalSettings.from_mapping({}), ["selected__v17.json"])
    assert calls[0]["model_entries"] == [entry]
    assert calls[0]["model_ids"] == [entry["id"]]


def test_cloud_worker_accepts_metadata_newer_than_deployed_image(tmp_path, monkeypatch):
    import ast
    import io
    import time
    import traceback
    import urllib.request
    from types import SimpleNamespace
    from modal_backend.modal_app import volume_target_path
    source = (ROOT / "modal_backend/modal_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sync_models_from_source")
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    models, loras = tmp_path / "models", tmp_path / "loras"
    models.mkdir()
    loras.mkdir()
    real_path = Path

    def mapped_path(value):
        if str(value) == "/loras":
            return loras
        if str(value) == "/opt/soya/install_manifest.json":
            raise AssertionError("must use supplied pack metadata")
        return real_path(value)

    namespace = {"Path": mapped_path, "os": os, "json": json, "time": time, "traceback": traceback, "_announce_call_started": lambda *args: None, "COMFY_MODELS_MOUNT_PATH": str(models), "volume_target_path": volume_target_path, "models_volume": SimpleNamespace(commit=lambda: None), "loras_volume": SimpleNamespace(commit=lambda: None)}
    payload = b"new pack model"
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **kw: io.BytesIO(payload)))
    exec(compile(module, "<sync-models-test>", "exec"), namespace)
    entry = {"id": "new", "relative_path": "models/checkpoints/new.bin", "url": "https://example.invalid/new", "sha256": hashlib.sha256(payload).hexdigest()}
    result = namespace["sync_models_from_source"](["new"], model_entries=[entry])
    assert result["results"][0]["state"] == "downloaded"
    assert (models / "checkpoints/new.bin").read_bytes() == payload


@pytest.mark.parametrize("platform_name,tags,expected", [("Darwin", ["macosx_14_0_arm64"], False), ("Windows", ["win_amd64"], True), ("Linux", ["manylinux_2_17_x86_64"], False)])
def test_legacy_windows_wheel_is_skipped_on_other_platforms(monkeypatch, platform_name, tags, expected):
    from comfy_installer import dependency_installer as installer
    monkeypatch.setattr(installer.platform, "system", lambda: platform_name)
    monkeypatch.setattr(installer, "platform_tags", lambda: iter(tags))
    manifest = json.loads((ROOT / "comfy_installer/resources/install_manifest.json").read_text(encoding="utf-8"))
    wheel = manifest["python"]["preinstall_wheels"][0]
    assert installer._wheel_matches_platform(wheel) == expected
    assert installer._wheel_matches_platform({"filename": "example-1.0-py3-none-any.whl"})
    assert not installer._wheel_matches_platform({"filename": "example-1.0-py3-none-any.whl", "platforms": ["Unsupported"]})


def test_qwen_cleanup_uses_staging_time_config(tmp_path):
    from modes.qwen_edit_mode import QwenEditMode, QWEN_EDIT_INPUT_SUBDIR
    mode = object.__new__(QwenEditMode)
    mode._pending_inputs = {"a-job": {"source": b"source", "mask": b"mask"}}
    original = tmp_path / "original"
    other = tmp_path / "other"
    original.mkdir()
    sibling = other / QWEN_EDIT_INPUT_SUBDIR / "a-job"
    sibling.mkdir(parents=True)
    (sibling / "keep.png").write_bytes(b"keep")
    params = {"job_id": "a-job"}
    staged = Path(mode._prepare_shared_comfy_inputs(params, {"comfy_input_dir": str(original)}))
    mode.cleanup_staged_request(params, {"comfy_input_dir": str(other)})
    assert not staged.exists()
    assert (sibling / "keep.png").read_bytes() == b"keep"


def test_training_cleanup_preserves_local_folder_with_similar_name(tmp_path):
    from queue_manager import _cleanup_remote_training_staging
    shared = tmp_path / "modal_jobs" / "my-local-data"
    shared.mkdir(parents=True)
    (shared / "original.png").write_bytes(b"original")
    _cleanup_remote_training_staging(str(shared))
    _cleanup_remote_training_staging(str(shared), staging_root=str(tmp_path / "actual-staging"), job_id="job-1")
    assert (shared / "original.png").read_bytes() == b"original"


@pytest.mark.parametrize("scope,key", [("SOYA_BOT_LORA", "bot_lora_load_path"), ("SOYA_INSTANCE_LORA", "instance_lora_load_path"), ("SOYA_STYLE_LORA", "style_lora_load_path")])
def test_remote_lora_returns_to_its_configured_picker_folder(tmp_path, scope, key):
    from modal_backend.service import ModalService
    source = tmp_path / "download.bin"
    source.write_bytes(b"result")
    config = {"lora_load_path": str(tmp_path / "main"), "bot_lora_load_path": str(tmp_path / "bots"), "instance_lora_load_path": str(tmp_path / "instances"), "style_lora_load_path": str(tmp_path / "styles")}
    service = ModalService(tmp_path, lambda: {})
    stored = service._store_modal_artifacts([{"path": str(source), "relative_path": f"{scope}/some/model.bin"}], config)
    expected = Path(config[key]) / "some/model.bin"
    assert Path(stored[0]["local_path"]) == expected
    assert expected.read_bytes() == b"result"


def test_app_update_preserves_linked_user_directory(tmp_path):
    updater = updater_module()
    payload, data, outside = tmp_path / "payload", tmp_path / "data", tmp_path / "outside"
    (payload / "prompts").mkdir(parents=True)
    data.mkdir()
    outside.mkdir()
    (payload / "prompts/default.md").write_text("new", encoding="utf-8")
    (outside / "default.md").write_text("original", encoding="utf-8")
    try:
        (data / "prompts").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlink unavailable: {exc}")
    updater.inventory(payload)
    assert updater.update(payload, data) == ["prompts/default.md"]
    assert (outside / "default.md").read_text() == "original"


def test_config_logs_do_not_copy_any_slot_secrets(tmp_path, monkeypatch, capsys):
    import server
    config_path = tmp_path / "config.json"
    old, new = {}, {}
    for slot in range(1, 11):
        suffix = "" if slot == 1 else str(slot)
        for field in ("llm_custom_headers", "llm_custom_body", "llm_api_key"):
            old[field + suffix] = f"secret-before-{field}-{slot}"
            new[field + suffix] = f"secret-after-{field}-{slot}"
    write_json(config_path, old)
    monkeypatch.setattr(server, "CONFIG_FILE", str(config_path))
    monkeypatch.setattr(server, "RUNTIME_BACKUP_DIR", str(tmp_path / "backups"))
    server.save_config(new)
    output = capsys.readouterr().out
    assert "[CONFIG_DIFF]" in output
    assert all(value not in output for value in [*old.values(), *new.values()])
    assert json.loads(config_path.read_text(encoding="utf-8")) == new
    backups = list((tmp_path / "backups/config").glob("*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == old


def test_settings_fetch_failure_prevents_dom_reads_and_save():
    import runpy
    helpers = runpy.run_path(str(ROOT / "tests/test_frontend_settings_form_guard.py"))
    _slice_function, FRONTEND = helpers["_slice_function"], helpers["FRONTEND"]
    populate = _slice_function(FRONTEND, "async function populateSettingsForm()")
    save = _slice_function(FRONTEND, "async function saveSettings()")
    script = """
const assert = require('node:assert/strict');
let settingsFormPopulated = false;
let currentConfig = {saved: true};
const loadCurrentConfig = async () => { throw new Error('offline'); };
const document = {getElementById: () => { throw new Error('unexpected DOM read'); }};
""" + populate + "\n" + save + """
(async () => {
  await assert.rejects(saveSettings(), /offline/);
  assert.equal(settingsFormPopulated, false);
  assert.deepEqual(currentConfig, {saved: true});
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
    result = subprocess.run(["node", "-"], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def updater_module():
    spec = importlib.util.spec_from_file_location("soya_update", ROOT / "packaging/macos/update_payload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_update_preserves_data_and_updates_complete_inventory(tmp_path):
    updater = updater_module()
    payload, data = tmp_path / "payload", tmp_path / "data"
    payload.mkdir()
    files = ["server.py", "prompts/default.md", "customprompt/script.py", "comfy_installer/resources/install_manifest.json", "vast_backend/service.py", ".python-version", "retired.py"]
    for name in files:
        file = payload / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("old", encoding="utf-8")
    updater.inventory(payload)
    updater.update(payload, data)
    for name in ["prompts/default.md", "comfy_installer/resources/install_manifest.json"]:
        (data / name).write_text("user edited", encoding="utf-8")
    extra = data / "customprompt/history.json"
    extra.write_text("history", encoding="utf-8")
    for name in files:
        (payload / name).write_text("new", encoding="utf-8")
    (payload / "retired.py").unlink()
    updater.inventory(payload)
    conflicts = updater.update(payload, data)
    assert set(conflicts) == {"prompts/default.md", "comfy_installer/resources/install_manifest.json"}
    assert extra.read_text() == "history"
    assert not (data / "retired.py").exists()
    for name in files[:-1]:
        assert (data / name).read_text() == ("user edited" if name in conflicts else "new")
    assert list((data / "backups/app_updates").glob("*/previous/server.py"))
    assert list((data / "backups/app_updates").glob("*/incoming/prompts/default.md"))
    assert set(updater.update(payload, data)) == set(conflicts)


def test_app_partial_first_install_can_resume_without_overwriting(tmp_path):
    updater = updater_module()
    payload, data = tmp_path / "payload", tmp_path / "data"
    payload.mkdir()
    data.mkdir()
    (payload / "server.py").write_text("code", encoding="utf-8")
    (payload / "prompt.md").write_text("default", encoding="utf-8")
    (data / "prompt.md").write_text("local", encoding="utf-8")
    updater.inventory(payload)
    assert updater.update(payload, data) == ["prompt.md"]
    assert (data / "server.py").read_text() == "code"
    assert (data / "prompt.md").read_text() == "local"
