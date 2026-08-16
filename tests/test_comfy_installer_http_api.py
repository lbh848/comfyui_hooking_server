from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from comfy_installer.http_api import register_comfy_installer_routes


@pytest.mark.asyncio
async def test_installer_status_and_pack_upload_routes(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    requirements = tmp_path / "requirements"
    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=requirements,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/comfy-installer/status")
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["state"] == "idle"
        assert payload["manifest"]["workflow_count"] > 0
        assert "civitai_key" not in str(payload)
        assert "workflow_key" not in str(payload)

        form = FormData()
        form.add_field(
            "pack",
            b"SOYAWFP1" + b"\x00" * 32,
            filename="workflows.soyawfp",
            content_type="application/octet-stream",
        )
        response = await client.post(
            "/api/comfy-installer/workflow-pack", data=form
        )
        assert response.status == 200
        uploaded = await response.json()
        assert uploaded["ok"] is True
        assert len(uploaded["upload_id"]) == 32
        saved = (
            tmp_path
            / ".work"
            / "comfy-installer"
            / "uploads"
            / f"{uploaded['upload_id']}.soyawfp"
        )
        assert saved.read_bytes().startswith(b"SOYAWFP1")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_e2e_catalog_and_start_routes_use_selected_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    catalog = {
        "release_version": "v1",
        "source_kind": "read_only_distribution_original",
        "items": [
            {
                "id": "debug_workflow_source_path",
                "e2e_profiles": ["standard"],
            }
        ],
        "skipped_count": 0,
    }
    received: list[dict] = []
    monkeypatch.setattr(service, "e2e_workflow_catalog", lambda: catalog)

    def fake_start_e2e(**kwargs):
        received.append(kwargs)
        return {"state": "running", "operation": "e2e"}

    monkeypatch.setattr(service, "start_e2e", fake_start_e2e)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/comfy-installer/e2e-catalog")
        assert response.status == 200
        payload = await response.json()
        assert payload == {"ok": True, "catalog": catalog}

        response = await client.post(
            "/api/comfy-installer/e2e",
            json={
                "release_version": "v1",
                "selected_item_ids": ["debug_workflow_source_path"],
            },
        )
        assert response.status == 200
        assert await response.json() == {
            "ok": True,
            "state": "running",
            "operation": "e2e",
        }
        assert received == [
            {
                "release_version": "v1",
                "selected_item_ids": ["debug_workflow_source_path"],
            }
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_installer_rejects_non_pack_upload(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field(
            "pack",
            b"not-a-workflow-pack",
            filename="bad.bin",
            content_type="application/octet-stream",
        )
        response = await client.post(
            "/api/comfy-installer/workflow-pack", data=form
        )
        assert response.status == 400
        payload = await response.json()
        assert payload["ok"] is False
        assert "SOYAWFP1" in payload["error"]
        invalid_files = list(
            (
                tmp_path / ".work" / "comfy-installer" / "uploads"
            ).glob("*.invalid*")
        )
        assert len(invalid_files) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_installer_start_rejects_invalid_selected_item_ids(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/start",
            json={
                "release_version": "v1",
                "selected_item_ids": "not-a-list",
            },
        )
        assert response.status == 409
        payload = await response.json()
        assert payload["ok"] is False
        assert "문자열 배열" in payload["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_civitai_key_api_returns_plain_saved_value(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/civitai-key",
            json={"api_key": "plain-civitai-key"},
        )
        assert response.status == 200
        assert (await response.json())["api_key"] == "plain-civitai-key"

        response = await client.get("/api/comfy-installer/civitai-key")
        assert response.status == 200
        assert (await response.json())["api_key"] == "plain-civitai-key"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_troubleshooting_replaces_installer_and_lora_manager_civitai_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    (tmp_path / "comfy" / "custom_nodes" / "comfyui-lora-manager").mkdir(
        parents=True
    )
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    lora_settings = (
        local_app_data / "ComfyUI-LoRA-Manager" / "settings.json"
    )
    lora_settings.parent.mkdir(parents=True)
    lora_settings.write_text(
        json.dumps({"civitai_api_key": "old-key", "theme": "dark"}),
        encoding="utf-8",
    )

    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/troubleshooting/civitai-key",
            json={"api_key": "replacement-key"},
        )
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["key_set"] is True
        assert payload["restart_required"] is True
        assert "replacement-key" not in json.dumps(payload)

        installer_key = json.loads(
            (tmp_path / "key" / "civitai_key.json").read_text(encoding="utf-8")
        )
        assert installer_key == {"api_key": "replacement-key"}
        updated_lora_settings = json.loads(lora_settings.read_text(encoding="utf-8"))
        assert updated_lora_settings == {
            "civitai_api_key": "replacement-key",
            "theme": "dark",
        }
        assert Path(payload["backup_paths"]["lora_manager"]).parent == (
            lora_settings.parent / "backups"
        )
        assert not (tmp_path / "requirements").exists()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_preflight_uses_selected_workflow_model_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    received = {}

    def fake_preflight_selection(**kwargs):
        received.update(kwargs)
        return {
            "selection": {"model_ids": ["fixed-model"], "model_bytes": 123},
            "disk": {"free": 999, "required": 123, "enough": True},
        }

    monkeypatch.setattr(service, "preflight_selection", fake_preflight_selection)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/preflight",
            json={
                "release_version": "v1",
                "selected_item_ids": ["qwen_edit_workflow_source_path"],
            },
        )
        assert response.status == 200
        assert received == {
            "release_version": "v1",
            "selected_item_ids": ["qwen_edit_workflow_source_path"],
            "install_mode": "standard",
        }
        payload = await response.json()
        assert payload["preflight"]["selection"]["model_bytes"] == 123
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_installer_api_passes_nvidia_compatibility_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    received = {}

    def fake_start_install(**kwargs):
        received.update(kwargs)
        return {
            "state": "running",
            "operation": "install",
            "install_mode": kwargs["install_mode"],
            "logs": [],
        }

    monkeypatch.setattr(service, "start_install", fake_start_install)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/start",
            json={
                "release_version": "v1",
                "selected_item_ids": ["qwen_edit_workflow_source_path"],
                "install_mode": "nvidia_compatibility",
            },
        )
        assert response.status == 200
        assert received == {
            "release_version": "v1",
            "selected_item_ids": ["qwen_edit_workflow_source_path"],
            "install_mode": "nvidia_compatibility",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_preflight_without_body_keeps_general_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    monkeypatch.setattr(service, "preflight", lambda **kwargs: {"mode": "general"})
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/api/comfy-installer/preflight")
        assert response.status == 200
        assert (await response.json())["preflight"] == {"mode": "general"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_migration_api_starts_background_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    received = []

    def fake_start(old_comfy_root):
        received.append(old_comfy_root)
        return {
            "state": "running",
            "operation": "migrate",
            "progress": {"event": "migration_scan"},
            "logs": [],
        }

    monkeypatch.setattr(service, "start_migration", fake_start)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/migrate",
            json={"old_comfy_root": r"E:\\old-comfy"},
        )
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["state"] == "running"
        assert payload["operation"] == "migrate"
        assert received == [r"E:\\old-comfy"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_config_retarget_api_updates_without_starting_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    calls = []

    def fake_retarget():
        calls.append(True)
        return {
            "config": {
                "updated_paths": ["$.comfy_workflow_source_path"],
                "missing_targets": [],
                "already_retargeted": False,
            }
        }

    monkeypatch.setattr(service, "retarget_config_to_embedded", fake_retarget)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/api/comfy-installer/retarget-config")
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["config"]["updated_paths"] == [
            "$.comfy_workflow_source_path"
        ]
        assert calls == [True]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shutdown_after_update_requires_successful_update(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    shutdown_calls = []

    async def shutdown_after_update():
        shutdown_calls.append(True)
        return {"manager_shutdown_scheduled": True}

    app = web.Application()
    register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
        shutdown_after_update=shutdown_after_update,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/shutdown-after-update"
        )
        assert response.status == 409
        assert shutdown_calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shutdown_after_update_runs_once_after_success(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    shutdown_calls = []

    async def shutdown_after_update():
        shutdown_calls.append(True)
        return {
            "managed_comfy_instances": [1, 2],
            "manager_shutdown_scheduled": True,
        }

    app = web.Application()
    service = register_comfy_installer_routes(
        app,
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
        shutdown_after_update=shutdown_after_update,
    )
    with service._lock:
        service._state.update(
            {
                "state": "succeeded",
                "operation": "update",
                "result": {"restart_required": True},
            }
        )

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/comfy-installer/shutdown-after-update"
        )
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["shutdown"]["managed_comfy_instances"] == [1, 2]
        assert "재시작해주세요" in payload["message"]
        assert shutdown_calls == [True]

        duplicate = await client.post(
            "/api/comfy-installer/shutdown-after-update"
        )
        assert duplicate.status == 409
        assert shutdown_calls == [True]
    finally:
        await client.close()
