from __future__ import annotations

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
        assert payload["manifest"]["workflow_count"] == 17
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
async def test_installer_start_rejects_non_boolean_restore_option(
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
                "upload_id": "0" * 32,
                "workflow_key": "secret",
                "civitai_key": "secret",
                "restore_config_after_success": "true",
            },
        )
        assert response.status == 409
        payload = await response.json()
        assert payload["ok"] is False
        assert "boolean" in payload["error"]
    finally:
        await client.close()
