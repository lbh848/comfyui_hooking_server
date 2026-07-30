from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy_installer.patch_importer import (
    PATCH_BOT_MEMBER,
    PATCH_FORMAT,
    PATCH_FORMAT_VERSION,
    PATCH_MANIFEST_MEMBER,
    PATCH_PRESETS_MEMBER,
    PatchImportError,
    PatchImporter,
    register_patch_import_routes,
)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _presets(**overrides) -> dict:
    data = {
        "characters": {},
        "appearances": {},
        "outfits": {},
        "expressions": {},
        "composition_presets": {},
        "artist_presets": {},
        "quality_presets": {},
        "negative_presets": {},
    }
    data.update(overrides)
    return data


def _bot_payload() -> dict:
    return {
        "bot": {
            "name": "nikke",
            "characters": [
                {
                    "name": "alice",
                    "rep_images": ["alice.webp"],
                    "face_tags": "blonde hair",
                    "eye_tags": "blue eyes",
                }
            ],
            "system_prompt": "",
        }
    }


def _write_patch(
    path: Path,
    payload: dict[str, bytes],
    *,
    corrupt_digest_for: str | None = None,
) -> None:
    files = []
    for name, content in payload.items():
        digest = hashlib.sha256(content).hexdigest()
        if name == corrupt_digest_for:
            digest = "0" * 64
        files.append(
            {
                "path": name,
                "size": len(content),
                "sha256": digest,
            }
        )
    manifest = {
        "format": PATCH_FORMAT,
        "format_version": PATCH_FORMAT_VERSION,
        "patch_id": "test-patch",
        "name": "테스트 패치",
        "files": files,
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        archive.writestr(PATCH_MANIFEST_MEMBER, _json_bytes(manifest))
        for name, content in payload.items():
            archive.writestr(name, content)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_import_merges_conflicts_and_preserves_existing_user_data(
    tmp_path: Path,
) -> None:
    tags = _presets(
        characters={"표정프로필": {"appearance": "사용자 설정"}},
        appearances={
            "동일 외모": ["same"],
            "충돌 외모": ["local"],
        },
    )
    hidden = {"appearances": {}}
    bot = {
        "bots": [{"name": "NIKKE", "characters": []}],
        "positive_whitelist": [],
        "positive_blacklist": [],
        "system_prompt_presets": {},
    }
    _write_json(tmp_path / "asset_data" / "tags.json", tags)
    _write_json(tmp_path / "asset_data" / "hidden_tags.json", hidden)
    _write_json(tmp_path / "asset_data" / "bot.json", bot)

    _write_json(
        tmp_path / "chain_presets" / "동일 체인.json",
        {"chains": [{"value": "same"}], "repeat": 1},
    )
    _write_json(
        tmp_path / "chain_presets" / "충돌 체인.json",
        {"chains": [{"value": "local"}], "repeat": 1},
    )
    _write_json(tmp_path / "pose_data" / "기존 pose.json", {"local": True})
    (tmp_path / "pose_data" / "기존 pose.webp").write_bytes(b"local-preview")
    existing_asset = (
        tmp_path / "asset" / "Eren_soya" / "dress" / "smile" / "existing.webp"
    )
    existing_asset.parent.mkdir(parents=True, exist_ok=True)
    existing_asset.write_bytes(b"user-asset")

    payload = {
        PATCH_PRESETS_MEMBER: _json_bytes(
            _presets(
                characters={
                    "표정프로필": {"appearance": ""},
                    "Eren_soya": {
                        "appearance": "",
                        "outfit": "",
                        "expression": "",
                    },
                },
                appearances={
                    "동일 외모": ["same"],
                    "충돌 외모": ["patch"],
                    "신규 외모": ["new"],
                },
            )
        ),
        PATCH_BOT_MEMBER: _json_bytes(_bot_payload()),
        "payload/asset/Eren_soya/dress/smile/existing.webp": b"patch-asset",
        "payload/asset/Eren_soya/dress/smile/new.webp": b"new-asset",
        "payload/pose_data/기존 pose.json": _json_bytes({"patch": True}),
        "payload/pose_data/기존 pose.webp": b"patch-preview",
        "payload/pose_data/신규 pose.json": _json_bytes({"new": True}),
        "payload/pose_data/신규 pose.webp": b"new-preview",
        "payload/pose_data/models/model.onnx": b"model",
        "payload/chain_presets/동일 체인.json": _json_bytes(
            {"chains": [{"value": "same"}], "repeat": 1}
        ),
        "payload/chain_presets/충돌 체인.json": _json_bytes(
            {"chains": [{"value": "patch"}], "repeat": 1}
        ),
        "payload/chain_presets/신규 체인.json": _json_bytes(
            {"chains": [{"value": "new"}], "repeat": 2}
        ),
        "payload/workflow_backup/20260731_052057_f5265469.json": b"{}",
        "payload/workflow_backup/20260731_052057_f5265469.webp": b"backup",
        "payload/workflow_backup/20260731_052057_f5265469_info.json": b"{}",
        "payload/bot/nikke/alice/alice.webp": b"bot-image",
    }
    patch_path = tmp_path / "test.soyapatch"
    _write_patch(patch_path, payload)

    result = PatchImporter(tmp_path).import_package(patch_path)

    merged_tags = json.loads(
        (tmp_path / "asset_data" / "tags.json").read_text(encoding="utf-8")
    )
    merged_hidden = json.loads(
        (tmp_path / "asset_data" / "hidden_tags.json").read_text(
            encoding="utf-8"
        )
    )
    assert merged_tags["characters"]["표정프로필"] == {
        "appearance": "사용자 설정"
    }
    assert "Eren_soya" in merged_tags["characters"]
    assert merged_tags["appearances"]["동일 외모"] == ["same"]
    assert merged_tags["appearances"]["신규 외모"] == ["new"]
    assert merged_hidden["appearances"]["충돌 외모_conflict"] == ["patch"]

    assert existing_asset.read_bytes() == b"user-asset"
    assert existing_asset.with_name("new.webp").read_bytes() == b"new-asset"
    assert (
        tmp_path / "pose_data" / "기존 pose.json"
    ).read_text(encoding="utf-8").find("local") >= 0
    assert (
        tmp_path / "pose_data" / "기존 pose.webp"
    ).read_bytes() == b"local-preview"
    assert (tmp_path / "pose_data" / "신규 pose.json").is_file()
    assert (tmp_path / "pose_data" / "models" / "model.onnx").is_file()

    assert (tmp_path / "chain_presets" / "신규 체인.json").is_file()
    assert (
        tmp_path / "chain_presets" / "hidden" / "충돌 체인_conflict.json"
    ).is_file()
    assert not (tmp_path / "bot" / "nikke" / "alice" / "alice.webp").exists()
    assert (
        tmp_path / "workflow_backup" / "20260731_052057_f5265469.webp"
    ).is_file()

    backup_dir = tmp_path / "asset_data" / "backup"
    assert len(list(backup_dir.glob("tags_*.json"))) == 1
    assert len(list(backup_dir.glob("hidden_tags_*.json"))) == 1
    assert len(list(backup_dir.glob("bot_*.json"))) == 1
    assert result["summary"]["presets"]["conflicts_hidden"] == 1
    assert result["summary"]["chains"]["conflicts_hidden"] == 1
    assert result["summary"]["poses"]["skipped_names"] == 1
    assert result["summary"]["bot"]["skipped_existing"] == 1

    repeated = PatchImporter(tmp_path).import_package(patch_path)
    repeated_hidden = json.loads(
        (tmp_path / "asset_data" / "hidden_tags.json").read_text(
            encoding="utf-8"
        )
    )
    assert "충돌 외모_conflict_2" not in repeated_hidden["appearances"]
    assert repeated["summary"]["presets"]["conflicts_hidden"] == 0
    assert repeated["summary"]["chains"]["conflicts_hidden"] == 0


def test_import_adds_nikke_when_missing(tmp_path: Path) -> None:
    _write_json(tmp_path / "asset_data" / "tags.json", _presets())
    _write_json(tmp_path / "asset_data" / "hidden_tags.json", {})
    _write_json(
        tmp_path / "asset_data" / "bot.json",
        {
            "bots": [{"name": "other", "characters": []}],
            "positive_whitelist": [],
            "positive_blacklist": [],
            "system_prompt_presets": {},
        },
    )
    payload = {
        PATCH_PRESETS_MEMBER: _json_bytes(_presets()),
        PATCH_BOT_MEMBER: _json_bytes(_bot_payload()),
        "payload/bot/nikke/alice/alice.webp": b"alice",
    }
    patch_path = tmp_path / "bot.soyapatch"
    _write_patch(patch_path, payload)

    result = PatchImporter(tmp_path).import_package(patch_path)

    bot_data = json.loads(
        (tmp_path / "asset_data" / "bot.json").read_text(encoding="utf-8")
    )
    assert [item["name"] for item in bot_data["bots"]] == ["other", "nikke"]
    assert (
        tmp_path / "bot" / "nikke" / "alice" / "alice.webp"
    ).read_bytes() == b"alice"
    assert result["summary"]["bot"]["added"] == 1


def test_corrupt_patch_is_rejected_before_runtime_backup(tmp_path: Path) -> None:
    original_tags = _presets(appearances={"사용자": ["keep"]})
    _write_json(tmp_path / "asset_data" / "tags.json", original_tags)
    _write_json(tmp_path / "asset_data" / "hidden_tags.json", {})
    _write_json(
        tmp_path / "asset_data" / "bot.json",
        {
            "bots": [],
            "positive_whitelist": [],
            "positive_blacklist": [],
            "system_prompt_presets": {},
        },
    )
    payload = {
        PATCH_PRESETS_MEMBER: _json_bytes(_presets()),
        PATCH_BOT_MEMBER: _json_bytes(_bot_payload()),
    }
    patch_path = tmp_path / "corrupt.soyapatch"
    _write_patch(
        patch_path,
        payload,
        corrupt_digest_for=PATCH_PRESETS_MEMBER,
    )

    with pytest.raises(PatchImportError, match="SHA-256"):
        PatchImporter(tmp_path).import_package(patch_path)

    assert json.loads(
        (tmp_path / "asset_data" / "tags.json").read_text(encoding="utf-8")
    ) == original_tags
    assert not (tmp_path / "asset_data" / "backup").exists()


def test_unsafe_archive_path_is_rejected(tmp_path: Path) -> None:
    _write_json(tmp_path / "asset_data" / "tags.json", _presets())
    payload = {
        PATCH_PRESETS_MEMBER: _json_bytes(_presets()),
        PATCH_BOT_MEMBER: _json_bytes(_bot_payload()),
        "payload/asset/../../outside.txt": b"unsafe",
    }
    patch_path = tmp_path / "unsafe.soyapatch"
    _write_patch(patch_path, payload)

    with pytest.raises(PatchImportError, match="안전하지"):
        PatchImporter(tmp_path).import_package(patch_path)

    assert not (tmp_path.parent / "outside.txt").exists()


def test_frontend_exposes_chunked_patch_import() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    server = (
        Path(__file__).resolve().parents[1] / "server.py"
    ).read_text(encoding="utf-8")

    assert 'id="comfy-installer-patch-btn"' in frontend
    assert ">패치파일 가져오기</button>" in frontend
    assert "async function comfyInstallerImportPatch(input)" in frontend
    assert "/api/patch-import/upload/start" in frontend
    assert "/api/patch-import/upload/chunk" in frontend
    assert "/api/patch-import/upload/complete" in frontend
    assert "/api/patch-import/upload/abort" in frontend
    assert "register_patch_import_routes(" in server


@pytest.mark.asyncio
async def test_chunked_upload_api_imports_patch_and_cleans_temp_file(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "asset_data" / "tags.json", _presets())
    _write_json(tmp_path / "asset_data" / "hidden_tags.json", {})
    _write_json(
        tmp_path / "asset_data" / "bot.json",
        {
            "bots": [],
            "positive_whitelist": [],
            "positive_blacklist": [],
            "system_prompt_presets": {},
        },
    )
    patch_path = tmp_path / "api-source.soyapatch"
    _write_patch(
        patch_path,
        {
            PATCH_PRESETS_MEMBER: _json_bytes(
                _presets(appearances={"API 외모": ["api"]})
            ),
            PATCH_BOT_MEMBER: _json_bytes(_bot_payload()),
        },
    )
    patch_bytes = patch_path.read_bytes()
    reload_calls = []
    app = web.Application(client_max_size=200 * 1024**2)
    register_patch_import_routes(
        app,
        project_root=tmp_path,
        reload_asset_tags=lambda: reload_calls.append(True),
        installer_status=lambda: {"state": "idle"},
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        start_response = await client.post(
            "/api/patch-import/upload/start",
            json={
                "filename": "tutorial.soyapatch",
                "file_size": len(patch_bytes),
            },
        )
        assert start_response.status == 200
        started = await start_response.json()
        upload_id = started["upload_id"]

        midpoint = len(patch_bytes) // 2
        first_response = await client.post(
            f"/api/patch-import/upload/chunk?upload_id={upload_id}&offset=0",
            data=patch_bytes[:midpoint],
            headers={"Content-Type": "application/octet-stream"},
        )
        assert first_response.status == 200
        second_response = await client.post(
            f"/api/patch-import/upload/chunk?upload_id={upload_id}&offset={midpoint}",
            data=patch_bytes[midpoint:],
            headers={"Content-Type": "application/octet-stream"},
        )
        assert second_response.status == 200

        complete_response = await client.post(
            "/api/patch-import/upload/complete",
            json={"upload_id": upload_id},
        )
        assert complete_response.status == 200
        completed = await complete_response.json()
        assert completed["success"] is True
        assert completed["summary"]["presets"]["added"] == 1
        assert reload_calls == [True]
        assert not list(
            (tmp_path / ".work" / "patch-import" / "uploads").glob("*")
        )
    finally:
        await client.close()
