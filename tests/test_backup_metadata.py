"""save_backup 이 기록하는 이미지 크기·raw_extension 메타데이터와 영상화 참조
옵션 조회의 메타데이터 우선 경로(A)를 검증한다."""

import json
import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def _png_bytes(size, color="white"):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _patch_backup_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "current_original_workflow", {})
    monkeypatch.setattr(server, "current_api_workflow", {})
    monkeypatch.setattr(server, "current_conversion_info", {})
    monkeypatch.setattr(server, "cleanup_backups", lambda: None)
    monkeypatch.setattr(server, "_invalidate_backup_filter_cache", lambda: None)

    async def ignore_notify(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "notify_frontend", ignore_notify)


@pytest.mark.asyncio
async def test_save_backup_records_size_and_missing_raw_for_clean_backup(
    tmp_path, monkeypatch
):
    # 대사 합성 없이 저장한 백업: 메인 이미지가 곧 원본 → raw_extension=""
    _patch_backup_env(monkeypatch, tmp_path)

    backup_name, _ = await server.save_backup(
        _png_bytes((31, 17)), "meta-clean", "positive", "negative"
    )

    info = json.loads(
        (tmp_path / f"{backup_name}_info.json").read_text(encoding="utf-8")
    )
    assert info["image_width"] == 31
    assert info["image_height"] == 17
    assert info["raw_extension"] == ""
    raw_dir = tmp_path / "_raw"
    assert not raw_dir.exists() or not list(raw_dir.iterdir())


@pytest.mark.asyncio
async def test_save_backup_records_raw_extension_for_composited_backup(
    tmp_path, monkeypatch
):
    # 대사 합성이 적용된 백업: _raw 보존 + speak_text 기록 + raw_extension=".webp"
    _patch_backup_env(monkeypatch, tmp_path)

    def fake_compose(image_bytes, speak_text, settings, bot_name):
        return image_bytes  # 합성 결과로 원본 bytes 그대로 반환(테스트 단순화)

    monkeypatch.setattr("modes.postprocess.compose_postprocess", fake_compose)

    backup_name, _ = await server.save_backup(
        _png_bytes((24, 24)),
        "meta-composited",
        "positive",
        "negative",
        postprocess_settings={"placement": "bottom"},
        speak_text='hero: "대사"',
    )

    info = json.loads(
        (tmp_path / f"{backup_name}_info.json").read_text(encoding="utf-8")
    )
    assert info["raw_extension"] == ".webp"
    assert (tmp_path / "_raw" / f"{backup_name}.webp").is_file()
    assert info["image_width"] == 24
    assert info["image_height"] == 24
    assert info["speak_text"] == 'hero: "대사"'


@pytest.mark.asyncio
async def test_save_backup_records_visual_profile_state_for_followup_edits(
    tmp_path, monkeypatch
):
    _patch_backup_env(monkeypatch, tmp_path)

    backup_name, _ = await server.save_backup(
        _png_bytes((20, 20)),
        "meta-visual-profile",
        "positive",
        "negative",
        illustration_visual_states={
            "Hero": {
                "visual_profile_id": "awakened",
                "profile_embedding": True,
            },
        },
        llm_final_result={
            "raw_positive": "[NAME]\nHero",
            "character_names": ["Hero"],
            "visual_states": {
                "Hero": {"visual_profile_id": "awakened"},
            },
        },
    )

    info = json.loads(
        (tmp_path / f"{backup_name}_info.json").read_text(encoding="utf-8")
    )
    assert info["illustration_visual_states"] == {
        "Hero": {
            "visual_profile_id": "awakened",
            "profile_embedding": True,
        },
    }
    assert info["llm_final_result"]["visual_states"] == {
        "Hero": {"visual_profile_id": "awakened"},
    }
    assert server._read_backup_visual_states(backup_name, "positive") == {
        "Hero": {
            "visual_profile_id": "awakened",
            "profile_embedding": True,
        },
    }


def test_legacy_backup_recovers_profile_state_from_embedding_control_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    cache_payload = {
        "list": [{
            "CHAR": "Hero",
            "emb_path": (
                "soya_bot/Legacy Bot/Hero/"
                "_visual_profiles/awakened/cache.pt"
            ),
        }],
    }
    source_positive = "[CACHE_PATH]\n" + json.dumps(cache_payload)

    assert server._read_backup_visual_states(
        "legacy-profile-backup",
        source_positive,
    ) == {
        "Hero": {
            "visual_profile_id": "awakened",
            "profile_embedding": True,
        },
    }


@pytest.mark.asyncio
async def test_reference_options_pages_without_resolving_legacy_dimensions(
    tmp_path, monkeypatch
):
    # 탐색 목록은 메타데이터가 없는 구 백업도 _resolve_reference/PIL 열기 없이
    # 후보로 등록한다. 크기는 실제로 선택된 뒤 브라우저 이미지 로드로 채운다.
    backup_dir = tmp_path
    recorded = "20260816_120000_meta0001"
    Image.new("RGB", (40, 30), "red").save(
        backup_dir / f"{recorded}.webp", format="WEBP"
    )
    (backup_dir / f"{recorded}_info.json").write_text(
        json.dumps({"raw_extension": "", "image_width": 40, "image_height": 30}),
        encoding="utf-8",
    )
    legacy = "20260701_090000_legacy01"
    Image.new("RGB", (20, 20), "blue").save(
        backup_dir / f"{legacy}.webp", format="WEBP"
    )
    (backup_dir / f"{legacy}_info.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server, "get_backup_base_dir", lambda: str(backup_dir))
    monkeypatch.setattr(
        server.video_mode, "get_backup_dir", lambda: str(backup_dir)
    )
    probed = []
    def spy_resolve(reference, *, raw=True):
        probed.append(reference.get("name"))
        raise AssertionError("탐색 목록에서 이미지 참조를 해석하면 안 됩니다")

    monkeypatch.setattr(server.video_mode, "_resolve_reference", spy_resolve)

    request = SimpleNamespace(
        query={"tab": "illustration", "offset": "0", "limit": "40"}
    )
    response = await server.handle_api_video_reference_options(request=request)
    payload = json.loads(response.text)

    assert payload["success"] is True
    options = {option["name"]: option for option in payload["options"]}
    assert options[recorded]["source_width"] == 40
    assert options[recorded]["source_height"] == 30
    assert options[recorded]["is_animated"] is False
    assert options[legacy]["source_width"] == 0
    assert options[legacy]["source_height"] == 0
    assert payload["has_more"] is False
    assert probed == []


@pytest.mark.asyncio
async def test_reference_options_direct_raw_lookup_does_not_scan_all_backups(
    tmp_path, monkeypatch
):
    backup_dir = tmp_path
    name = "20260820_063530_4b54ac63"
    Image.new("RGB", (32, 24), "green").save(
        backup_dir / f"{name}.webp", format="WEBP"
    )
    raw_dir = backup_dir / "_raw"
    raw_dir.mkdir()
    Image.new("RGB", (32, 20), "yellow").save(
        raw_dir / f"{name}.webp", format="WEBP"
    )
    (backup_dir / f"{name}_info.json").write_text(
        json.dumps(
            {
                "raw_extension": ".webp",
                "image_width": 32,
                "image_height": 20,
                "speak_text": 'hero: "대사"',
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "get_backup_base_dir", lambda: str(backup_dir))

    def reject_glob(*args, **kwargs):
        raise AssertionError("단건 현재 카드 조회에서 전체 백업 glob을 하면 안 됩니다")

    monkeypatch.setattr(server.glob, "glob", reject_glob)
    request = SimpleNamespace(query={"tab": "illustration", "name": name})
    response = await server.handle_api_video_reference_options(request=request)
    payload = json.loads(response.text)

    assert payload["success"] is True
    assert payload["has_more"] is False
    assert payload["options"][0]["reference"] == {"kind": "backup", "name": name}
    assert payload["options"][0]["image_url"] == f"/api/backup_raw/{name}"


@pytest.mark.asyncio
async def test_reference_options_asset_tab_is_paginated(tmp_path, monkeypatch):
    asset_module = importlib.import_module("modes.asset_mode")
    monkeypatch.setattr(asset_module, "ASSET_DIR", str(tmp_path))
    monkeypatch.setattr(server.asset_mode, "list_characters", lambda: ["hero"])
    expression_dir = tmp_path / "hero" / "casual" / "smile"
    expression_dir.mkdir(parents=True)
    for index, color in enumerate(("red", "blue"), start=1):
        Image.new("RGB", (12, 12), color).save(
            expression_dir / f"image-{index}.webp", format="WEBP"
        )

    request = SimpleNamespace(query={"tab": "asset", "offset": "0", "limit": "1"})
    response = await server.handle_api_video_reference_options(request=request)
    payload = json.loads(response.text)

    assert payload["success"] is True
    assert payload["tab"] == "asset"
    assert len(payload["options"]) == 1
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1
    reference = payload["options"][0]["reference"]
    assert reference["kind"] == "asset"
    assert reference["character"] == "hero"
    assert reference["outfit"] == "casual"
    assert reference["expression"] == "smile"
