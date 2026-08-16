"""save_backup 이 기록하는 이미지 크기·raw_extension 메타데이터와 영상화 참조
옵션 조회의 메타데이터 우선 경로(A)를 검증한다."""

import json
import sys
from io import BytesIO
from pathlib import Path

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
async def test_reference_options_prefers_recorded_metadata_without_probe(
    tmp_path, monkeypatch
):
    # 메타데이터(raw_extension·크기)가 기록된 백업은 _resolve_reference/PIL 열기
    # 없이 후보로 등록된다. raw_extension 키가 없는 구 백업만 기존 탐색을 거친다.
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
    real_resolve = server.video_mode._resolve_reference

    def spy_resolve(reference, *, raw=True):
        probed.append(reference.get("name"))
        return real_resolve(reference, raw=raw)

    monkeypatch.setattr(server.video_mode, "_resolve_reference", spy_resolve)

    response = await server.handle_api_video_reference_options(request=None)
    payload = json.loads(response.text)

    assert payload["success"] is True
    options = {option["name"]: option for option in payload["options"]}
    assert options[recorded]["source_width"] == 40
    assert options[recorded]["source_height"] == 30
    assert options[recorded]["is_animated"] is False
    assert options[legacy]["source_width"] == 20
    assert options[legacy]["source_height"] == 20
    # 메타데이터 경로의 백업은 참조 해석을 거치지 않고, 구 백업만 탐색했다.
    assert probed == [legacy]
