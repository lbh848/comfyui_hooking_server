import importlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from modes.asset_mode import AssetMode


asset_mode_module = importlib.import_module("modes.asset_mode")


def _animated_webp_bytes() -> bytes:
    output = io.BytesIO()
    frames = [
        Image.new("RGB", (24, 24), "red"),
        Image.new("RGB", (24, 24), "blue"),
    ]
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        lossless=True,
    )
    return output.getvalue()


def _static_source(root: Path) -> tuple[Path, dict]:
    expression_dir = root / "alice" / "uniform" / "smile"
    expression_dir.mkdir(parents=True)
    source_path = expression_dir / "source.webp"
    Image.new("RGB", (32, 32), "green").save(source_path, format="WEBP")
    (expression_dir / "source_prompt.json").write_text(
        json.dumps(
            {
                "positive": "1girl, green coat",
                "negative": "",
                "character": "alice",
                "outfit": "uniform",
                "expression": "smile",
            }
        ),
        encoding="utf-8",
    )
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": "source.webp"}),
        encoding="utf-8",
    )
    reference = {
        "kind": "asset",
        "character": "alice",
        "outfit": "uniform",
        "expression": "smile",
        "filename": "source.webp",
    }
    return source_path, reference


def _mapping(export_format="png", export_quality=12):
    return {
        "export_name": "hero",
        "outfit_mapping": {"uniform": "school"},
        "expression_mapping": {"smile": "happy"},
        "export_format": export_format,
        "export_quality": export_quality,
        "naming_order": ["character", "outfit", "expression"],
        "naming_enabled": {
            "character": True,
            "outfit": True,
            "expression": True,
        },
    }


def _make_animated_representative(asset_root: Path) -> tuple[Path, dict]:
    expression_dir = asset_root / "alice" / "uniform" / "smile"
    expression_dir.mkdir(parents=True, exist_ok=True)
    source_path = expression_dir / "representative.webp"
    source_path.write_bytes(_animated_webp_bytes())
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": source_path.name}),
        encoding="utf-8",
    )
    return source_path, {
        "kind": "asset",
        "character": "alice",
        "outfit": "uniform",
        "expression": "smile",
        "filename": source_path.name,
    }


def test_asset_video_commit_adds_new_file_without_touching_source_or_representative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    source_before = source_path.read_bytes()

    staging = tmp_path / "staging"
    staging.mkdir()
    main_path = staging / "main.webp"
    raw_path = staging / "raw.webp"
    animated = _animated_webp_bytes()
    main_path.write_bytes(animated)
    raw_path.write_bytes(animated)

    result = mode.commit_video_result(
        reference,
        str(main_path),
        str(raw_path),
        ".webp",
        {
            "base_name": "20260813_120000_newvideo",
            "positive": "official H3 prompt",
            "instruction": "move gently",
            "instruction_source": "llm",
            "auto_instruction": True,
            "visual_context": "visual_context:\nAlice stands by a window.",
            "visual_context_source": "prompt",
            "llm_trace": ["video-trace-1"],
            "mode": "i2v",
            "last_ref": {},
            "duration": 5,
            "fps": 24,
            "video_seed": 123,
            "output_width": 512,
            "output_height": 512,
            "upscale_enabled": True,
            "upscale_scale": 2,
            "upscale_model": "realesr-animevideov3",
            "output_format": "webp",
        },
    )

    expression_dir = source_path.parent
    output_path = expression_dir / result["filename"]
    assert source_path.read_bytes() == source_before
    assert output_path.read_bytes() == animated
    assert (expression_dir / "_raw" / result["filename"]).read_bytes() == animated
    assert json.loads(
        (expression_dir / "_representative.json").read_text(encoding="utf-8")
    ) == {"filename": "source.webp"}

    listing = mode.list_images("alice", "uniform", "smile")
    output_record = next(
        item for item in listing["images"] if item["filename"] == result["filename"]
    )
    assert output_record["is_animated"] is True
    assert output_record["is_video_animation"] is True
    assert output_record["is_representative"] is False
    assert output_record["video_instruction"] == "move gently"
    assert output_record["video_instruction_source"] == "llm"
    assert output_record["video_auto_instruction"] is True
    assert output_record["video_visual_context"] == "visual_context:\nAlice stands by a window."
    assert output_record["video_visual_context_source"] == "prompt"
    prompt = json.loads(
        (expression_dir / "20260813_120000_newvideo_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    assert prompt["video_source_filename"] == "source.webp"
    assert prompt["video_mode"] == "i2v"
    assert prompt["video_instruction"] == "move gently"
    assert prompt["video_instruction_source"] == "llm"
    assert prompt["video_auto_instruction"] is True
    assert prompt["video_visual_context"] == "visual_context:\nAlice stands by a window."
    assert prompt["video_visual_context_source"] == "prompt"
    assert prompt["llm_trace"] == ["video-trace-1"]


def test_export_video_session_keeps_one_temporary_result_and_zip_uses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    session_root = tmp_path / "asset_data" / "export_video_sessions"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(
        asset_mode_module,
        "EXPORT_VIDEO_SESSION_DIR",
        str(session_root),
    )
    mode = AssetMode()
    source_path, reference = _make_animated_representative(asset_root)
    mapping = _mapping(export_format="png", export_quality=40)

    session = mode.create_export_video_session(
        "alice",
        ["uniform"],
        ["smile"],
        mapping,
    )
    assert len(session["items"]) == 1
    item = session["items"][0]
    assert item["status"] == "idle"

    job = mode.prepare_export_video_jobs(
        session["session_id"],
        [item["slot_id"]],
    )[0]
    mode.mark_export_video_jobs_queued(
        session["session_id"],
        {item["slot_id"]: "queue-1"},
    )
    mode.update_export_video_job_progress(
        session["session_id"],
        item["slot_id"],
        job["revision"],
        {"percentage": 50, "phase": "video_upscale"},
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    main_path = staging / "main.webp"
    raw_path = staging / "raw.webp"
    first_result = b"first-temporary-animation"
    main_path.write_bytes(first_result)
    raw_path.write_bytes(b"unused-raw")
    mode.commit_export_video_result(
        session["session_id"],
        item["slot_id"],
        job["revision"],
        reference,
        str(main_path),
        str(raw_path),
        ".webp",
        {"fps": 24, "output_width": 48, "output_height": 48},
    )

    completed = mode.get_export_video_session(session["session_id"])["items"][0]
    assert completed["status"] == "completed"
    result_path = Path(
        mode.get_export_video_result_path(
            session["session_id"],
            item["slot_id"],
        )
    )
    assert result_path.read_bytes() == first_result
    assert list(result_path.parent.iterdir()) == [result_path]
    assert list(source_path.parent.glob("*_prompt.json")) == []

    retry = mode.prepare_export_video_jobs(
        session["session_id"],
        [item["slot_id"]],
    )[0]
    retry_main = staging / "retry.webp"
    retry_raw = staging / "retry_raw.webp"
    second_result = b"replacement-temporary-animation"
    retry_main.write_bytes(second_result)
    retry_raw.write_bytes(b"unused-retry-raw")
    mode.commit_export_video_result(
        session["session_id"],
        item["slot_id"],
        retry["revision"],
        reference,
        str(retry_main),
        str(retry_raw),
        ".webp",
        {"fps": 30, "output_width": 72, "output_height": 72},
    )
    assert result_path.read_bytes() == second_result
    assert list(result_path.parent.iterdir()) == [result_path]

    failed_retry = mode.prepare_export_video_jobs(
        session["session_id"],
        [item["slot_id"]],
    )[0]
    mode.mark_export_video_job_failed(
        session["session_id"],
        item["slot_id"],
        failed_retry["revision"],
        "encoder failed",
    )
    preserved = mode.get_export_video_session(session["session_id"])["items"][0]
    assert preserved["status"] == "completed"
    assert "기존 임시 결과를 유지" in preserved["error"]
    assert result_path.read_bytes() == second_result

    manifest, plan, overrides = mode.build_export_video_session_plan(
        session["session_id"]
    )
    archive = mode.export_character_zip(
        manifest["character"],
        export_plan=plan,
        video_overrides=overrides,
    )
    assert archive is not None
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == ["hero_school_happy.webp"]
        assert zipped.read("hero_school_happy.webp") == second_result

    mode.delete_export_video_session(session["session_id"])
    assert not (session_root / session["session_id"]).exists()
    assert source_path.exists()


def test_representative_animation_is_playable_and_exported_verbatim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    expression_dir = source_path.parent
    animation_path = expression_dir / "animation.webp"
    animation_bytes = _animated_webp_bytes()
    animation_path.write_bytes(animation_bytes)
    (expression_dir / "animation_prompt.json").write_text(
        json.dumps(
            {
                "positive": "official H3 prompt",
                "character": "alice",
                "outfit": "uniform",
                "expression": "smile",
                "is_video_animation": True,
                "video_source_ref": reference,
            }
        ),
        encoding="utf-8",
    )
    mode.set_representative("alice", "uniform", "smile", "animation.webp")

    gallery = mode.list_character_gallery("alice")
    assert gallery[0]["representative"] == "animation.webp"
    assert gallery[0]["representative_is_animated"] is True
    poster_path = Path(mode.get_video_poster_path({**reference, "filename": "animation.webp"}))
    assert poster_path.is_file()
    with Image.open(poster_path) as poster:
        assert getattr(poster, "is_animated", False) is False

    plan = mode.build_character_export_plan(
        "alice",
        ["uniform"],
        ["smile"],
        _mapping(export_format="png", export_quality=12),
    )
    assert plan["success"] is True
    assert plan["files"][0]["filename"] == "hero_school_happy.webp"
    assert plan["files"][0]["is_video_animation"] is True

    archive_buffer = mode.export_character_zip(
        "alice",
        ["uniform"],
        ["smile"],
        mapping_override=_mapping(export_format="png", export_quality=12),
        export_plan=plan,
    )
    with zipfile.ZipFile(archive_buffer) as archive:
        assert archive.namelist() == ["hero_school_happy.webp"]
        assert archive.read("hero_school_happy.webp") == animation_bytes


def test_export_zip_mixes_video_and_static_image(tmp_path: Path, monkeypatch) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()

    # 영상 표정: 애니메이션 webp + is_video_animation 프롬프트 메타데이터
    video_dir = asset_root / "alice" / "uniform" / "smile"
    video_dir.mkdir(parents=True)
    animation_bytes = _animated_webp_bytes()
    (video_dir / "animation.webp").write_bytes(animation_bytes)
    (video_dir / "animation_prompt.json").write_text(
        json.dumps({"is_video_animation": True}),
        encoding="utf-8",
    )
    (video_dir / "_representative.json").write_text(
        json.dumps({"filename": "animation.webp"}),
        encoding="utf-8",
    )

    # 정적 이미지 표정: PNG 원본 → png 포맷 변환 경로 확인용
    static_dir = asset_root / "alice" / "uniform" / "wink"
    static_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "purple").save(static_dir / "static.png", format="PNG")
    (static_dir / "_representative.json").write_text(
        json.dumps({"filename": "static.png"}),
        encoding="utf-8",
    )

    mapping = _mapping(export_format="png", export_quality=40)
    mapping["expression_mapping"] = {"smile": "happy", "wink": "wink"}

    plan = mode.build_character_export_plan(
        "alice", ["uniform"], ["smile", "wink"], mapping
    )
    assert plan["success"] is True
    by_name = {item["filename"]: item for item in plan["files"]}
    assert by_name["hero_school_happy.webp"]["is_video_animation"] is True
    assert by_name["hero_school_wink.png"]["is_video_animation"] is False

    archive = mode.export_character_zip(
        "alice",
        ["uniform"],
        ["smile", "wink"],
        mapping_override=mapping,
        export_plan=plan,
    )
    with zipfile.ZipFile(archive) as zipped:
        assert sorted(zipped.namelist()) == [
            "hero_school_happy.webp",
            "hero_school_wink.png",
        ]
        # 영상은 포맷·품질 설정과 무관하게 원본 그대로
        assert zipped.read("hero_school_happy.webp") == animation_bytes
        # 정적 이미지는 png로 변환되어 저장
        with Image.open(io.BytesIO(zipped.read("hero_school_wink.png"))) as converted:
            assert converted.format == "PNG"


def test_export_zip_mixes_video_override_with_static_image(
    tmp_path: Path, monkeypatch
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()

    # 영상 표정 (대표 원본)
    video_source, _ = _make_animated_representative(asset_root)
    # 정적 이미지 표정
    static_dir = asset_root / "alice" / "uniform" / "wink"
    static_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "purple").save(static_dir / "static.png", format="PNG")
    (static_dir / "_representative.json").write_text(
        json.dumps({"filename": "static.png"}),
        encoding="utf-8",
    )

    mapping = _mapping(export_format="png", export_quality=40)
    mapping["expression_mapping"] = {"smile": "happy", "wink": "wink"}

    # 영상 후처리 임시 결과(override)
    override_dir = tmp_path / "export_video_sessions" / "session-1"
    override_dir.mkdir(parents=True)
    override_path = override_dir / "result.webp"
    override_path.write_bytes(b"postprocessed-animation")

    plan = mode.build_character_export_plan(
        "alice",
        ["uniform"],
        ["smile", "wink"],
        mapping,
        video_overrides={
            mode._export_video_slot_id("uniform", "smile", video_source.name): {
                "path": str(override_path),
                "extension": ".webp",
            }
        },
    )
    assert plan["success"] is True

    archive = mode.export_character_zip(
        "alice",
        ["uniform"],
        ["smile", "wink"],
        mapping_override=mapping,
        export_plan=plan,
        video_overrides={
            mode._export_video_slot_id("uniform", "smile", video_source.name): {
                "path": str(override_path),
                "extension": ".webp",
            }
        },
    )
    with zipfile.ZipFile(archive) as zipped:
        assert sorted(zipped.namelist()) == [
            "hero_school_happy.webp",
            "hero_school_wink.png",
        ]
        # 영상 슬롯은 임시 후처리 결과를 사용
        assert zipped.read("hero_school_happy.webp") == b"postprocessed-animation"
        with Image.open(io.BytesIO(zipped.read("hero_school_wink.png"))) as converted:
            assert converted.format == "PNG"


def test_asset_video_reference_list_includes_static_and_animated_media(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    animation_path = source_path.parent / "animation.webp"
    animation_path.write_bytes(_animated_webp_bytes())
    (source_path.parent / "animation_prompt.json").write_text(
        json.dumps({"is_video_animation": True}),
        encoding="utf-8",
    )

    references = mode.list_video_references("alice")
    by_filename = {
        item["reference"]["filename"]: item for item in references
    }
    assert by_filename["source.webp"]["is_animated"] is False
    assert by_filename["animation.webp"]["is_animated"] is True
    assert mode.resolve_video_reference(reference)["positive"] == "1girl, green coat"
