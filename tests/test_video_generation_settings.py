from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class _ConfigRequest:
    method = "POST"

    def __init__(self, body: object):
        self._body = body

    async def json(self):
        return self._body


def test_video_generation_defaults_normalize_every_persisted_option() -> None:
    normalized = server.normalize_video_generation_defaults(
        {
            "mode": "first_last",
            "workflow_variant": "standard",
            "duration": 12,
            "aspect_ratio": "16:9",
            "quality_level": "high",
            "loop": True,
            "visual_context_source": "prompt",
            "prompt_generation_mode": "best_of_three",
            "translate_instruction_to_english": True,
            "instruction_language": "en",
            "include_dialogue_context": False,
            "allow_camera_motion": False,
            "allow_background_change": True,
            "refine_version": "v2",
            "upscale_model": "none",
            "upscale_scale": 4,
            "output_format": "webp",
        }
    )

    assert normalized == {
        "mode": "first_last",
        "workflow_variant": "standard",
        "duration": 12,
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "loop": True,
        "visual_context_source": "prompt",
        "prompt_generation_mode": "best_of_three",
        "translate_instruction_to_english": True,
        "instruction_language": "en",
        "include_dialogue_context": False,
        "allow_camera_motion": False,
        "allow_background_change": True,
        "refine_version": "v2",
        "upscale_model": "none",
        "upscale_scale": 4,
        "output_format": "webp",
        "encode_quality": 80,
        "sharpen_enabled": False,
        "sharpen_radius": 0.8,
        "sharpen_amount": 0.5,
        "sharpen_threshold": 4,
    }


def test_fast_video_defaults_keep_mp_choice_and_reject_ultrawide() -> None:
    settings = copy.deepcopy(server.DEFAULT_VIDEO_GENERATION_DEFAULTS)
    settings.update(
        {
            "workflow_variant": "fast",
            "aspect_ratio": "16:9",
            "quality_level": "low",
        }
    )

    normalized = server.normalize_video_generation_defaults(settings)

    assert normalized["workflow_variant"] == "fast"
    assert normalized["aspect_ratio"] == "16:9"
    # 고속 MP 단계는 실험적 선택으로 그대로 저장된다.
    assert normalized["quality_level"] == "low"

    omitted = copy.deepcopy(settings)
    del omitted["quality_level"]
    # 화질을 생략한 고속 기본값은 768p(native)를 유지한다.
    assert (
        server.normalize_video_generation_defaults(omitted)["quality_level"]
        == "native"
    )

    settings["aspect_ratio"] = "21:9"
    with pytest.raises(ValueError, match="고속 영상"):
        server.normalize_video_generation_defaults(settings)


def test_ref_defaults_preserve_experimental_values_and_default_when_omitted() -> None:
    settings = copy.deepcopy(server.DEFAULT_VIDEO_GENERATION_DEFAULTS)
    settings.update(
        {
            "mode": "ref2v",
            "workflow_variant": "standard",
            "aspect_ratio": "21:9",
            "quality_level": "low",
        }
    )

    normalized = server.normalize_video_generation_defaults(settings)

    assert normalized["mode"] == "ref2v"
    assert normalized["workflow_variant"] == "standard"
    assert normalized["aspect_ratio"] == "21:9"
    assert normalized["quality_level"] == "low"

    omitted = copy.deepcopy(settings)
    omitted.pop("aspect_ratio")
    omitted.pop("quality_level")
    normalized_omitted = server.normalize_video_generation_defaults(omitted)
    assert normalized_omitted["aspect_ratio"] == "16:9"
    assert normalized_omitted["quality_level"] == "native"
    assert server.DEFAULT_CONFIG["video_workflow_source_paths"]["ref2v"] == ""
    assert server.DEFAULT_CONFIG["video_workflow_source_paths"]["ref2v_fast"] == ""


def test_ref_request_preserves_standard_and_fast_experimental_resolution() -> None:
    assert server.normalize_video_workflow_selection(
        {
            "mode": "ref2v",
            "workflow_variant": "standard",
            "aspect_ratio": "21:9",
            "quality_level": "high",
        },
        log_prefix="TEST:REF2V",
    ) == ("standard", "21:9", "high")

    assert server.normalize_video_workflow_selection(
        {
            "mode": "ref2v",
            "workflow_variant": "fast",
            "aspect_ratio": "21:9",
            "quality_level": "medium",
        },
        log_prefix="TEST:REF2V",
    ) == ("fast", "21:9", "medium")

    assert server.normalize_video_workflow_selection(
        {
            "mode": "ref2v",
            "workflow_variant": "fast",
        },
        log_prefix="TEST:REF2V",
    ) == ("fast", "16:9", "native")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "t2v"),
        ("workflow_variant", "turbo"),
        ("duration", 16),
        ("aspect_ratio", "8:5"),
        ("quality_level", "ultra"),
        ("loop", "true"),
        ("visual_context_source", "metadata"),
        ("prompt_generation_mode", "all"),
        ("translate_instruction_to_english", "true"),
        ("instruction_language", "ja"),
        ("refine_version", "v4"),
        ("upscale_model", "unknown"),
        ("upscale_scale", 8),
        ("output_format", "gif"),
        ("encode_quality", 0),
        ("encode_quality", 101),
        ("encode_quality", "high"),
    ],
)
def test_video_generation_defaults_reject_invalid_values(field: str, value: object) -> None:
    settings = copy.deepcopy(server.DEFAULT_VIDEO_GENERATION_DEFAULTS)
    settings[field] = value

    with pytest.raises(ValueError):
        server.normalize_video_generation_defaults(settings)


def test_video_generation_defaults_accept_japanese_animation_profile() -> None:
    settings = copy.deepcopy(server.DEFAULT_VIDEO_GENERATION_DEFAULTS)
    settings["refine_version"] = "v3"

    normalized = server.normalize_video_generation_defaults(settings)

    assert normalized["refine_version"] == "v3"


def test_load_config_inherits_legacy_video_postprocess_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_postprocess": {
                    "enabled": False,
                    "scale": 4,
                    "model": "anime4k-fast-m",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "CONFIG_FILE", str(config_path))

    loaded = server.load_config()

    assert loaded["video_generation_defaults"]["upscale_model"] == "none"
    assert loaded["video_generation_defaults"]["upscale_scale"] == 4


def test_video_engine_runtime_migration_preserves_existing_config_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    legacy = {
        "comfyui_port": 9001,
        "comfy_task_allocations": {"video_generation": 2},
        "backup_max_count": 321,
        "user_extension_field": {"preserve": [1, 2, 3]},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )
    saved: list[dict] = []
    monkeypatch.setattr(server, "CONFIG_FILE", str(config_path))
    monkeypatch.setattr(
        server,
        "save_config",
        lambda value: saved.append(copy.deepcopy(value)),
    )

    loaded = server.load_config()

    assert saved == [
        {
            **legacy,
            "video_engine_port": 8093,
            "video_engine_project_path": "",
            "video_engine_auto_start": False,
        }
    ]
    assert loaded["video_engine_port"] == 8093
    assert loaded["video_engine_project_path"] == ""
    assert loaded["video_engine_auto_start"] is False
    assert loaded["comfyui_port"] == 9001
    assert loaded["comfy_task_allocations"]["video_generation"] == 2
    assert loaded["backup_max_count"] == 321
    assert loaded["user_extension_field"] == {"preserve": [1, 2, 3]}


@pytest.mark.asyncio
async def test_config_api_persists_video_generation_defaults(monkeypatch) -> None:
    config = copy.deepcopy(server.DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(
        server,
        "save_config",
        lambda value: saved.append(copy.deepcopy(value)),
    )
    defaults = copy.deepcopy(server.DEFAULT_VIDEO_GENERATION_DEFAULTS)
    defaults.update(
        {
            "duration": 9,
            "quality_level": "native",
            "allow_background_change": True,
            "output_format": "webp",
        }
    )

    response = await server.handle_api_config(
        _ConfigRequest(
            {
                "video_generation_defaults": defaults,
                "video_secondary_motion": False,
            }
        )
    )

    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["success"] is True
    assert saved[-1]["video_generation_defaults"] == defaults
    assert saved[-1]["video_secondary_motion"] is False


@pytest.mark.asyncio
async def test_config_api_rejects_invalid_video_generation_defaults(monkeypatch) -> None:
    config = copy.deepcopy(server.DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _ConfigRequest(
            {
                "video_generation_defaults": {
                    **server.DEFAULT_VIDEO_GENERATION_DEFAULTS,
                    "duration": 0,
                }
            }
        )
    )

    assert response.status == 400
    assert saved == []


@pytest.mark.asyncio
async def test_config_api_persists_video_engine_runtime_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = copy.deepcopy(server.DEFAULT_CONFIG)
    saved: list[dict] = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(
        server,
        "save_config",
        lambda value: saved.append(copy.deepcopy(value)),
    )

    response = await server.handle_api_config(
        _ConfigRequest(
            {
                "video_engine_port": 8094,
                "video_engine_project_path": str(tmp_path),
                "video_engine_auto_start": True,
            }
        )
    )

    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["success"] is True
    assert saved[-1]["video_engine_port"] == 8094
    assert saved[-1]["video_engine_project_path"] == str(tmp_path)
    assert saved[-1]["video_engine_auto_start"] is True


@pytest.mark.asyncio
async def test_config_api_rejects_non_boolean_video_engine_autostart(
    monkeypatch,
) -> None:
    config = copy.deepcopy(server.DEFAULT_CONFIG)
    saved: list[dict] = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _ConfigRequest({"video_engine_auto_start": "yes"})
    )

    assert response.status == 400
    assert saved == []


@pytest.mark.asyncio
async def test_config_api_rejects_engine_identity_change_while_managed_process_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = copy.deepcopy(server.DEFAULT_CONFIG)
    config["video_engine_project_path"] = str(tmp_path)
    config["video_engine_port"] = 8093
    saved: list[dict] = []
    manager = server.VideoEngineRuntimeManager()
    monkeypatch.setattr(
        manager,
        "running_identity",
        lambda: (str(tmp_path), 8093),
    )
    monkeypatch.setattr(server, "video_engine_runtime_manager", manager)
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _ConfigRequest({"video_engine_port": 8094})
    )

    assert response.status == 409
    assert saved == []
