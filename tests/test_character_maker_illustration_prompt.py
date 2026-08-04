import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _profile_tags():
    return {
        "quality_presets": {
            "sdxl-q": ["sdxl_quality"],
            "anima-q": ["anima_quality"],
        },
        "artist_presets": {
            "sdxl-a": ["sdxl_artist"],
            "anima-a": ["anima_artist"],
        },
        "negative_presets": {
            "sdxl-n": ["sdxl_bad"],
            "anima-n": ["anima_bad"],
        },
        "character_negative_presets": {
            "char-n": ["wrong_hair"],
        },
    }


def _session(*, generation_workflow="illustration"):
    settings = {
        "generation_workflow": generation_workflow,
        "quality_preset": "sdxl-q",
        "artist_preset": "sdxl-a",
        "negative_preset": "sdxl-n",
        "anima_quality_preset": "anima-q",
        "anima_artist_preset": "anima-a",
        "anima_negative_preset": "anima-n",
        "character_negative_preset": "char-n",
        "img_w": 704,
        "img_h": 1024,
        "seed": 123,
        "lora_enabled": True,
        "lora_list": [
            {
                "name": "hero",
                "lora_path": "hero.safetensors",
                "strength": 0.7,
                "trigger": "hero_trigger",
                "BASE": "anima",
                "source": "asset",
            }
        ],
        "style_lora_enabled": False,
        "style_lora_list": [],
        "face_lora_enabled": False,
        "face_lora_list": [],
        "face_lora_upscale_size": "",
        "face_tags": "detailed face",
        "eye_tags": "blue eyes",
        "hrf_sdxl": False,
        "hrf_anima": False,
        "hrf_size": 2.0,
        "hrf_restore_size": True,
        "sdxl_fd_enabled": False,
        "sdxl_hd_enabled": False,
        "sdxl_ed_enabled": False,
        "anima_fd_enabled": False,
        "anima_hd_enabled": False,
        "anima_ed_enabled": False,
        "face_crop_top": 2.5,
        "face_crop_bottom": 1.0,
    }
    fields = {
        "appearance": ["silver_hair", "blue_eyes"],
        "outfit": ["black_coat"],
        "expression": ["gentle_smile"],
        "composition": ["portrait"],
    }
    return {
        "settings": settings,
        "fields": copy.deepcopy(fields),
        "llm_fields": copy.deepcopy(fields),
        "natural_language": "soft rim light",
        "llm_natural_language": "dramatic rim light",
        "editable_preset_tags": {},
        "editable_preset_enabled": {},
    }


def test_character_maker_v3_builder_uses_structured_fields_without_bot_detection(
    monkeypatch,
):
    import server

    monkeypatch.setattr(server.asset_mode, "_tags", _profile_tags())
    monkeypatch.setitem(server.app_config, "illustration_workflow_type", "v3")

    built = server._build_character_maker_illustration_prompt(
        _session(),
        source="user",
    )

    assert built["provider"] == "comfy"
    assert built["prompt_format"] == "v3"
    assert "[ANIMA_ALL]" in built["positive"]
    assert "portrait, soft rim light" in built["positive"]
    assert "hero_trigger, silver_hair, blue_eyes, black_coat, gentle_smile" in built["positive"]
    assert "[LORA_ACTIVATE]\ntrue" in built["positive"]
    assert "SOYA_CHAR_LORA\\\\hero.safetensors" in built["positive"]
    assert "wrong_hair" in built["negative"]
    assert built["positive"].endswith("[END]")


@pytest.mark.parametrize(
    ("workflow_type", "provider", "prompt_format", "required", "forbidden"),
    [
        ("v1", "comfy", "v1", "[Positive]", "[LORA_ACTIVATE]"),
        ("chansub", "chansub", "chansub", "silver_hair", "[END]"),
    ],
)
def test_character_maker_builder_selects_explicit_illustration_profile(
    monkeypatch,
    workflow_type,
    provider,
    prompt_format,
    required,
    forbidden,
):
    import server

    monkeypatch.setattr(server.asset_mode, "_tags", _profile_tags())
    monkeypatch.setitem(server.app_config, "illustration_workflow_type", workflow_type)
    monkeypatch.setitem(server.app_config, "chansub_workflow_type", "anima")

    built = server._build_character_maker_illustration_prompt(
        _session(),
        source="user",
    )

    assert built["provider"] == provider
    assert built["prompt_format"] == prompt_format
    assert required in built["positive"]
    assert forbidden not in built["positive"]
