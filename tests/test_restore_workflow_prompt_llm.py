import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
import workflow_profiles
from customprompt import restore_workflow_prompt_llm as restore_llm
from modes import postprocess


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _character_context(name, gender, appearance, outfit):
    return {
        "name": name,
        "gender_tag": gender,
        "appearance_tags": appearance,
        "outfit_tags": outfit,
    }


def test_scene_payload_parses_two_characters_and_restores_required_tags():
    contexts = [
        _character_context("Alice", "1girl", ["blue eyes"], ["red dress"]),
        _character_context("Bob", "1boy", ["black hair"], ["school uniform"]),
    ]
    result = json.dumps({
        "setup": "wide shot, rooftop, sunset",
        "characters": [
            {
                "name": "Alice",
                "tags": "smile, waving",
                "position": "on the left",
            },
            {
                "name": "Bob",
                "tags": "looking at alice, standing",
                "position": "on the right",
            },
        ],
        "supplement": "wind, rim light",
        "dialogue": [
            {
                "speaker": "Alice",
                "type": "speech",
                "text": "오늘도 늦었네.",
            }
        ],
    }, ensure_ascii=False)

    parsed = restore_llm._parse_scene_payload(result, contexts, True)

    assert parsed["characters"][0]["positive"].startswith(
        "1girl, blue eyes, red dress"
    )
    assert parsed["characters"][1]["positive"].startswith(
        "1boy, black hair, school uniform"
    )
    assert parsed["dialogue"][0]["speaker"] == "Alice"


def test_scene_payload_rejects_extra_or_reordered_character():
    contexts = [
        _character_context("Alice", "1girl", [], []),
        _character_context("Bob", "1boy", [], []),
    ]
    result = json.dumps({
        "setup": "classroom",
        "characters": [
            {"name": "Bob", "tags": "1boy, standing", "position": "left"},
            {"name": "Alice", "tags": "1girl, sitting", "position": "right"},
        ],
        "supplement": "",
        "dialogue": [],
    })

    with pytest.raises(ValueError, match="순서/이름 불일치"):
        restore_llm._parse_scene_payload(result, contexts, False)


@pytest.mark.asyncio
async def test_run_builds_two_character_v3_sections_and_llm_speak(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.json"
    bot_path = tmp_path / "bot.json"
    config_path.write_text(
        json.dumps({"bot_selected": "Example Bot"}, ensure_ascii=False),
        encoding="utf-8",
    )
    bot_path.write_text(
        json.dumps({
            "bots": [{
                "name": "Example Bot",
                "characters": [
                    {"name": "Alice", "gender_tag": "1girl"},
                    {"name": "Bob", "gender_tag": "1boy"},
                ],
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(restore_llm, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(restore_llm, "BOT_DATA_PATH", str(bot_path))
    monkeypatch.setattr(
        restore_llm,
        "_get_lb_extra_entry",
        lambda _bot, name: {
            "appearance": [{"tag": "blue eyes" if name == "Alice" else "black hair"}],
            "outfit": [{"tag": "red dress" if name == "Alice" else "school uniform"}],
        },
    )

    async def fake_notify(_event_type, _data=None):
        return None

    monkeypatch.setattr(restore_llm, "_notify_llm_widget", fake_notify)

    response_text = json.dumps({
        "setup": "wide shot, rooftop, sunset",
        "characters": [
            {
                "name": "Alice",
                "tags": "smile, waving",
                "position": "left",
            },
            {
                "name": "Bob",
                "tags": "standing, looking at alice",
                "position": "right",
            },
        ],
        "supplement": "wind, rim light",
        "dialogue": [
            {"speaker": "Bob", "type": "speech", "text": "기다렸어?"}
        ],
    }, ensure_ascii=False)

    async def fake_call(_task_key, _messages, **kwargs):
        assert kwargs["json_mode"] is True
        assert kwargs["result_validator"](response_text) == (True, "")
        return response_text

    monkeypatch.setattr("modes.llm_service.callLLMTask", fake_call)
    monkeypatch.setattr(
        "modes.lighbd_service._log_lighbd_history",
        lambda _entry: None,
    )

    result = await restore_llm.run(
        char_names=["Alice", "Bob"],
        situation="옥상에서 만나는 장면",
        postprocess_test=True,
        speak_text="",
        postprocess_mode="vn",
    )

    assert "[SPEAK]\nBob: \"기다렸어?\"" in result["positive"]
    assert "[Name]\nAlice, Bob" in result["positive"]
    assert "1girl, blue eyes, red dress" in result["positive"]
    assert "1boy, black hair, school uniform" in result["positive"]
    assert [item["name"] for item in result["characters"]] == ["Alice", "Bob"]


def test_manual_speak_plain_text_gets_single_character_name():
    assert server._normalize_restore_manual_speak_text(
        "안녕하세요.",
        ["Alice"],
    ) == 'Alice: "안녕하세요."'


def test_manual_speak_requires_speaker_names_for_two_characters():
    with pytest.raises(ValueError, match="2인 후처리 텍스트"):
        server._normalize_restore_manual_speak_text(
            "안녕하세요.",
            ["Alice", "Bob"],
        )


@pytest.mark.asyncio
async def test_restore_multi_char_context_uses_existing_layout_pipeline(monkeypatch):
    layout = {
        "background_prompt": "wide shot, rooftop, sunset",
        "composition_prompt": "two distinct people, facing each other",
        "character_order": ["Bob", "Alice"],
        "regions": [
            {
                "name": "Bob",
                "character_prompt": "1boy, black hair, standing",
                "x": 0.05,
                "y": 0.05,
                "width": 0.45,
                "height": 0.9,
                "channel": "R",
            },
            {
                "name": "Alice",
                "character_prompt": "1girl, blue eyes, waving",
                "x": 0.5,
                "y": 0.05,
                "width": 0.45,
                "height": 0.9,
                "channel": "G",
            },
        ],
    }

    monkeypatch.setattr(
        server.illustration_context_pipeline,
        "load_prompt_files",
        lambda: {"multi_char_mask": "layout prompt"},
    )

    async def fake_calculate(descriptors, prompt):
        assert prompt == "layout prompt"
        descriptor = descriptors[0]
        by_name = {
            character["name"]: character
            for character in descriptor["characters"]
        }
        descriptor["characters"] = [by_name["Bob"], by_name["Alice"]]
        descriptor["multi_char_layout"] = layout

    monkeypatch.setattr(
        server.illustration_context_pipeline,
        "calculate_multi_char_layouts",
        fake_calculate,
    )

    context = await server._build_restore_manual_multi_char_context({
        "setup": "wide shot, rooftop",
        "supplement": "sunset",
        "speak_text": "",
        "characters": [
            {"name": "Alice", "positive": "1girl, blue eyes", "position": "right"},
            {"name": "Bob", "positive": "1boy, black hair", "position": "left"},
        ],
    })

    assert context["enable"] is True
    assert context["character_order"] == ["Bob", "Alice"]
    assert [item["name"] for item in context["characters"]] == ["Bob", "Alice"]
    assert context["mask_location"] == "region_mask"


def test_forced_postprocess_ignores_master_and_bot_enabled_toggles(monkeypatch):
    stored = postprocess._default_vn()
    stored["enabled"] = False
    monkeypatch.setattr(postprocess, "_load_bot_vn", lambda _bot: stored)

    assert postprocess.get_vn_settings(
        {"postprocess_enabled": False},
        bot_name="Example",
    ) is None
    forced = postprocess.get_vn_settings(
        {"postprocess_enabled": False},
        bot_name="Example",
        force=True,
    )
    assert forced is not None
    assert forced["placement"] == stored["placement"]


def _write_fake_restore_module(tmp_path):
    module_path = tmp_path / "restore_workflow_prompt_llm.py"
    module_path.write_text(
        """
async def run(
    char_names=None,
    situation=None,
    postprocess_test=False,
    speak_text=None,
    postprocess_mode="vn",
):
    names = list(char_names or [])
    return {
        "positive": (
            "[Name]\\n" + ", ".join(names)
            + "\\n[SETUP]\\nwide shot, rooftop"
            + "\\n[CHAR]\\n1girl, blue eyes\\n\\n1boy, black hair"
            + "\\n[SUPPLEMENT]\\nsunset"
        ),
        "negative": "lowres",
        "setup": "wide shot, rooftop",
        "supplement": "sunset",
        "characters": [
            {"name": names[0], "positive": "1girl, blue eyes", "position": "left"},
            {"name": names[1], "positive": "1boy, black hair", "position": "right"},
        ],
        "speak_text": "",
    }
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow_type", "multi_char_mask_enabled", "expected_provider", "expects_mask"),
    [
        ("v3", True, "comfy", True),
        ("v3", False, "comfy", False),
        ("chansub", True, "chansub", False),
    ],
)
async def test_manual_draw_routes_local_v3_mask_and_chansub_without_mask(
    monkeypatch,
    tmp_path,
    workflow_type,
    multi_char_mask_enabled,
    expected_provider,
    expects_mask,
):
    _write_fake_restore_module(tmp_path)
    config = {
        "illustration_workflow_type": workflow_type,
        "restore_prompt_file": "restore_workflow_prompt_llm.py",
        "bot_selected": "Example Bot",
        "illustration_context_toggles": {
            "multi_char_mask_enabled": multi_char_mask_enabled,
            "prompt_format": "v3",
        },
    }
    workflow_profiles.normalize_workflow_config(config)
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "CUSTOMPROMPT_DIR", str(tmp_path))

    multi_context_calls = 0

    async def fake_multi_context(_result):
        nonlocal multi_context_calls
        multi_context_calls += 1
        return {
            "enable": True,
            "char_num": 2,
            "characters": [],
            "character_order": ["Alice", "Bob"],
            "layout": {"character_order": ["Alice", "Bob"], "regions": []},
            "mask_location": "region_mask",
        }

    monkeypatch.setattr(
        server,
        "_build_restore_manual_multi_char_context",
        fake_multi_context,
    )
    queued = {}
    queued_event = asyncio.Event()

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        queued.update({
            "item_type": item_type,
            "label": label,
            "params": params,
            "priority": priority,
        })
        queued_event.set()

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    response = await server.handle_api_restore_manual_draw(_JsonRequest({
        "character_count": 2,
        "char_names": ["Alice", "Bob"],
        "situation_mode": "llm",
        "situation": "",
        "postprocess_test": False,
        "postprocess_text_mode": "llm",
        "postprocess_text": "",
    }))
    await asyncio.wait_for(queued_event.wait(), timeout=1)

    assert response.status == 200
    raw_body = queued["params"]["raw_body"]
    assert raw_body["illustration_provider"] == expected_provider
    assert ("illustration_multi_char" in raw_body) is expects_mask
    assert multi_context_calls == (1 if expects_mask else 0)
