import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_context_pipeline as pipeline


def test_context_and_result_transport_markers():
    session_id = "session_12345678"
    context_payload = {
        "session_id": session_id,
        "target_slotted": "첫 문장.\n\n[Slot 0]\n\n둘째 문장.",
        "chats": [
            {"role": "user", "data": "질문"},
            {"role": "assistant", "content": "응답"},
        ],
    }
    parsed = pipeline.parse_context_request(
        pipeline.CONTEXT_PREFIX + "\n" + json.dumps(context_payload, ensure_ascii=False)
    )

    assert parsed["session_id"] == session_id
    assert parsed["chats"] == [
        {"role": "user", "data": "질문"},
        {"role": "char", "data": "응답"},
    ]
    assert "[Slot 0]" in parsed["target_slotted"]
    assert pipeline.parse_result_request(
        pipeline.RESULT_PREFIX + "\n" + json.dumps({"session_id": session_id, "slot": -1})
    ) == {"session_id": session_id, "index": None, "slot": -1}
    assert pipeline.parse_regenerate_request(
        pipeline.REGENERATE_PREFIX + "\n" + json.dumps({"session_id": session_id, "slot": 0})
    ) == {"session_id": session_id, "slot": 0}


def test_toon_parse_and_slot_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    toon = """<lb-xnai>
keyvis:
  camera: close-up
  characters[1]:
    - name: hana
      positive: 1girl, black hair
      negative: bad hands
  scene: classroom
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, black hair, smile
    scene: classroom, sunset
    slot: 0
</lb-xnai>"""
    items = pipeline.parse_toon_plan(toon, pipeline.merged_toggles({"scene_max": 11}))

    assert [item["slot"] for item in items] == [-1, 0]
    pipeline.create_session("cache_test_1234", "context")
    pipeline.set_session_result("cache_test_1234", items, [b"keyvis", b"scene"])
    assert pipeline.session_image_by_slot("cache_test_1234", -1) == b"keyvis"
    assert pipeline.session_image_by_slot("cache_test_1234", 0) == b"scene"
    assert pipeline.session_item_by_slot("cache_test_1234", 0)["scene"] == "classroom, sunset"
    assert pipeline.update_session_image_by_slot("cache_test_1234", 0, b"rerolled")
    assert pipeline.session_image_by_slot("cache_test_1234", 0) == b"rerolled"
    pipeline._SESSIONS.pop("cache_test_1234")
    assert pipeline.session_item_by_slot("cache_test_1234", 0)["scene"] == "classroom, sunset"


def test_call2_macro_render_has_no_risu_macros():
    prompts = pipeline.load_prompt_files()
    rendered = pipeline.render_call2_prompt(
        prompts["call2_system"],
        pipeline.merged_toggles({
            "nsfw": False,
            "supplement": True,
            "key_visual": True,
            "character_limit": 2,
            "context_history": True,
            "focus": "hana",
            "direction": "Prefer cinematic lighting.",
        }),
        "### hana\n-Appearance: 1girl, black hair",
    )

    assert "{{" not in rendered
    assert "Limit characters to max 2" in rendered
    assert "### hana" in rendered
    assert 'focus on the character(s): "hana"' in rendered
    assert "Prefer cinematic lighting." in rendered


@pytest.mark.asyncio
async def test_call1_enrichment_is_passed_to_call2_without_changing_slots(monkeypatch):
    calls = []
    responses = [
        """[Position]
첫 문장.
[/Position]
[Visual Content #01]
A character turns toward the window.
[DynamicPrompt scene="01"]
looking away, window
[/DynamicPrompt]
[CharacterBaseTags]
hana : 1girl, black hair
[/CharacterBaseTags]""",
        """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, black hair
    scene: classroom
    slot: 0
</lb-xnai>""",
    ]

    async def fake_call(task_key, messages, **kwargs):
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "pipeline_test_1234",
            "target_slotted": "첫 문장.\n\n[Slot 0]\n\n둘째 문장.",
            "chats": [
                {"role": "user", "data": "창밖을 봐."},
                {"role": "char", "data": "첫 문장.\n\n둘째 문장."},
            ],
        },
        {"call3_enabled": False, "speak_enabled": False, "key_visual": False},
        "### hana\n-Appearance: 1girl, black hair",
    )

    call2_text = "\n".join(message["content"] for message in calls[1])
    assert "[Visual Content #01]" in call2_text
    assert "[Slot 0]" in call2_text
    assert result["items"][0]["slot"] == 0
    assert result["items"][0]["raw_positive"]
