"""Independent offline regressions for the redesigned ordinary illustration route."""

import json

import pytest

from modes import illustration_context_pipeline as pipeline
from modes.visual_profiles import cards_to_character_profiles


def _one_profile():
    return {"Ari": cards_to_character_profiles("Ari", [{
        "id": "ari_card", "label": "Ari", "appearance": ["red hair"],
        "default_outfit": ["blue coat"],
    }])}


def _detail(slot=0):
    return ("<lb-xnai>\nscenes:\n"
            f"  - slot: {slot}\n"
            "    camera: medium shot from the left\n"
            "    characters:\n"
            "      - name: Ari\n"
            "        positive: deliberately selected silhouette\n"
            "        negative: unwanted artifact\n"
            "        position: left\n"
            "    scene: Ari holds a key.\n"
            "    supplement: her hand encloses the key\n"
            "</lb-xnai>")


@pytest.mark.asyncio
async def test_call3_hostile_extra_fields_and_slots_do_not_mutate_locked_scene(monkeypatch):
    calls = []

    async def mocked_llm(stage, _messages, *_args, **_kwargs):
        calls.append(stage)
        if stage == "CALL1":
            return json.dumps({"state_at_start": [{"character": "Ari", "status": "blue coat and tied hair"}],
                               "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [{"anchor_segment": "C001", "characters": ["Ari"],
                                               "scene_brief": "Ari holds a key in a medium shot."}]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
        if stage == "CALL2-DETAIL":
            return _detail()
        if stage == "CALL3":
            return ('[Scene slot=0]\nAri: "I have the key." #normal\n'
                    'camera: hacked camera\npositive: hacked coat\n'
                    'scene: hacked event\nslot: 5\nidentity: Eve\n'
                    '[Scene slot=9]\nEve: "not a selected scene" #normal')
        raise AssertionError(f"unexpected LLM call: {stage}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", mocked_llm)
    result = await pipeline.build_from_context(
        {"session_id": "offline-call3", "chats": [{"role": "char", "data": "Ari holds a key.\n\nAri smiles."}]},
        {"output_count_min": 1, "output_count_max": 1, "key_visual": False,
         "call3_enabled": True, "speak_enabled": True, "speak_language": "English"},
        "### Ari\n-Name\nAri\n-Appearance\nred hair",
        visual_profiles=_one_profile(),
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}]},
    )
    assert calls == ["CALL1", "CALL2-PLAN", "CALL2-AUDIT", "CALL2-DETAIL", "CALL3"]
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert (item["slot"], item["camera"], item["scene"], item["characters"]) == (
        0, "medium shot from the left", "Ari holds a key.",
        [{"name": "Ari", "positive": "deliberately selected silhouette",
          "negative": "unwanted artifact", "position": "left"}],
    )
    assert 'I have the key.' in item["speak"]
    assert not any(word in item["speak"] for word in ("hacked", "Eve", "slot: 5"))
    assert "hacked" not in item["raw_positive"]
    assert "deliberately selected silhouette" in item["raw_positive"]


@pytest.mark.asyncio
async def test_next_request_uses_saved_call1_chain_not_previous_detail_tags(monkeypatch):
    starts = []
    details = []
    call_count = 0

    async def mocked_llm(stage, messages, *_args, **_kwargs):
        nonlocal call_count
        if stage == "CALL1":
            call_count += 1
            starts.append(messages[-1]["content"])
            if call_count == 1:
                return json.dumps({"state_at_start": [{"character": "Ari", "status": "blue coat, tied hair"}],
                                   "events": [{"segment_id": "C002", "character": "Ari",
                                               "status": "coat carried in hand, hair loose"}]})
            return json.dumps({"state_at_start": [{"character": "Ari",
                                                    "status": "coat carried in hand, hair loose"}],
                               "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [{"anchor_segment": "C001", "characters": ["Ari"],
                                               "scene_brief": "source-grounded beat"}]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
        if stage == "CALL2-DETAIL":
            prepared = json.loads(messages[-1]["content"].split(
                "# ONLY THIS SCENE'S PREPARED DATA\n", 1)[1].split("\n\nReturn only", 1)[0])
            details.append(prepared)
            return _detail()
        raise AssertionError(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", mocked_llm)
    options = {"output_count_min": 1, "output_count_max": 1,
               "key_visual": False, "call3_enabled": False}
    kwargs = {"visual_profiles": _one_profile(),
              "pre_resolved_profile_result": {"current_characters": [{"name": "Ari"}]}}
    first = await pipeline.build_from_context(
        {"session_id": "offline-previous", "chats": [{"role": "char", "data": "Ari enters.\n\nAri removes her coat."}]},
        options, "### Ari\n-Name\nAri\n-Appearance\nred hair", **kwargs,
    )
    assert first["character_states_after"]["ari"]["call1_status"] == "coat carried in hand, hair loose"
    second = await pipeline.build_from_context(
        {"session_id": "offline-current", "chats": [{"role": "char", "data": "Ari returns.\n\nAri pauses."}]},
        options, "### Ari\n-Name\nAri\n-Appearance\nred hair", **kwargs,
        history_plan={"state_before": first["character_states_after"], "current_message_id": "next"},
    )
    assert len(starts) == len(details) == 2
    assert '"Ari": ["blue coat, tied hair", "coat carried in hand, hair loose"]' in starts[1]
    assert "deliberately selected silhouette" not in starts[1]
    assert details[0]["characters"][0]["state"] == ["blue coat, tied hair"]
    assert details[1]["characters"][0]["state"] == ["coat carried in hand, hair loose"]
    assert second["character_states_after"]["ari"]["call1_status"] == "coat carried in hand, hair loose"
