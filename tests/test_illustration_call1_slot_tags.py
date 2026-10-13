"""Sparse CALL1 checkpoints, independent forward filling and real handoff paths."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import illustration_flow as flow
from modes import illustration_context_pipeline as pipeline
from modes import illustration_prompt_builders as builders
from modes import illustration_scene_pipeline as scenes


@pytest.mark.parametrize("names,outfits,hairs", [
    (["Ari", "Bao"], ["blue coat, white shirt", "white shirt", "green robe"], ["braid", "hair down"]),
    (["Mira", "Ren"], ["red jacket, black shirt", "black shirt", "yellow raincoat"], ["ponytail", "short hair"]),
    (["서연", "지우"], ["school uniform, cardigan", "school uniform", "track suit"], ["twintails", "hair down"]),
])
def test_checkpoints_replace_whole_aspect_and_restart_only_its_own_forward_fill(names, outfits, hairs):
    first, second = names
    # The response object's insertion order does not change story chronology.
    changes = {
        "SLOT4": {first: {"outfit": outfits[2]}},
        "SLOT0": {first: {"outfit": outfits[0], "hair": hairs[0]},
                  second: {"outfit": "brown jacket", "hair": "short hair"}},
        "SLOT2": {first: {"outfit": outfits[1]}},
        "SLOT3": {first: {"hair": hairs[1]}},
    }
    original = deepcopy(changes)
    parsed = scenes.parse_state(json.dumps(changes, ensure_ascii=False), names, list(range(6)))
    table = scenes.fill_state_table(parsed, names, list(range(6)))
    states = table["states"]
    assert [states[f"SLOT{i}"][first]["outfit"] for i in range(6)] == [
        outfits[0], outfits[0], outfits[1], outfits[1], outfits[2], outfits[2]]
    assert [states[f"SLOT{i}"][first]["hair"] for i in range(6)] == [hairs[0]] * 3 + [hairs[1]] * 3
    assert all(states[f"SLOT{i}"][second] == changes["SLOT0"][second] for i in range(6))
    assert changes == original == table["changes"]
    states["SLOT5"][first]["outfit"] = "mutation"
    assert states["SLOT4"][first]["outfit"] == outfits[2] and changes == original


def test_explicit_empty_checkpoint_clears_without_resurrecting_previous_outfit():
    changes = {"SLOT7": {"Nora": {"outfit": "blue coat", "hair": "braid"}},
               "SLOT11": {"Nora": {"outfit": ""}},
               "SLOT15": {"Nora": {"hair": "hair down"}}}
    slots = [7, 9, 11, 15, 18]
    parsed = scenes.parse_state(json.dumps(changes), ["Nora"], slots)
    states = scenes.fill_state_table(parsed, ["Nora"], slots)["states"]
    assert states["SLOT9"]["Nora"] == {"outfit": "blue coat", "hair": "braid"}
    assert states["SLOT11"]["Nora"] == {"outfit": "", "hair": "braid"}
    assert states["SLOT18"]["Nora"] == {"outfit": "", "hair": "hair down"}


def test_blank_input_table_keeps_actual_slot_labels_names_and_source_alignment():
    prompts = pipeline.load_prompt_files()
    messages = builders.call1_messages(prompts, ["Nora", "지우"], {}, {}, "Earlier history", "[C001]\nNow",
        slots=[7, 11, 18], slotted="Now[Slot 7]", segment_slots={"C001": 7})
    user = messages[1]["content"]
    assert "| Character | SLOT7 | SLOT11 | SLOT18 |" in user
    assert "| Nora |  |  |  |" in user and "| 지우 |  |  |  |" in user
    assert "Now[Slot 7]" in user and '"C001": 7' in user
    assert "Earlier history" in user


@pytest.mark.parametrize("response", [
    {}, {"SLOT0": {"Nora": {"outfit": "coat"}}},
    {"SLOT99": {"Nora": {"outfit": "coat", "hair": "braid"}}},
    {"SLOT0": {"Someone": {"outfit": "coat", "hair": "braid"}}},
    {"SLOT0": {"Nora": {"outfit": [], "hair": "braid"}}},
])
def test_unplaceable_or_missing_initial_data_logs_the_input(response, capsys):
    with pytest.raises(ValueError):
        scenes.parse_state(json.dumps(response), ["Nora"], [0, 1])
    assert "[ILLUST_CONTEXT:CALL1]" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sparse_call1_reaches_plan_detail_history_and_clickable_flow_node(monkeypatch):
    names = ["Nora", "Jun"]
    initial = {"Nora": {"outfit": "blue coat, white shirt", "hair": "brown hair, braid"},
               "Jun": {"outfit": "brown jacket", "hair": "short hair"}}
    changes = {"SLOT0": initial, "SLOT1": {"Nora": {"outfit": "white shirt"}},
               "SLOT2": {"Nora": {"hair": "brown hair, hair down"}}}
    captured = {}
    selected = [{"anchor_segment": f"C{i + 1:03d}", "characters": ["Nora"]} for i in range(3)]

    @flow.llm_call
    async def llm(stage, messages, *_args, **kwargs):
        text = str(messages[-1]["content"])
        captured.setdefault(stage, []).append(text)
        if stage == "CALL1":
            assert "| Character | SLOT0 | SLOT1 | SLOT2 |" in text
            response = json.dumps(changes)
            assert kwargs["result_validator"](response) == (True, "")
            return response
        if stage == "CALL2-SELECT":
            return json.dumps({"scene_plan": selected})
        if stage.startswith("CALL2-PLAN"):
            payload = json.loads(text.split("# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)\n")[1].split("\n\n")[0])
            table = json.loads(text.split("# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n")[1].split("\n\n")[0])
            assert all("Jun" not in state for state in table.values())
            assert table["SLOT2"]["Nora"] == {"outfit": "white shirt", "hair": "brown hair, hair down"}
            return json.dumps({"scene_plan": [{**item, "scene_brief": "Nora pauses at this boundary"} for item in payload["scene_plan"]]})
        if stage == "CALL2-AUDIT":
            return '{"audited_scene_plan":[]}'
        if stage.startswith("CALL2-DETAIL"):
            payload = json.loads(text.split("# ONLY THIS BATCH'S PREPARED SCENES\n")[1].split("\n\n")[0])
            captured.setdefault("prepared", []).extend(payload["scenes"])
            return "<lb-xnai>\n" + json.dumps({"scenes": [
                {"slot": item["slot"], "camera": "medium shot", "characters": [
                    {"name": "Nora", "positive": "white shirt", "negative": "", "position": "center"}],
                 "scene": "quiet room", "supplement": ""} for item in payload["scenes"]]}) + "\n</lb-xnai>"
        raise AssertionError(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm)
    item = SimpleNamespace(id="call1-slot-tags-qa", label="삽화", type="illustration_llm_build", params={}, status="pending")
    flow.queue_added(item)

    class Runner:
        @flow.queue_execution
        async def execute(self, item):
            return await pipeline.build_from_context(
                {"session_id": "slot-tag-qa", "chats": [{"role": "char", "data":
                    "Nora and Jun arrive.\n\nNora removes her coat.\n\nNora unties her braid."}],
                 "target_slotted": "Nora and Jun arrive.[Slot 0]\n\nNora removes her coat.[Slot 1]\n\nNora unties her braid.[Slot 2]"},
                {"output_count_min": 3, "output_count_max": 3, "key_visual": False,
                 "call3_enabled": False, "call2_parallel_enabled": False, "call2_detail_parallel_enabled": False},
                "### Nora\n-Appearance\nbrown hair\n### Jun\n-Appearance\nshort hair",
                pre_resolved_profile_result={"current_characters": [{"name": name} for name in names]})

    result = await Runner().execute(item)
    table = result["call1_result"]["state_table"]
    assert table["changes"] == changes and table["characters"] == names
    prepared = sorted(captured["prepared"], key=lambda s: s["slot"])
    assert [s["characters"][0]["state"] for s in prepared] == [table["states"][f"SLOT{i}"]["Nora"] for i in range(3)]
    assert all("anchor_event" not in s["characters"][0] for s in prepared)
    assert json.loads(result["character_states_after"]["nora"]["call1_status"]) == table["states"]["SLOT2"]["Nora"]
    graph = flow.snapshot()
    node = next(n for n in graph["nodes"] if n["label"] == "CALL1")
    assert "call1_state_table" not in node
    detail = flow.detail(graph["id"], node["id"])
    assert detail["call1_state_table"] == table and json.loads(detail["output"]) == changes
