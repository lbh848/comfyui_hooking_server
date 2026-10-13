"""Offline contracts for the ordinary SELECT -> PLAN -> AUDIT -> DETAIL path."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import illustration_flow as flow
from modes import illustration_context_pipeline as pipeline
from modes import illustration_scene_pipeline as scene_pipeline
from modes.visual_profiles import cards_to_character_profiles


def _profiles():
    def card(index, *, name="Ari"):
        return {
            "id": f"card_{index}", "label": f"{name} card {index}",
            "visual_context": f"{name} context {index}",
            "appearance": [f"{name} face {index}"],
            "default_outfit": [f"{name} coat {index}"],
            "gender_tag": "girl" if name == "Ari" else "boy",
            "face_tags": f"{name} face details {index}",
        }
    return {
        "Ari": cards_to_character_profiles("Ari", [card(i) for i in range(1, 4)]),
        "Bao": cards_to_character_profiles("Bao", [card(1, name="Bao")]),
    }


def _profile_result():
    return {
        "current_characters": [{"name": "Ari"}, {"name": "Bao"}],
        "profile_events": [
            {"segment_id": "START", "character": "Ari", "profile_id": "card_1"},
            {"segment_id": "C030", "character": "Ari", "profile_id": "card_2"},
            {"segment_id": "C070", "character": "Ari", "profile_id": "card_3"},
            {"segment_id": "START", "character": "Bao", "profile_id": "card_1"},
        ],
    }


def _segments(count=90):
    return {f"C{i:03d}": {"text": f"beat {i}"} for i in range(1, count + 1)}


def _tracking():
    return {
        "SLOT0": {"Ari": {"outfit": "  blue coat, open coat  ", "hair": "ponytail"},
                  "Bao": {"outfit": "brown jacket", "hair": "short hair"}},
        "SLOT1": {"Ari": {"outfit": "  blue coat, open coat  ", "hair": "ponytail"},
                  "Bao": {"outfit": "", "hair": "short hair"}},
        "SLOT2": {"Ari": {"outfit": "", "hair": "hair down"},
                  "Bao": {"outfit": "", "hair": "short hair"}},
        "SLOT3": {"Ari": {"outfit": "green coat", "hair": "hair down"},
                  "Bao": {"outfit": "", "hair": "short hair"}},
    }


def _scene(anchor, names=("Ari",)):
    return {"anchor_segment": anchor, "scene_brief": f"Visual action at {anchor}",
            "characters": list(names)}


def test_profile_transitions_selected_by_numeric_anchor_and_start_only():
    profiles = _profiles()
    events = _profile_result()
    order = {sid: index for index, sid in enumerate(_segments())}
    assert scene_pipeline.active_profile("Ari", "START", events, profiles, order)["visual_profile_id"] == "card_1"
    for anchor, expected in [("C020", "card_1"), ("C030", "card_2"),
                             ("C044", "card_2"), ("C070", "card_3"),
                             ("C085", "card_3")]:
        assert scene_pipeline.active_profile("Ari", anchor, events, profiles, order)["visual_profile_id"] == expected


def test_selected_profile_and_raw_status_slice_never_leak_other_cards_or_people():
    profiles = _profiles()
    result = _profile_result()
    segments = _segments()
    tracking = _tracking()
    extra = "### Ari\n-Name\nAri\n-Appearance\nlegacy Ari\n\n### Bao\n-Name\nBao\n-Appearance\nBao secret"
    slot_map = {anchor: index for index, anchor in enumerate(["C020", "C044", "C070", "C085"])}
    for anchor, card_id in [("C020", "card_1"), ("C044", "card_2"),
                            ("C070", "card_3"), ("C085", "card_3")]:
        sliced, bases = scene_pipeline.slice_scene(
            _scene(anchor), slot_map=slot_map, profile_result=result,
            visual_profiles=profiles, segments=segments, tracking=tracking,
            extra_reference=extra,
        )
        assert set(sliced) == {"anchor_segment", "slot", "brief", "characters"}
        assert sliced["characters"][0]["name"] == "Ari"
        assert bases["Ari"]["visual_profile_id"] == card_id
        profile = sliced["characters"][0]["profile"]
        assert profile["id"] == card_id
        assert profile["gender_tag"] == "girl"
        assert profile["visual_context"] == f"Ari context {card_id[-1]}"
        assert f"Ari coat {card_id[-1]}" in str(profile)
        assert "Bao" not in str(sliced) and "Bao" not in str(bases)
        assert "legacy Ari" not in str(sliced) and "Bao secret" not in str(sliced)
        if card_id != "card_1":
            assert "Ari coat 1" not in str(sliced)
    at_anchor, _ = scene_pipeline.slice_scene(
        _scene("C070"), slot_map=slot_map, profile_result=result,
        visual_profiles=profiles, segments=segments, tracking=tracking,
        extra_reference=extra,
    )
    ari = at_anchor["characters"][0]
    assert ari["state"] == {"outfit": "", "hair": "hair down"}
    assert "anchor_event" not in ari
    assert "green coat" not in str(at_anchor)
    assert tracking["SLOT0"]["Ari"]["outfit"] == "  blue coat, open coat  "


@pytest.mark.parametrize("name,past,memory,start,current_change,outfit,hair", [
    ("Shoko", "Shoko has completely undressed and set all clothing aside.", {},
     "Fully unclothed; long brown hair loose.", "She ties her loose hair into a braid.", "nude", "long hair, brown hair, braid"),
    ("Mira", "Mira removed the last garment and left it on the changing-room bench.", {},
     "Unclothed; hair loose.", "She puts on a green robe.", "green robe", "hair down"),
    ("Ren", "Ren waits in the swimsuit; no clothing or hair changes occurred.",
     {"Ren": ["Yellow swimsuit; hair in a braid."]},
     "Yellow swimsuit; hair in a braid.", "Ren takes the swimsuit off and holds it.", "nude", "braid"),
    ("Ari", "Ari's braid becomes windblown, still braided; her swimsuit is unchanged.",
     {"Ari": ["Blue swimsuit; hair in a braid."]},
     "Blue swimsuit; windblown braid still intact.", "Ari unties the braid and lets her hair down.", "blue swimsuit", "hair down"),
])
def test_narrative_memory_and_current_result_reach_own_slot_without_default_revival(
        name, past, memory, start, current_change, outfit, hair):
    from modes import illustration_prompt_builders as builders

    segments = {"C001": {"text": current_change}}
    prompts = pipeline.load_prompt_files()
    messages = builders.call1_messages(
        prompts, [name], memory, {name: {"default_outfit": ["old uniform"]}},
        past, "[C001]\n" + current_change)
    # Source chronology and detail memory stay separate and lossless; these
    # assertions check handoff, not the semantic accuracy of a mocked model.
    user = messages[1]["content"]
    assert past in user and current_change in user
    assert json.dumps(memory, ensure_ascii=False) in user
    assert "old uniform" in user
    assert user.index("# PAST HISTORY") < user.index("# CURRENT SEGMENTS")
    state = {"outfit": outfit, "hair": hair}
    tracking = scene_pipeline.parse_state(json.dumps({"SLOT0": {name: state}}), [name], [0])
    scene, _ = scene_pipeline.slice_scene(
        {"anchor_segment": "C001", "characters": [name], "scene_brief": current_change},
        slot_map={"C001": 0}, profile_result={}, visual_profiles=None,
        segments=segments, tracking=tracking,
        extra_reference=f"### {name}\n-Name\n{name}\n-Appearance\nshort hair")
    assert scene["characters"][0]["state"] == state
    assert "anchor_event" not in scene["characters"][0]
    assert tracking == {"SLOT0": {name: state}}
    assert "old uniform" not in str(state)


def test_prior_call1_status_ignores_last_image_and_real_lb_extra_is_name_bounded():
    prior = {"state_before": {
        "ari": {"canonical_name": "Ari", "call1_status": "vest removed, loose hair",
                "last_visual_positive": "stale school uniform"},
        "bao": {"canonical_name": "Bao", "last_visual_positive": "stale hat"},
    }}
    assert scene_pipeline.previous_statuses(["Ari", "Bao"], prior) == {
        "Ari": ["vest removed, loose hair"],
    }
    reference = "### Ari\n-Name\nAri\n-Appearance\nred hair\n-default_outfit\ncoat\n### Bao\n-Name\nBao\n-Appearance\nblack hair"
    assert "red hair" in scene_pipeline.profile_for_detail("Ari", {}, reference)["lb_extra"]
    assert "Bao" not in str(scene_pipeline.profile_for_detail("Ari", {}, reference))


def test_call1_requires_initial_tag_fields_and_uses_server_slot_order():
    tracking = _tracking()
    names = ["Ari", "Bao"]
    assert scene_pipeline.parse_state(json.dumps(tracking), names, [0, 1, 2, 3]) == tracking
    reversed_keys = dict(reversed(list(tracking.items())))
    assert scene_pipeline.fill_state_table(reversed_keys, names, [0, 1, 2, 3])["states"] == tracking
    for invalid in [
        {**tracking, "wardrobe_events": []},
        {**tracking, "SLOT99": tracking["SLOT0"]},
        {**tracking, "SLOT3": {"Someone": {"outfit": "coat", "hair": "braid"}}},
        {**tracking, "SLOT0": {"Ari": tracking["SLOT0"]["Ari"]}},
    ]:
        with pytest.raises(ValueError):
            scene_pipeline.parse_state(json.dumps(invalid), names, [0, 1, 2, 3])


def test_plan_preserves_briefs_and_rejects_structural_changes():
    segments = _segments(3)
    slot_map = {"C001": 0, "C002": 1, "C003": 1}
    toggles = {"output_count_min": 2, "output_count_max": 2, "character_limit": 2}
    original = [
        {"anchor_segment": "C001", "characters": ["Ari"], "scene_brief": "Keep  two spaces."},
        {"anchor_segment": "C002", "characters": ["Ari", "Bao"], "scene_brief": "Incorrect pose"},
    ]
    assert scene_pipeline.parse_plan(json.dumps({"scene_plan": original}),
        segments=segments, slot_map=slot_map, candidates=[0, 1], names=["Ari", "Bao"],
        toggles=toggles) == original
    with pytest.raises(ValueError, match="slot"):
        scene_pipeline.parse_plan(json.dumps({"scene_plan": [original[1],
            {**original[0], "anchor_segment": "C003"}]}),
            segments=segments, slot_map=slot_map, candidates=[0, 1], names=["Ari", "Bao"],
            toggles=toggles)
    for plan in [[{**original[0], "slot": 0}, original[1]],
                 [{**original[0], "characters": ["Unknown"]}, original[1]],
                 [original[1], original[0]]]:
        with pytest.raises(ValueError):
            scene_pipeline.parse_plan(json.dumps({"scene_plan": plan}),
                segments=segments, slot_map=slot_map, candidates=[0, 1],
                names=["Ari", "Bao"], toggles=toggles)
    with pytest.raises(ValueError, match="SINGLE"):
        scene_pipeline.parse_plan(json.dumps({"scene_plan": original}),
            segments=segments, slot_map=slot_map, candidates=[0, 1],
            names=["Ari", "Bao"], toggles=toggles, single_names=["Ari"])








@pytest.mark.parametrize("name", ["Mira", "Ren"])
@pytest.mark.parametrize("legacy_brief", [None, "", "An obsolete choice of instant."])
def test_select_returns_only_anchor_and_names_even_with_an_older_brief(name, legacy_brief):
    selected = {"anchor_segment": "C001", "characters": [name]}
    response = deepcopy(selected)
    if legacy_brief is not None:
        response["scene_brief"] = legacy_brief
    actual = scene_pipeline.parse_plan(json.dumps({"scene_plan": [response]}),
        segments={"C001": {"text": f"{name} reaches out, then lowers the hand."}},
        slot_map={"C001": 0}, candidates=[0], names=[name],
        toggles={"output_count_min": 1, "output_count_max": 1, "character_limit": 1},
        single_names=[name], stage="CALL2-SELECT")
    assert actual == [selected]
    actual[0]["characters"].append("Someone else")
    assert response["characters"] == [name]


@pytest.mark.parametrize("brief", [None, "", "   "])
def test_plan_still_requires_a_created_visual_brief(brief):
    selected = {"anchor_segment": "C001", "characters": ["Noel"]}
    response = deepcopy(selected)
    if brief is not None:
        response["scene_brief"] = brief
    with pytest.raises(ValueError, match="CALL2-PLAN"):
        scene_pipeline.parse_plan(json.dumps({"scene_plan": [response]}),
            segments={"C001": {"text": "Noel accepts the offered key."}},
            slot_map={"C001": 0}, candidates=[0], names=["Noel"],
            toggles={"output_count_min": 1, "output_count_max": 1, "character_limit": 1},
            assigned_scenes=[selected])


@pytest.mark.parametrize("change", ["unknown_anchor", "unknown_name", "order", "missing", "second_single_person"])
def test_anchor_only_selection_keeps_count_order_and_single_identity_constraints(change):
    selected = [{"anchor_segment": "C001", "characters": ["Lena"]},
                {"anchor_segment": "C002", "characters": ["Lena"]}]
    if change == "unknown_anchor":
        selected[1]["anchor_segment"] = "C999"
    elif change == "unknown_name":
        selected[1]["characters"] = ["Unknown"]
    elif change == "order":
        selected.reverse()
    elif change == "missing":
        selected.pop()
    else:
        selected[1]["characters"].append("Kian")
    with pytest.raises(ValueError, match="CALL2-SELECT"):
        scene_pipeline.parse_plan(json.dumps({"scene_plan": selected}),
            segments=_segments(2), slot_map={"C001": 0, "C002": 1}, candidates=[0, 1],
            names=["Lena", "Kian"], single_names=["Lena"],
            toggles={"output_count_min": 2, "output_count_max": 2, "character_limit": 2},
            stage="CALL2-SELECT")


@pytest.mark.asyncio
@pytest.mark.parametrize("candidates,narrative", [
    (["Doyun", "Shoko", "Hibiki"], "Doyun and Shoko interact while Hibiki observes behind them."),
    (["Mira", "Ren", "Sana"], "Mira hands a map to Ren; Sana waits beside the doorway."),
])
@pytest.mark.parametrize("selected_indexes,limit", [((0, 1), 2), ((2,), 2), ((0, 1, 2), 3)])
async def test_keyvis_candidate_pool_can_exceed_the_selected_cast(
        monkeypatch, candidates, narrative, selected_indexes, limit):
    selected = [candidates[index] for index in selected_indexes]
    output = "<lb-xnai>\n" + json.dumps({"keyvis": {
        "camera": "medium shot of the selected participants",
        "characters": [{"name": name, "positive": f"{name} visible in the frame",
                        "position": "foreground"} for name in selected],
        "scene": narrative, "supplement": "",
    }, "scenes": []}) + "\n</lb-xnai>"

    async def call(stage, messages, *_args, **kwargs):
        assert stage == "CALL2-KEYVIS"
        content = messages[-1]["content"]
        roster = content.split("# ALLOWED CANONICAL CHARACTER CANDIDATES (CHOOSE A SUBSET)\n", 1)[1]
        assert json.loads(roster.split("\n\n# OUTPUT SCHEMA", 1)[0]) == candidates
        accepted, _ = kwargs["result_validator"](output)
        assert accepted
        return output

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", call)
    descriptor, raw = await pipeline._run_call2_keyvis(
        call2_context_messages=[{"role": "user", "content": narrative}],
        allowed_character_names=candidates,
        toggles=pipeline.merged_toggles({"character_limit": limit}), stream_notify=None)
    assert [character["name"] for character in descriptor["characters"]] == selected
    assert raw == output


@pytest.mark.parametrize("names", [("Doyun", "Shoko", "Hibiki"), ("Mira", "Ren", "Sana")])
def test_keyvis_three_candidates_are_not_three_required_output_entries(names):
    # The reported failure and its renamed equivalent remain invalid; the
    # prompt fix must select a smaller cast rather than silently truncate it.
    output = "<lb-xnai>\n" + json.dumps({"keyvis": {
        "camera": "three participants in one frame",
        "characters": [{"name": name, "positive": f"{name} visible"} for name in names],
        "scene": "A shared interaction with an observer.", "supplement": "",
    }, "scenes": []}) + "\n</lb-xnai>"
    descriptor, reason = pipeline._parse_call2_keyvis_output(
        output, pipeline.merged_toggles({"character_limit": 2}), list(names), "TEST-KEYVIS-CAST-LIMIT")
    assert descriptor is None
    assert "count=3, limit=2" in reason


@pytest.mark.parametrize("names,brief", [
    (["Doyun", "Shoko"], "Shoko reacts on the bed while Doyun remains identifiable beside her in the same crop."),
    (["Doyun", "Shoko"], "An oblique view includes Shoko looking up and Doyun leaning above her, both identifiable."),
    (["Mira", "Ren"], "Ren looks up from the map toward Mira; the crop includes both identifiable participants."),
    (["Sana"], "Sana accepts a key from one anonymous hand entering at the frame edge; its owner stays outside."),
])
def test_plan_visible_cast_handoff_keeps_assigned_identities_and_single_fragment(names, brief):
    from modes import illustration_prompt_builders as builders

    selected = [{"anchor_segment": "C001", "characters": names}]
    segments = {"C001": {"text": brief}}
    tracking = {"SLOT0": {name: {"outfit": "coat", "hair": "short hair"} for name in names}}
    messages = builders.plan_messages(pipeline.load_prompt_files(), "", selected, names,
        None, tracking, [], "", "[C001]\n" + brief, {})
    assigned = messages[1]["content"].split("\n", 1)[1].split("\n\n# CURRENT TRACKED", 1)[0]
    assert json.loads(assigned)["scene_plan"] == selected
    planned = scene_pipeline.parse_plan(json.dumps({"scene_plan": [
        {**selected[0], "scene_brief": brief}]}), segments=segments,
        slot_map={"C001": 0}, candidates=[0], names=names,
        toggles={"output_count_min": 1, "output_count_max": 1, "character_limit": 2},
        assigned_scenes=selected)
    prepared, _ = scene_pipeline.slice_scene(planned[0], slot_map={"C001": 0},
        profile_result={}, visual_profiles=None, segments=segments, tracking=tracking, extra_reference="")
    assert prepared["brief"] == brief
    assert [character["name"] for character in prepared["characters"]] == names
    detail = {"slot": 0, "camera": "the planned crop", "scene": brief, "supplement": "",
              "characters": [{"name": name, "positive": f"{name} visible", "negative": "",
                              "position": "in frame"} for name in names]}
    output = lambda item: "<lb-xnai>\n" + json.dumps({"scenes": [item]}) + "\n</lb-xnai>"
    parsed = scene_pipeline.parse_detail(output(detail), prepared)
    assert [character["name"] for character in parsed["characters"]] == names
    # An unseen POV owner cannot disappear from an assigned pair; a valid
    # anonymous fragment cannot turn into an additional named character.
    wrong = deepcopy(detail)
    if len(names) == 2:
        wrong["characters"] = wrong["characters"][1:]
    else:
        wrong["characters"].append({**wrong["characters"][0], "name": "Off-frame partner"})
    with pytest.raises(ValueError, match="캐릭터 수 불일치"):
        scene_pipeline.parse_detail(output(wrong), prepared)


def _detail(slot=0, *, name="Ari", positive="unusual scar, blue hair"):
    return ("<lb-xnai>\nscenes:\n"
            f"  - slot: {slot}\n"
            "    camera: close-up, oblique\n"
            "    characters:\n"
            f"      - name: {name}\n"
            f"        positive: {positive}\n"
            "        negative: unwanted marks\n"
            "        position: foreground\n"
            "    scene: Ari reaches for a key.\n"
            "    supplement: hand enters from left\n"
            "</lb-xnai>")


def test_detail_schema_and_raw_transport_do_not_restore_default_outfit():
    prepared = {"slot": 0, "characters": [{"name": "Ari"}]}
    descriptor = scene_pipeline.parse_detail(_detail(), prepared)
    assert descriptor["characters"][0]["positive"] == "unusual scar, blue hair"
    assert "outfit_state" not in str(descriptor)
    prompts = {"call2_preset": "[Positive]\n[SETUP]\n{setup}\n[CHAR]\n{char}\n[SUPPLEMENT]\n{supplement}\n[Negative]\nbase"}
    raw, negative = pipeline.build_raw_prompt(descriptor, "narrative", prompts,
        {"prompt_format": "v3"})
    assert "unusual scar, blue hair" in raw
    assert "Ari coat" not in raw
    assert "hand enters from left" in raw
    assert negative == "base, unwanted marks"
    with pytest.raises(ValueError, match="스키마"):
        scene_pipeline.parse_detail(_detail().replace("    supplement:", "    outfit_state: clothed\n    supplement:"), prepared)
    with pytest.raises(ValueError, match="name"):
        scene_pipeline.parse_detail(_detail(name="Bao"), prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("Ari", "Bao"), ("Lena", "Tao")])
@pytest.mark.parametrize("enable_masks", [True, False])
async def test_ready_scene_does_not_wait_for_sibling_mask_and_call3_graph_joins_workers(
    monkeypatch, names, enable_masks,
):
    slow_mask, release_mask, keyvis_started, release_keyvis = (asyncio.Event() for _ in range(4))
    fast_registered = asyncio.Event()
    registered = []
    selected = [_scene("C001", names), _scene("C002", names), _scene("C003", names[:1])]

    @flow.llm_call
    async def llm_stub(stage, messages, *_args, **kwargs):
        flow.llm_metadata(status="processing")
        base = stage.split()[0]
        if base == "CALL1":
            return json.dumps({"SLOT0": {name: {"outfit": "coat", "hair": "short hair"} for name in names}})
        if base in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": selected})
        if base == "CALL2-DETAIL":
            batch = json.JSONDecoder().raw_decode(messages[-1]["content"].split(
                "# ONLY THIS BATCH'S PREPARED SCENES\n", 1)[1])[0]["scenes"]
            return "<lb-xnai>\n" + json.dumps({"scenes": [
                {"slot": scene["slot"], "camera": "oblique medium shot", "scene": scene["brief"],
                 "supplement": "", "characters": [{"name": ch["name"], "positive": "coat, reaching",
                     "negative": "", "position": "center"} for ch in scene["characters"]]}
                for scene in batch]}) + "\n</lb-xnai>"
        if base == "MULTI-CHAR-MASK":
            payload = json.loads(messages[-1]["content"])
            if payload["slot"] == 0:
                slow_mask.set()
                await release_mask.wait()
            mask_names = [character["name"] for character in payload["characters"]]
            output = json.dumps({"background_prompt": "room, daylight", "composition_prompt": f"{len(mask_names)} figures reaching",
                "regions": [dict(name=name, x=x, y=y, width=w, height=h)
                    for name, (x, y, w, h) in zip(mask_names, [(0.25, 0.2, 0.5, 0.6), (0.2, 0.4, 0.6, 0.5)])]})
            assert kwargs["result_validator"](output) == (True, "")
            return output
        if base == "CALL2-KEYVIS":
            keyvis_started.set()
            await release_keyvis.wait()
            return "key visual"
        if base == "CALL3":
            return "dialogue"
        raise AssertionError(stage)

    async def keyvis_stub(**kwargs):
        await llm_stub("CALL2-KEYVIS", [])
        return ({"kind": "keyvis", "slot": -1, "camera": "wide shot", "scene": "two figures reaching",
            "supplement": "", "characters": [{"name": name, "positive": "coat, reaching",
                "negative": "", "position": "center"} for name in names]}, "key visual")

    async def call3_stub(*args, **kwargs):
        return {"output": await llm_stub("CALL3", [])}

    async def register(data):
        descriptor = data["items"][0]
        if enable_masks:
            assert descriptor["multi_char_layout"]["mask_render_version"] == pipeline.multi_char_mask.MASK_RENDER_VERSION
        registered.append(descriptor["slot"])
        flow.queue_added(SimpleNamespace(id=f"slot-{descriptor['slot']}", label=f"scene-{descriptor['slot']}",
            type="illustration", params={}, status="pending"))
        if {1, 2}.issubset(registered):
            fast_registered.set()

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_run_call2_keyvis", keyvis_stub)
    monkeypatch.setattr(pipeline, "_build_call3_dialogue_with_recovery", call3_stub)
    job = SimpleNamespace(id="mask-ready-flow", label="mask-ready-flow", type="illustration_llm_build",
                          params={}, status="pending")
    flow.queue_added(job)

    class Runner:
        @flow.queue_execution
        async def execute(self, item):
            return await pipeline.build_from_context(
                {"session_id": "mask-ready-flow", "chats": [{"role": "char", "data":
                    "Two visitors enter the room.\n\nThey reach for a box.\n\nOne visitor closes the lid.\n\nThey leave the room."}]},
                {"output_count_min": 3, "output_count_max": 3, "call2_parallel_enabled": False,
                 "call2_detail_parallel_enabled": False, "key_visual": True, "speak_enabled": True},
                "\n\n".join(f"### {name}\n-Name\n{name}\n-Appearance\ncoat" for name in names),
                pre_resolved_profile_result={"current_characters": [{"name": name} for name in names]},
                enable_multi_char_layout=enable_masks, on_call2_ready=register, on_keyvis_ready=register)

    running = asyncio.create_task(Runner().execute(job))
    try:
        await asyncio.wait_for(keyvis_started.wait(), 3)
        await asyncio.wait_for(fast_registered.wait(), 3)
        assert not any(node["label"] == "CALL3" for node in flow.snapshot()["nodes"])
        if enable_masks:
            await asyncio.wait_for(slow_mask.wait(), 3)
            assert 0 not in registered
            release_mask.set()
        release_keyvis.set()
        result = await asyncio.wait_for(running, 3)
    finally:
        release_mask.set(); release_keyvis.set()
        await asyncio.gather(running, return_exceptions=True)
    assert sorted(registered) == [-1, 0, 1, 2]
    assert [item["slot"] for item in result["items"]] == [-1, 0, 1, 2]
    graph = flow.snapshot()
    nodes = {node["label"]: node for node in graph["nodes"]}
    call3_dependencies = set(nodes["CALL3"]["dependencies"])
    expected = ["CALL2-DETAIL", "MULTI-CHAR-MASK slot=-1", "MULTI-CHAR-MASK slot=0", "MULTI-CHAR-MASK slot=1"] if enable_masks else ["CALL2-DETAIL", "CALL2-KEYVIS"]
    assert {nodes[label]["id"] for label in expected}.issubset(call3_dependencies)
    assert "CALL2-AUDIT" not in nodes
    assert nodes["scene-1"]["dependencies"] == [nodes["MULTI-CHAR-MASK slot=1" if enable_masks else "CALL2-DETAIL"]["id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialogue_enabled", [True, False])
async def test_mocked_pipeline_passes_plan_brief_to_detail_and_saves_tracking(monkeypatch, dialogue_enabled):
    narrative = "Ari lifts a lantern.\n\nAri walks past the gate.\n\nAri puts it down."
    calls = []
    captured = {}
    responses = {
        "CALL1": json.dumps({"SLOT0": {"Ari": {"outfit": "old cloak", "hair": "ponytail"}},
                             "SLOT1": {"Ari": {"outfit": "new cape", "hair": "hair down"}}}),
        "CALL2-PLAN": json.dumps({"scene_plan": [
            {"anchor_segment": "C001", "characters": ["Ari"], "scene_brief": "lantern moment"},
            {"anchor_segment": "C002", "characters": ["Ari"], "scene_brief": "Ari passes the actual gate in profile"},
        ]}),

    }

    async def llm_stub(stage, messages, *_args, **_kwargs):
        calls.append(stage)
        content = messages[-1]["content"]
        captured.setdefault(stage, []).append(content)
        if stage == "CALL2-DETAIL":
            batch = json.JSONDecoder().raw_decode(content.split("# ONLY THIS BATCH'S PREPARED SCENES\n", 1)[1])[0]["scenes"]
            assert len(batch) == 1
            scene = batch[0]
            assert set(scene) == {"anchor_segment", "slot", "brief", "characters"}
            return _detail(scene["slot"], positive=f"scene {scene['slot']}, chosen tags")
        if stage == "CALL2-SELECT":
            return json.dumps({"scene_plan": [
                {"anchor_segment": scene["anchor_segment"], "characters": scene["characters"]}
                for scene in json.loads(responses["CALL2-PLAN"])["scene_plan"]]})
        if stage.startswith("CALL2-PLAN"):
            batch = json.JSONDecoder().raw_decode(content.split(
                "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)\n", 1)[1])[0]
            assert all(set(scene) == {"anchor_segment", "characters"} for scene in batch["scene_plan"])
            planned = {scene["anchor_segment"]: scene
                       for scene in json.loads(responses["CALL2-PLAN"])["scene_plan"]}
            return json.dumps({"scene_plan": [planned[scene["anchor_segment"]]
                                              for scene in batch["scene_plan"]]})
        return responses[stage]

    async def call3_stub(*_args, **_kwargs):
        captured["call3_calls"] = captured.get("call3_calls", 0) + 1
        return {"output": "dialogue-only"}

    async def plan_callback(data):
        captured["plan_callback"] = data

    async def ready_callback(data):
        captured.setdefault("ready_items", []).extend(deepcopy(data["items"]))

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_build_call3_dialogue_with_recovery", call3_stub)
    monkeypatch.setattr(pipeline, "parse_speak_output", lambda *_args, **_kwargs: {0: "first line", 1: "second line"})
    profiles = {"Ari": _profiles()["Ari"]}
    history = {"state_before": {"ari": {"canonical_name": "Ari", "call1_status": "old cloak, tied hair",
                                       "last_visual_positive": "irrelevant stale dress"}},
               "current_message_id": "turn-2"}
    result = await pipeline.build_from_context(
        {"session_id": "mock-session", "chats": [{"role": "char", "data": narrative}]},
        {"output_count_min": 2, "output_count_max": 2,
         "call2_parallel_enabled": True, "key_visual": False,
         "speak_enabled": dialogue_enabled},
        extra_reference="### Ari\n-Name\nAri\n-Appearance\nlegacy appearance",
        visual_profiles=profiles,
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}],
                                     "profile_events": [{"character": "Ari", "segment_id": "START", "profile_id": "card_1"},
                                                        {"character": "Ari", "segment_id": "C001", "profile_id": "card_2"}]},
        history_plan=history, on_call2_plan_ready=plan_callback, on_call2_ready=ready_callback,
        first_pass_single_v5=True,
    )
    assert calls == ["CALL1", "CALL2-SELECT", "CALL2-PLAN 1/2", "CALL2-PLAN 2/2",
                     "CALL2-DETAIL", "CALL2-DETAIL"]
    assert captured["CALL1"][0].count("old cloak, tied hair") >= 1
    assert "irrelevant stale dress" not in captured["CALL1"][0]
    assert "SELECTED PROFILE / LB.EXTRA BASELINE" in captured["CALL1"][0]
    assert "Ari coat 1" in captured["CALL1"][0].split("# SELECTED PROFILE / LB.EXTRA BASELINE", 1)[1].split("# PAST HISTORY", 1)[0]
    detail_payloads = [json.JSONDecoder().raw_decode(content.split(
        "# ONLY THIS BATCH'S PREPARED SCENES\n", 1)[1])[0]["scenes"][0]
        for content in captured["CALL2-DETAIL"]]
    by_slot = {item["slot"]: item for item in detail_payloads}
    assert by_slot[0]["brief"] == "lantern moment"
    assert by_slot[1]["brief"] == "Ari passes the actual gate in profile"
    assert by_slot[0]["characters"][0]["profile"]["id"] == "card_2"
    assert by_slot[0]["characters"][0]["state"] == {"outfit": "old cloak", "hair": "ponytail"}
    assert by_slot[1]["characters"][0]["state"] == {"outfit": "new cape", "hair": "hair down"}
    assert "###" not in captured["CALL2-DETAIL"][0]
    assert "legacy appearance" not in captured["CALL2-DETAIL"][0]
    assert captured["plan_callback"]["scene_slots"] == [0, 1]
    assert "anonymous_partner_fragment" not in result["call2_output"]
    assert [entry["kind"] for entry in result["items"]] == ["scene", "scene"]
    assert captured.get("call3_calls", 0) == int(dialogue_enabled)
    assert [entry["speak"] for entry in result["items"]] == (
        ["first line", "second line"] if dialogue_enabled else ["", ""])
    for slot, entry in enumerate(result["items"]):
        assert entry["characters"][0]["positive"] == f"scene {slot}, chosen tags"
        assert f"scene {slot}, chosen tags" in entry["raw_positive"]
        assert "old cloak" not in entry["raw_positive"]
        assert next(item for item in captured["ready_items"] if item["slot"] == slot)["speak"] == ""
    assert json.loads(result["character_states_after"]["ari"]["call1_status"]) == {"outfit": "new cape", "hair": "hair down"}
    assert [json.loads(value) for value in result["character_states_after"]["ari"]["call1_status_history"]] == [
        {"outfit": "old cloak", "hair": "ponytail"}, {"outfit": "new cape", "hair": "hair down"}]
    assert "wardrobe_events" not in result and "hairstyle_events" not in result
    from modes.illustration_chat_history import _safe_pipeline_snapshot
    assert "wardrobe_events" not in _safe_pipeline_snapshot(result)
    assert result["character_states_after"]["ari"]["last_seen_message_id"] == "turn-2"


@pytest.mark.asyncio
async def test_partial_call1_checkpoints_are_preserved_across_requests_without_detail_tags(monkeypatch):
    initial = {"outfit": "blue coat, buttoned coat", "hair": "ponytail"}
    opened = {"outfit": "blue coat, open coat", "hair": "hair down"}
    rolled = {"outfit": "blue coat, open coat, rolled up sleeves", "hair": "hair down"}
    starts = []
    details = []
    turn = 0

    async def llm_stub(stage, messages, *_args, **_kwargs):
        nonlocal turn
        content = messages[-1]["content"]
        if stage == "CALL1":
            starts.append(content)
            turn += 1
            statuses = [
                {"SLOT0": {"Ari": initial}, "SLOT1": {"Ari": opened}},
                {"SLOT0": {"Ari": opened}, "SLOT1": {"Ari": {"outfit": rolled["outfit"]}}},
                {"SLOT0": {"Ari": rolled}},
            ]
            return json.dumps(statuses[turn - 1])
        if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-DETAIL":
            details.extend(json.JSONDecoder().raw_decode(content.split(
                "# ONLY THIS BATCH'S PREPARED SCENES\n", 1)[1])[0]["scenes"])
            return _detail(positive="selected render-only garment tag")
        raise AssertionError(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    options = {"output_count_min": 1, "output_count_max": 1,
               "key_visual": False, "call3_enabled": False}
    profile_result = {"current_characters": [{"name": "Ari"}]}
    texts = ["Ari arrives.\n\nAri opens the coat and frees her hair.\n\nAri waits.",
             "Ari looks over.\n\nAri rolls up her sleeves.\n\nAri waits.",
             "Ari waits quietly.\n\nAri looks back.\n\nAri waves."]
    state_before = {}
    histories = []
    for index, text in enumerate(texts):
        result = await pipeline.build_from_context(
            {"session_id": f"partial-{index}", "chats": [{"role": "char", "data": text}]},
            options, "### Ari\n-Name\nAri\n-Appearance\nred hair",
            visual_profiles={"Ari": _profiles()["Ari"]},
            pre_resolved_profile_result=profile_result,
            history_plan={"state_before": state_before, "current_message_id": f"turn-{index}"},
        )
        histories.append(result["character_states_after"]["ari"]["call1_status_history"])
        state_before = deepcopy(result["character_states_after"])
        state_before["ari"]["call1_status"] = "poisoned legacy scalar"
        state_before["ari"]["last_visual_positive"] = "poisoned image descriptor"
        assert "wardrobe_events" not in result and "hairstyle_events" not in result
        assert "selected render-only garment tag" not in str(state_before["ari"]["call1_status_history"])

    first = [initial, opened]
    second = [opened, rolled]
    assert [[json.loads(status) for status in chain] for chain in histories] == [first, second, [rolled]]
    for index, expected in [(1, first), (2, second)]:
        saved = json.loads(starts[index].split(
            "# PREVIOUS SAVED CALL1 STATE (RAW CHRONOLOGICAL DETAIL MEMORY; APPLY NARRATIVE CHANGES)\n", 1)[1]
            .split("\n\n# SELECTED PROFILE / LB.EXTRA BASELINE", 1)[0])
        assert {name: [json.loads(status) for status in chain] for name, chain in saved.items()} == {"Ari": expected}
        assert "poisoned legacy scalar" not in starts[index]
        assert "poisoned image descriptor" not in starts[index]
    assert details[2]["characters"][0]["state"] == rolled
    assert details[0]["characters"][0]["profile"]["id"] == "card_1"


@pytest.mark.asyncio
async def test_plan_routed_llm_success_and_failure_emit_stream_and_one_terminal_history(monkeypatch):
    from modes import llm_service
    responses = iter(["{\"scene_plan\":[]}", "[LLM 실패] route unavailable"])
    task_keys = []
    records = []
    events = []

    async def fake_llm(task_key, _messages, **_kwargs):
        task_keys.append(task_key)
        return next(responses)

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "callLLMTask", fake_llm)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", lambda record: records.append(dict(record)))
    assert await pipeline._call_pipeline_llm("CALL2-PLAN", [{"role": "user", "content": "mocked"}], notify) == '{"scene_plan":[]}'
    with pytest.raises(RuntimeError, match="route unavailable"):
        await pipeline._call_pipeline_llm("CALL2-PLAN", [{"role": "user", "content": "mocked"}], notify)
    assert task_keys == ["illustration_call2_plan", "illustration_call2_plan"]
    assert [(record["call_name"], record["status"]) for record in records] == [
        ("CALL2-PLAN", "ok"), ("CALL2-PLAN", "error"),
    ]
    assert {event["type"] for event in events} == {"start", "done", "error"}
    assert all(event["queue_subtask"]["group_id"] == "call2_plan" for event in events)


@pytest.mark.asyncio
async def test_independent_key_visual_receives_resolved_card_timeline_and_registers_first(monkeypatch):
    narrative = "Ari finds a candle.\n\nAri turns toward the window."
    stages = []
    keyvis_messages = []
    callbacks = []
    keyvis_registered = asyncio.Event()

    async def llm_stub(stage, messages, *_args, **_kwargs):
        stages.append(stage)
        if stage == "CALL1":
            return json.dumps({"SLOT0": {"Ari": {"outfit": "white shirt", "hair": "hair down"}}})
        if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-DETAIL":
            await asyncio.wait_for(keyvis_registered.wait(), 3)
            assert callbacks == ["keyvis"]
            return _detail()
        raise AssertionError(stage)

    async def keyvis_stub(*, call2_context_messages, allowed_character_names,
                          toggles, stream_notify):
        stages.append("CALL2-KEYVIS")
        keyvis_messages.extend(call2_context_messages)
        assert allowed_character_names == ["Ari"]
        return ({"kind": "keyvis", "slot": -1, "camera": "cover portrait",
                 "characters": [{"name": "Ari", "positive": "illustrated candle",
                                 "negative": "", "position": "center"}],
                 "scene": "promotional candle", "supplement": ""}, "independent keyvis output")

    async def keyvis_callback(data):
        callbacks.append("keyvis")
        keyvis_registered.set()
        assert data["total_count"] == 2
        assert data["items"][0]["kind"] == "keyvis"
        assert "illustrated candle" in data["items"][0]["raw_positive"]

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_run_call2_keyvis", keyvis_stub)
    result = await pipeline.build_from_context(
        {"session_id": "mock-keyvis", "chats": [{"role": "char", "data": narrative}]},
        {"output_count_min": 1, "output_count_max": 1, "key_visual": True,
         "call3_enabled": False},
        "### Ari\n-Name\nAri\n-Appearance\nlegacy",
        visual_profiles={"Ari": _profiles()["Ari"]},
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}],
            "profile_events": [{"character": "Ari", "segment_id": "START", "profile_id": "card_1"},
                               {"character": "Ari", "segment_id": "C002", "profile_id": "card_2"}]},
        on_keyvis_ready=keyvis_callback,
    )
    assert stages[:2] == ["CALL1", "CALL2-SELECT"]
    # Independent DETAIL/KEYVIS tasks may enter in either order; the callback
    # assertions above still require KEYVIS to register before DETAIL completes.
    assert sorted(stages[2:]) == ["CALL2-DETAIL", "CALL2-KEYVIS", "CALL2-PLAN"]
    assert result["call2_keyvis_output"] == "independent keyvis output"
    assert [entry["kind"] for entry in result["items"]] == ["keyvis", "scene"]
    authority = next(message["content"] for message in keyvis_messages
        if "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT" in message["content"])
    assert '"from_segment": "START"' in authority
    assert '"from_segment": "C002"' in authority
    assert "Ari coat 1" in authority and "Ari coat 2" in authority
    assert "white shirt" in authority and "hair down" in authority


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked_history", [True, False])
async def test_key_visual_dictionary_bounded_only_on_tracked_history_route(monkeypatch, tracked_history):
    keyvis_messages = []

    async def llm_stub(stage, _messages, *_args, **_kwargs):
        if stage == "CALL1":
            return json.dumps({"SLOT0": {"Ari": {"outfit": "linen coat", "hair": "ponytail"}}})
        if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-DETAIL":
            return _detail()
        raise AssertionError(stage)

    async def keyvis_stub(*, call2_context_messages, allowed_character_names,
                          toggles, stream_notify):
        keyvis_messages.extend(call2_context_messages)
        assert allowed_character_names == ["Ari"]
        return ({"kind": "keyvis", "slot": -1, "camera": "portrait",
                 "characters": [{"name": "Ari", "positive": "linen coat",
                                 "negative": "", "position": "center"}],
                 "scene": "Ari by a window", "supplement": ""}, "keyvis mock")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_run_call2_keyvis", keyvis_stub)
    await pipeline.build_from_context(
        {"session_id": "keyvis-dictionary", "chats": [{"role": "char", "data":
            "Ari reaches the window.\n\nAri opens it."}]},
        {"call1_enabled": True, "key_visual": True, "output_count_min": 1,
         "output_count_max": 1, "call3_enabled": False},
        "### Ari\n-Appearance\nAri eyes\n\n### Bob\n-Appearance\nBob coat",
        visual_profiles={"Ari": _profiles()["Ari"]},
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}]},
        history_plan={"state_before": {}, "current_message_id": "turn-keyvis"} if tracked_history else None,
    )
    dictionary = next(message["content"] for message in keyvis_messages
                      if message["content"].startswith("# CHARACTER DICTIONARY"))
    assert "### Ari\n-Appearance\nAri eyes" in dictionary
    assert ("### Bob" in dictionary) is not tracked_history
    authority = next(message["content"] for message in keyvis_messages
                     if message["content"].startswith("# AUTHORITATIVE FIXED APPEARANCE"))
    assert "Ari coat 1" in authority and "linen coat" in authority and "ponytail" in authority
    assert "Bob coat" not in authority


@pytest.mark.asyncio
@pytest.mark.parametrize("layout_case", ["ready", "missing", "error"])
async def test_multi_character_keyvis_only_enqueues_early_with_layout(
    monkeypatch, capsys, layout_case,
):
    early = []
    ordinary_ready = []

    async def llm_stub(stage, _messages, *_args, **_kwargs):
        if stage == "CALL1":
            return json.dumps({"SLOT0": {"Ari": {"outfit": "blue coat", "hair": "ponytail"},
                                          "Bao": {"outfit": "brown jacket", "hair": "short hair"}}})
        if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-DETAIL":
            return _detail()
        raise AssertionError(stage)

    async def keyvis_stub(*, call2_context_messages, allowed_character_names,
                          toggles, stream_notify):
        assert allowed_character_names == ["Ari", "Bao"]
        return ({"kind": "keyvis", "slot": -1, "camera": "wide portrait",
                 "characters": [
                     {"name": name, "positive": f"{name} portrait", "negative": "", "position": "center"}
                     for name in ("Ari", "Bao")],
                 "scene": "two people by a window", "supplement": ""}, "keyvis mock")

    async def layout_stub(descriptors, *_args, **_kwargs):
        if descriptors[0]["kind"] == "keyvis":
            if layout_case == "ready":
                descriptors[0]["multi_char_layout"] = {"character_order": ["Ari", "Bao"]}
            elif layout_case == "error":
                descriptors[0]["multi_char_layout_error"] = "mask unavailable"
            # A missing layout without an explicit error is also a failed slot.

    async def early_callback(data):
        descriptor = data["items"][0]
        if not isinstance(descriptor.get("multi_char_layout"), dict):
            raise RuntimeError("server would reject layout-free Key Visual early registration")
        early.append(deepcopy(data))

    async def ordinary_callback(data):
        ordinary_ready.append(deepcopy(data))

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_run_call2_keyvis", keyvis_stub)
    monkeypatch.setattr(pipeline, "calculate_multi_char_layouts", layout_stub)
    result = await pipeline.build_from_context(
        {"session_id": f"keyvis-{layout_case}", "chats": [{"role": "char", "data":
            "Ari and Bao enter.\n\nAri opens a book."}]},
        {"key_visual": True, "output_count_min": 1, "output_count_max": 1,
         "call3_enabled": False},
        "### Ari\n-Name\nAri\n### Bao\n-Name\nBao",
        visual_profiles=_profiles(),
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}, {"name": "Bao"}]},
        enable_multi_char_layout=True,
        on_keyvis_ready=early_callback, on_call2_ready=ordinary_callback,
    )
    assert [item["kind"] for item in result["items"]] == ["keyvis", "scene"]
    assert len(ordinary_ready) == 1
    assert len(early) == (1 if layout_case == "ready" else 0)
    if layout_case != "ready":
        assert "조기 이미지 등록 생략" in capsys.readouterr().out
        assert ("mask unavailable" if layout_case == "error" else "layout missing") in str(
            result["items"][0].get("multi_char_layout_error") or "layout missing")


@pytest.mark.asyncio
async def test_keyvis_early_and_ordinary_registration_share_backtranslated_context(monkeypatch):
    narrative = "Ari keeps a candle.\n\nAri sets the candle down."
    captured = {}

    async def translate_stub(slotted, *_args, **_kwargs):
        translated = (slotted.replace("Ari keeps a candle.", "Ari carries a luminous candle.")
                      .replace("Ari sets the candle down.", "Ari leaves it by the door."))
        return translated, [{"status": "translated"}]

    async def llm_stub(stage, _messages, *_args, **_kwargs):
        if stage == "CALL1":
            return json.dumps({"SLOT0": {"Ari": {"outfit": "linen coat", "hair": "ponytail"}}})
        if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-DETAIL":
            return _detail()
        raise AssertionError(stage)

    async def keyvis_stub(**_kwargs):
        return ({"kind": "keyvis", "slot": -1, "camera": "portrait",
                 "characters": [{"name": "Ari", "positive": "Ari and a candle",
                                 "negative": "", "position": "center"}],
                 "scene": "Ari holds a candle", "supplement": ""}, "keyvis mock")

    async def early_callback(data):
        captured["early"] = deepcopy(data)

    async def ordinary_callback(data):
        captured["ordinary"] = deepcopy(data)

    monkeypatch.setattr(pipeline, "backtranslate_current_context", translate_stub)
    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    monkeypatch.setattr(pipeline, "_run_call2_keyvis", keyvis_stub)
    result = await pipeline.build_from_context(
        {"session_id": "backtranslated-keyvis", "chats": [{"role": "char", "data": narrative}]},
        {"call1_backtranslate_enabled": True, "key_visual": True,
         "output_count_min": 1, "output_count_max": 1, "call3_enabled": False},
        "### Ari\n-Name\nAri",
        visual_profiles={"Ari": _profiles()["Ari"]},
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}]},
        on_keyvis_ready=early_callback, on_call2_ready=ordinary_callback,
    )
    assert captured["early"]["context"] == captured["ordinary"]["context"] == result["context"]
    assert "Ari carries a luminous candle." in result["context"]
    assert "Ari keeps a candle." not in result["context"]
    assert captured["early"]["items"][0]["kind"] == "keyvis"


def test_renderer_can_bind_identity_without_reintroducing_absolute_card_tags():
    from modes.illust_prompt_builder import IllustPromptBuilder

    raw = "[Name]\nAri\n[SETUP]\nmedium shot\n[CHAR]\nchosen face and hairstyle\n[SUPPLEMENT]\nlantern"
    lb_extra = [{"name": "Ari", "appearance": [{"tag": "old eye detail"}],
                 "outfit": [{"tag": "old dress"}]}]
    card = {"name": "Ari", "absolute_tags": "fixed eye detail, default coat"}
    legacy = IllustPromptBuilder.parse_sections(raw, lb_extra=lb_extra, characters=[card])
    selected = IllustPromptBuilder.parse_sections(raw, lb_extra=lb_extra,
        characters=[{**card, "absolute_tags": ""}])
    assert "fixed eye detail" in legacy["char"]
    assert "default coat" in legacy["char"]
    assert "chosen face and hairstyle" in selected["char"]
    assert "fixed eye detail" not in selected["char"]
    assert "default coat" not in selected["char"]
    assert "old dress" not in selected["char"]


def test_detail_negative_keeps_configured_single_v5_preset_without_fragment_flag():
    from modes.illust_prompt_builder import (
        IllustPromptBuilder, allows_anonymous_fragment_negative_relaxation,
    )
    from server import _append_authoritative_scene_negative

    bot = {"system_prompt_preset": "배포_1차 싱글 V5"}
    assert allows_anonymous_fragment_negative_relaxation(bot, False) is False
    configured = IllustPromptBuilder.build_negative_prompt(
        {"negative_presets": {"base": ["extra person", "bad anatomy"]}},
        {"anima_negative_preset": "base", "sdxl_negative_preset": "base",
         "chansub_workflow_type": "anima"},
        ["Ari"], bot, allow_anonymous_fragment=False,
    )
    rendered = _append_authoritative_scene_negative(
        configured, "DETAIL unwanted artifact", provider="comfy", prompt_format="v3",
    )
    anima, sdxl = rendered.split("\n[SDXL]\n")
    for section in (anima, sdxl):
        assert "extra person" in section
        assert "DETAIL unwanted artifact" in section
