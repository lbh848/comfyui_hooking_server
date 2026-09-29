"""Offline contracts for the ordinary PLAN -> AUDIT -> DETAIL path."""

import asyncio
from copy import deepcopy
import json

import pytest

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
        "state_at_start": [
            {"character": "Ari", "status": "  OLD blue coat and tied hair  "},
            {"character": "Bao", "status": "Bao brown jacket and short hair"},
        ],
        "events": [
            {"segment_id": "C020", "character": "Ari", "status": "blue coat open; hair tied"},
            {"segment_id": "C044", "character": "Bao", "status": "brown jacket removed; short hair"},
            {"segment_id": "C070", "character": "Ari", "status": "coat removed; hair tied"},
            {"segment_id": "C070", "character": "Ari", "status": "coat removed; hair loose"},
            {"segment_id": "C080", "character": "Ari", "status": "green coat; loose hair"},
        ],
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
    assert ari["state"] == ["  OLD blue coat and tied hair  ", "blue coat open; hair tied"]
    assert ari["anchor_event"] == ["coat removed; hair tied", "coat removed; hair loose"]
    assert "green coat" not in str(at_anchor)
    assert tracking["state_at_start"][0]["status"] == "  OLD blue coat and tied hair  "


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


def test_call1_requires_exact_status_contract_and_preserves_same_segment_order():
    tracking = _tracking()
    names = ["Ari", "Bao"]
    assert scene_pipeline.parse_state(json.dumps(tracking), names, _segments()) == tracking
    for invalid in [
        {**tracking, "wardrobe_events": []},
        {**tracking, "events": list(reversed(tracking["events"]))},
        {**tracking, "events": tracking["events"] + [{"segment_id": "C090", "character": "Someone", "status": "no"}]},
        {**tracking, "state_at_start": [tracking["state_at_start"][0]]},
    ]:
        with pytest.raises(ValueError):
            scene_pipeline.parse_state(json.dumps(invalid), names, _segments())


def test_plan_audit_pass_and_replacement_are_exact_and_reject_structural_changes():
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
    response = [{"anchor_segment": "C001", "brief": "PASS"},
                {"anchor_segment": "C002", "brief": "Repaired full scene and camera"}]
    audited = scene_pipeline.parse_audit(json.dumps({"audited_scene_plan": response}), original)
    assert audited[0]["scene_brief"] == "Keep  two spaces."
    assert audited[1]["scene_brief"] == "Repaired full scene and camera"
    assert original[1]["scene_brief"] == "Incorrect pose"
    for invalid in [response[:1], list(reversed(response)),
                    [response[0], {**response[1], "characters": ["Bao"]}],
                    [response[0], {**response[1], "slot": 1}]]:
        with pytest.raises(ValueError):
            scene_pipeline.parse_audit(json.dumps({"audited_scene_plan": invalid}), original)
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
async def test_mocked_pipeline_audits_before_per_scene_detail_and_saves_tracking(monkeypatch):
    narrative = "Ari lifts a lantern.\n\nAri walks past the gate.\n\nAri puts it down."
    calls = []
    captured = {}
    responses = {
        "CALL1": json.dumps({"state_at_start": [{"character": "Ari", "status": "old cloak, tied hair"}],
                             "events": [{"segment_id": "C002", "character": "Ari", "status": "new cape, loose hair"}]}),
        "CALL2-PLAN": json.dumps({"scene_plan": [
            {"anchor_segment": "C001", "characters": ["Ari"], "scene_brief": "lantern moment"},
            {"anchor_segment": "C002", "characters": ["Ari"], "scene_brief": "wrong gate detail"},
        ]}),
        "CALL2-AUDIT": json.dumps({"audited_scene_plan": [
            {"anchor_segment": "C001", "brief": "PASS"},
            {"anchor_segment": "C002", "brief": "Ari passes the actual gate in profile"},
        ]}),
    }

    async def llm_stub(stage, messages, *_args, **_kwargs):
        calls.append(stage)
        content = messages[-1]["content"]
        captured.setdefault(stage, []).append(content)
        if stage == "CALL2-DETAIL":
            scene = json.loads(content.split("# ONLY THIS SCENE'S PREPARED DATA\n", 1)[1].split("\n\nReturn only", 1)[0])
            assert set(scene) == {"anchor_segment", "slot", "brief", "characters"}
            return _detail(scene["slot"], positive=f"scene {scene['slot']}, chosen tags")
        return responses[stage]

    async def call3_stub(*_args, **_kwargs):
        return {"output": "dialogue-only"}

    async def plan_callback(data):
        captured["plan_callback"] = data

    async def ready_callback(data):
        captured["ready_callback"] = deepcopy(data)

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
         "call3_enabled": True, "speak_enabled": True},
        extra_reference="### Ari\n-Name\nAri\n-Appearance\nlegacy appearance",
        visual_profiles=profiles,
        pre_resolved_profile_result={"current_characters": [{"name": "Ari"}],
                                     "profile_events": [{"character": "Ari", "segment_id": "START", "profile_id": "card_1"},
                                                        {"character": "Ari", "segment_id": "C001", "profile_id": "card_2"}]},
        history_plan=history, on_call2_plan_ready=plan_callback, on_call2_ready=ready_callback,
        first_pass_single_v5=True,
    )
    assert calls == ["CALL1", "CALL2-PLAN", "CALL2-AUDIT", "CALL2-DETAIL", "CALL2-DETAIL"]
    assert captured["CALL1"][0].count("old cloak, tied hair") >= 1
    assert "irrelevant stale dress" not in captured["CALL1"][0]
    assert "SELECTED PROFILE / LB.EXTRA BASELINE" in captured["CALL1"][0]
    assert captured["CALL1"][0].split("# SELECTED PROFILE / LB.EXTRA BASELINE", 1)[1].split("# PAST HISTORY", 1)[0].strip().endswith("{}")
    detail_payloads = [json.loads(content.split("# ONLY THIS SCENE'S PREPARED DATA\n", 1)[1].split("\n\nReturn only", 1)[0]) for content in captured["CALL2-DETAIL"]]
    by_slot = {item["slot"]: item for item in detail_payloads}
    assert by_slot[0]["brief"] == "lantern moment"
    assert by_slot[1]["brief"] == "Ari passes the actual gate in profile"
    assert by_slot[0]["characters"][0]["profile"]["id"] == "card_2"
    assert by_slot[0]["characters"][0]["state"] == ["old cloak, tied hair"]
    assert by_slot[1]["characters"][0]["anchor_event"] == ["new cape, loose hair"]
    assert "###" not in captured["CALL2-DETAIL"][0]
    assert "legacy appearance" not in captured["CALL2-DETAIL"][0]
    assert captured["plan_callback"]["scene_slots"] == [0, 1]
    assert "anonymous_partner_fragment" not in result["call2_output"]
    assert [entry["kind"] for entry in result["items"]] == ["scene", "scene"]
    assert [entry["speak"] for entry in result["items"]] == ["first line", "second line"]
    for slot, entry in enumerate(result["items"]):
        assert entry["characters"][0]["positive"] == f"scene {slot}, chosen tags"
        assert f"scene {slot}, chosen tags" in entry["raw_positive"]
        assert "old cloak" not in entry["raw_positive"]
        assert captured["ready_callback"]["items"][slot]["speak"] == ""
    assert result["character_states_after"]["ari"]["call1_status"] == "new cape, loose hair"
    assert result["character_states_after"]["ari"]["call1_status_history"] == [
        "old cloak, tied hair", "old cloak, tied hair", "new cape, loose hair",
    ]
    assert "wardrobe_events" not in result and "hairstyle_events" not in result
    from modes.illustration_chat_history import _safe_pipeline_snapshot
    assert "wardrobe_events" not in _safe_pipeline_snapshot(result)
    assert result["character_states_after"]["ari"]["last_seen_message_id"] == "turn-2"


@pytest.mark.asyncio
async def test_partial_call1_deltas_are_preserved_across_requests_without_detail_tags(monkeypatch):
    initial = "blue coat buttoned; hair tied"
    coat_delta = "coat open; the rest unchanged"
    hair_delta = "hair loose; the outfit unchanged"
    sleeve_delta = "sleeves rolled; coat and hair otherwise unchanged"
    starts = []
    audits = []
    details = []
    turn = 0

    async def llm_stub(stage, messages, *_args, **_kwargs):
        nonlocal turn
        content = messages[-1]["content"]
        if stage == "CALL1":
            starts.append(content)
            turn += 1
            statuses = [
                {"state_at_start": [{"character": "Ari", "status": initial}],
                 "events": [{"segment_id": "C002", "character": "Ari", "status": coat_delta},
                            {"segment_id": "C002", "character": "Ari", "status": hair_delta}]},
                {"state_at_start": [{"character": "Ari", "status": "blue coat open; hair loose"}],
                 "events": [{"segment_id": "C001", "character": "Ari", "status": sleeve_delta}]},
                {"state_at_start": [{"character": "Ari", "status": "blue coat open with rolled sleeves; hair loose"}],
                 "events": []},
            ]
            return json.dumps(statuses[turn - 1])
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            audits.append(content)
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
        if stage == "CALL2-DETAIL":
            details.append(json.loads(content.split("# ONLY THIS SCENE'S PREPARED DATA\n", 1)[1]
                                      .split("\n\nReturn only", 1)[0]))
            return _detail(positive="selected render-only garment tag")
        raise AssertionError(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    options = {"output_count_min": 1, "output_count_max": 1,
               "key_visual": False, "call3_enabled": False}
    profile_result = {"current_characters": [{"name": "Ari"}]}
    texts = ["Ari arrives.\n\nAri opens the coat and frees her hair.",
             "Ari rolls up her sleeves.\n\nAri looks over.",
             "Ari waits quietly.\n\nAri looks back."]
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

    first = [initial, coat_delta, hair_delta]
    second = first + ["blue coat open; hair loose", sleeve_delta]
    assert histories == [first, second, second + ["blue coat open with rolled sleeves; hair loose"]]
    for index, expected in [(1, first), (2, second)]:
        saved = json.loads(starts[index].split(
            "# PREVIOUS SAVED CALL1 STATE (RAW CHRONOLOGICAL STATUSES; ONLY TRACKING AUTHORITY)\n", 1)[1]
            .split("\n\n# SELECTED PROFILE / LB.EXTRA BASELINE", 1)[0])
        assert saved == {"Ari": expected}
        assert "poisoned legacy scalar" not in starts[index]
        assert "poisoned image descriptor" not in starts[index]
        assert '"previous_state": {"Ari": ' in audits[index]
        assert coat_delta in audits[index] and hair_delta in audits[index]
    assert details[2]["characters"][0]["state"] == ["blue coat open with rolled sleeves; hair loose"]
    assert details[0]["characters"][0]["profile"]["id"] == "card_1"


@pytest.mark.asyncio
async def test_audit_routed_llm_success_and_failure_emit_stream_and_one_terminal_history(monkeypatch):
    from modes import llm_service
    responses = iter(["{\"audited_scene_plan\":[]}", "[LLM 실패] route unavailable"])
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
    assert await pipeline._call_pipeline_llm("CALL2-AUDIT", [{"role": "user", "content": "mocked"}], notify) == '{"audited_scene_plan":[]}'
    with pytest.raises(RuntimeError, match="route unavailable"):
        await pipeline._call_pipeline_llm("CALL2-AUDIT", [{"role": "user", "content": "mocked"}], notify)
    assert task_keys == ["illustration_call2_plan", "illustration_call2_plan"]
    assert [(record["call_name"], record["status"]) for record in records] == [
        ("CALL2-AUDIT", "ok"), ("CALL2-AUDIT", "error"),
    ]
    assert {event["type"] for event in events} == {"start", "done", "error"}
    assert all(event["queue_subtask"]["group_id"] == "call2_audit" for event in events)


@pytest.mark.asyncio
async def test_independent_key_visual_receives_resolved_card_timeline_and_registers_first(monkeypatch):
    narrative = "Ari finds a candle.\n\nAri turns toward the window."
    stages = []
    keyvis_messages = []
    callbacks = []

    async def llm_stub(stage, messages, *_args, **_kwargs):
        stages.append(stage)
        if stage == "CALL1":
            return json.dumps({"state_at_start": [{"character": "Ari", "status": "linen coat and tied hair"}],
                               "events": [{"segment_id": "C002", "character": "Ari",
                                           "status": "coat carried, hair loose"}]})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
        if stage == "CALL2-DETAIL":
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
    assert stages == ["CALL1", "CALL2-PLAN", "CALL2-AUDIT", "CALL2-KEYVIS", "CALL2-DETAIL"]
    assert result["call2_keyvis_output"] == "independent keyvis output"
    assert [entry["kind"] for entry in result["items"]] == ["keyvis", "scene"]
    authority = next(message["content"] for message in keyvis_messages
        if "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT" in message["content"])
    assert '"from_segment": "START"' in authority
    assert '"from_segment": "C002"' in authority
    assert "Ari coat 1" in authority and "Ari coat 2" in authority
    assert "coat carried, hair loose" in authority


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked_history", [True, False])
async def test_key_visual_dictionary_bounded_only_on_tracked_history_route(monkeypatch, tracked_history):
    keyvis_messages = []

    async def llm_stub(stage, _messages, *_args, **_kwargs):
        if stage == "CALL1":
            return json.dumps({"state_at_start": [
                {"character": "Ari", "status": "linen coat; tied hair"}], "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
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
    assert "Ari coat 1" in authority and "linen coat; tied hair" in authority
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
            return json.dumps({"state_at_start": [
                {"character": "Ari", "status": "blue coat; hair tied"},
                {"character": "Bao", "status": "brown jacket; short hair"},
            ], "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
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
            return json.dumps({"state_at_start": [
                {"character": "Ari", "status": "linen coat; hair tied"}], "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C001", "brief": "PASS"}]})
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


@pytest.mark.asyncio
async def test_invalid_audit_aborts_before_detail_without_reusing_plan(monkeypatch):
    stages = []

    async def llm_stub(stage, _messages, *_args, **_kwargs):
        stages.append(stage)
        if stage == "CALL1":
            return json.dumps({"state_at_start": [{"character": "Ari", "status": "outfit and hair unchanged"}],
                               "events": []})
        if stage == "CALL2-PLAN":
            return json.dumps({"scene_plan": [_scene("C001")]})
        if stage == "CALL2-AUDIT":
            return json.dumps({"audited_scene_plan": [{"anchor_segment": "C999", "brief": "PASS"}]})
        raise AssertionError(f"Unexpected fallback/detail stage: {stage}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    with pytest.raises(ValueError, match="anchor"):
        await pipeline.build_from_context(
            {"session_id": "mock-invalid", "chats": [
                {"role": "char", "data": "Ari moves.\n\nAri sits."}]},
            {"output_count_min": 1, "output_count_max": 1, "key_visual": False},
            "### Ari\n-Name\nAri\n-Appearance\nlegacy",
            visual_profiles={"Ari": _profiles()["Ari"]},
            pre_resolved_profile_result={"current_characters": [{"name": "Ari"}]},
        )
    assert stages == ["CALL1", "CALL2-PLAN", "CALL2-AUDIT"]
