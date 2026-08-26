import asyncio
import json
from types import SimpleNamespace

import pytest

import server
from modes import llm_prompt_edit, multi_char_mask
from modes.illust_prompt_builder import IllustPromptBuilder


def _character(name: str, trigger: str, *, group: bool = False) -> dict:
    lora = {
        "lora_path": f"{name.lower()}-body.safetensors",
        "trigger": trigger,
        "strength": 0.75,
        "BASE": "anima",
    }
    return {
        "name": name,
        "gender_tag": "1girl",
        "face_tags": f"{name.lower()} hair, small nose",
        "eye_tags": f"{name.lower()} eyes",
        "character_negative": f"{name.lower()}-negative",
        "loras_solo": [] if group else [lora],
        "loras_group": [lora] if group else [],
        "face_loras": [{
            "lora_path": f"{name.lower()}-face.safetensors",
            "trigger": f"{name.lower()}_face",
            "strength": 0.6,
            "BASE": "anima",
        }],
        "style_loras": [{
            "lora_path": f"{name.lower()}-style.safetensors",
            "trigger": f"{name.lower()}_style",
            "strength": 0.5,
            "BASE": "anima",
        }],
    }


def _bot(*characters: dict) -> dict:
    settings = {
        "face_id_activate": True,
        "face_id_str": 0.61,
        "face_lora_upscale_size": "1024",
        "seed": 123,
    }
    return {
        "name": "Active Bot",
        "characters": list(characters),
        "illust_settings_solo": dict(settings),
        "illust_settings_group": dict(settings),
    }


def _tags() -> dict:
    return {
        "artist_presets": {},
        "quality_presets": {},
        "negative_presets": {},
        "anima_quality": ["quality"],
        "quality": ["sdxl quality"],
        "anima_negative": ["bad quality"],
        "negative": ["low quality"],
    }


def _json_block(positive: str, name: str) -> dict:
    return json.loads(llm_prompt_edit.parse_blocks(positive)[name])


def _layout(names: list[str]) -> dict:
    return {
        "background_prompt": "rooftop",
        "composition_prompt": "two people standing apart",
        "regions": [
            {
                "name": names[0],
                "character_prompt": f"{names[0]}, standing",
                "x": 0.0,
                "y": 0.0,
                "width": 0.55,
                "height": 1.0,
            },
            {
                "name": names[1],
                "character_prompt": f"{names[1]}, sitting",
                "x": 0.45,
                "y": 0.0,
                "width": 0.55,
                "height": 1.0,
            },
        ],
    }


def _write_backup(
    root,
    name: str,
    positive: str,
    *,
    info: dict,
    negative: str = "source negative",
) -> None:
    workflow = {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [positive]},
            {"title": "부정프롬프트", "widgets_values": [negative]},
        ]
    }
    (root / f"{name}.json").write_text(
        json.dumps(workflow, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / f"{name}_info.json").write_text(
        json.dumps(info, ensure_ascii=False),
        encoding="utf-8",
    )


def test_single_character_identity_rebuild_updates_face_lora_cache_and_negative():
    old = _character("Old", "old_trigger")
    new = _character("New", "new_trigger")
    bot = _bot(old, new)
    builder = IllustPromptBuilder()
    original = builder.build_positive_prompt(
        "daytime street",
        "Old, standing, school uniform",
        "looking at viewer",
        ["Old"],
        bot,
        _tags(),
        bot["illust_settings_solo"],
        bot["name"],
    )
    parsed = llm_prompt_edit.bind_scene_characters({
        "plan": "캐릭터 교체",
        "scene_setup": "night street",
        "scene_char": "smiling, school uniform",
        "scene_supplement": "looking at viewer",
    }, ["New"])
    blocks = llm_prompt_edit.parse_blocks(original)
    trigger_data = llm_prompt_edit.collect_character_triggers(bot, ["New"])
    reassembled, scene = llm_prompt_edit.reassemble(
        original,
        blocks,
        {
            "anima": trigger_data["anima_ordered"],
            "sdxl": trigger_data["sdxl_ordered"],
        },
        parsed,
    )
    rebuilt, negative = llm_prompt_edit.rebuild_v3_character_identity(
        reassembled,
        "old negative",
        bot_name=bot["name"],
        bot=bot,
        bot_root={"positive_whitelist": [], "positive_blacklist": []},
        tags=_tags(),
        character_names=["New"],
        scene=scene,
    )

    llm_prompt_edit.validate_v3_character_identity(rebuilt, ["New"])
    assert llm_prompt_edit.parse_blocks(rebuilt)["CHAR_LIST"] == "New"
    assert "new_trigger" in llm_prompt_edit.parse_blocks(rebuilt)["ANIMA_CONTENT"]
    assert "old_trigger" not in llm_prompt_edit.parse_blocks(rebuilt)["ANIMA_CONTENT"]
    assert _json_block(rebuilt, "CACHE_PATH")["list"][0]["CHAR"] == "New"
    assert _json_block(rebuilt, "FACE_ID_DIR")["list"] == [{
        "ipa_path": "soya_bot/Active Bot/New/cache.ipadpt",
        "str": 0.61,
        "CHAR": "New",
    }]
    assert _json_block(rebuilt, "LORA_DATA")["list"][0]["CHAR"] == "New"
    assert "new-body.safetensors" in _json_block(rebuilt, "LORA_DATA")["list"][0]["lora_path"]
    assert _json_block(rebuilt, "FACE_LORA_DATA")["list"][0]["CHAR"] == "New"
    assert _json_block(rebuilt, "STYLE_LORA_DATA")["list"][0]["lora_path"].endswith(
        "new-style.safetensors"
    )
    face_info = _json_block(rebuilt, "CHAR_FACE_TAG_INFORM")["list"][0]
    assert face_info["CHAR"] == "New"
    assert "new hair" in face_info["FACE_TAGS"]
    assert "new-negative" in negative
    assert "old-negative" not in negative


def test_identity_capability_allows_two_character_backup_with_valid_snapshot(
    tmp_path,
    monkeypatch,
):
    names = ["Old Left", "Old Right"]
    bot = _bot(*[
        _character(name, name.lower().replace(" ", "_"), group=True)
        for name in names
    ])
    snapshot = multi_char_mask.normalize_multi_char_snapshot({
        "enable": True,
        "character_order": names,
        "layout": _layout(names),
        "mask_location": "region_mask",
    })
    positive = IllustPromptBuilder().build_positive_prompt(
        "rooftop",
        "Old Left, standing | Old Right, sitting",
        "two people standing apart",
        names,
        bot,
        _tags(),
        bot["illust_settings_group"],
        bot["name"],
        multi_char_context={
            "enable": True,
            "char_name_list": names,
            "char_inform": ["Old Left, standing", "Old Right, sitting"],
            "background_prompt": "rooftop",
            "composition_prompt": "two people standing apart",
            "mask_fingerprint": snapshot["mask_fingerprint"],
        },
    )
    backup_name = "identity-capability-two-valid"
    _write_backup(
        tmp_path,
        backup_name,
        positive,
        info={
            "provider": "comfy",
            "bot_name": bot["name"],
            "illustration_multi_char": snapshot,
        },
    )
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))

    capability = server._llm_edit_identity_capability(backup_name)

    assert capability["enabled"] is True
    assert capability["character_names"] == names
    assert capability["requires_multi_char_snapshot"] is True


def test_two_character_identity_rebuild_remaps_fixed_mask_by_slot():
    old_names = ["Old Left", "Old Right"]
    new_names = ["New Left", "New Right"]
    old_chars = [_character(name, name.lower().replace(" ", "_"), group=True) for name in old_names]
    new_chars = [_character(name, name.lower().replace(" ", "_"), group=True) for name in new_names]
    bot = _bot(*(old_chars + new_chars))
    snapshot = multi_char_mask.normalize_multi_char_snapshot({
        "enable": True,
        "character_order": old_names,
        "layout": _layout(old_names),
        "mask_location": "region_mask",
    })
    remapped = multi_char_mask.remap_multi_char_snapshot(snapshot, new_names)
    builder = IllustPromptBuilder()
    original = builder.build_positive_prompt(
        "rooftop",
        "Old Left, standing | Old Right, sitting",
        "two people standing apart",
        old_names,
        bot,
        _tags(),
        bot["illust_settings_group"],
        bot["name"],
        multi_char_context={
            "enable": True,
            "char_name_list": old_names,
            "char_inform": ["Old Left, standing", "Old Right, sitting"],
            "background_prompt": "rooftop",
            "composition_prompt": "two people standing apart",
            "mask_fingerprint": snapshot["mask_fingerprint"],
        },
    )
    parsed = llm_prompt_edit.bind_scene_characters({
        "plan": "두 캐릭터 교체",
        "scene_setup": "sunset rooftop",
        "scene_char": "waving | laughing",
        "scene_supplement": "two people facing each other",
    }, new_names)
    blocks = llm_prompt_edit.parse_blocks(original)
    trigger_data = llm_prompt_edit.collect_character_triggers(bot, new_names)
    reassembled, scene = llm_prompt_edit.reassemble(
        original,
        blocks,
        {
            "anima": trigger_data["anima_ordered"],
            "sdxl": trigger_data["sdxl_ordered"],
        },
        parsed,
    )
    rebuilt, _ = llm_prompt_edit.rebuild_v3_character_identity(
        reassembled,
        "old negative",
        bot_name=bot["name"],
        bot=bot,
        bot_root={"positive_whitelist": [], "positive_blacklist": []},
        tags=_tags(),
        character_names=new_names,
        scene=scene,
        multi_char_snapshot=remapped,
    )

    llm_prompt_edit.validate_v3_character_identity(rebuilt, new_names)
    validated = multi_char_mask.validate_multi_char_prompt_context(rebuilt, remapped)
    payload = multi_char_mask.extract_multi_char_prompt_payload(rebuilt)
    assert snapshot["character_order"] == old_names
    assert remapped["character_order"] == new_names
    assert remapped["mask_fingerprint"] != snapshot["mask_fingerprint"]
    assert [region["name"] for region in remapped["layout"]["regions"]] == new_names
    assert validated["character_order"] == new_names
    assert payload["char_name_list"] == new_names
    assert payload["char_inform"] == ["New Left, waving", "New Right, laughing"]
    assert payload["mask_fingerprint"] == remapped["mask_fingerprint"]
    assert [item["CHAR"] for item in _json_block(rebuilt, "FACE_ID_DIR")["list"]] == new_names
    assert {item["CHAR"] for item in _json_block(rebuilt, "LORA_DATA")["list"]} == set(new_names)


def test_character_selection_rejects_duplicates_and_non_bot_names():
    bot = _bot(_character("A", "a"), _character("B", "b"))

    for invalid in (["A", "A"], ["A", "Missing"], [], ["A", "B", "C"]):
        try:
            llm_prompt_edit.validate_character_selection(bot, invalid)
        except ValueError:
            continue
        raise AssertionError(f"invalid selection accepted: {invalid}")


def test_single_character_binding_recovers_accidental_pipe_separator(capsys):
    parsed = llm_prompt_edit.bind_scene_characters({
        "plan": "우주선을 둥글게 변경",
        "scene_setup": "night, outdoors, round spaceship",
        "scene_char": "sua | 1girl, short hair, blue hair, smiling",
        "scene_supplement": "",
    }, ["sua"])

    assert parsed["scene_char"] == "sua, 1girl, short hair, blue hair, smiling"
    assert "단일 캐릭터 scene_char의 잘못된 블록 구분자를 자동 복구" in capsys.readouterr().out


def test_two_character_binding_still_rejects_extra_pipe_block():
    with pytest.raises(
        ValueError,
        match=r"characters=2, blocks=3",
    ):
        llm_prompt_edit.bind_scene_characters({
            "scene_char": "Left, smiling | Right, waving | unexpected",
        }, ["Left", "Right"])


@pytest.mark.asyncio
async def test_modified_regenerate_uses_active_bot_and_remapped_two_character_snapshot(
    tmp_path,
    monkeypatch,
):
    old_names = ["Old Left", "Old Right"]
    new_names = ["New Left", "New Right"]
    bot = _bot(*(
        [_character(name, name.lower().replace(" ", "_"), group=True) for name in old_names]
        + [_character(name, name.lower().replace(" ", "_"), group=True) for name in new_names]
    ))
    bot_root = {
        "bots": [bot],
        "positive_whitelist": [],
        "positive_blacklist": [],
    }
    snapshot = multi_char_mask.normalize_multi_char_snapshot({
        "enable": True,
        "character_order": old_names,
        "layout": _layout(old_names),
        "mask_location": "region_mask",
    })
    remapped = multi_char_mask.remap_multi_char_snapshot(snapshot, new_names)
    builder = IllustPromptBuilder()
    source_positive = builder.build_positive_prompt(
        "rooftop",
        "Old Left, standing | Old Right, sitting",
        "two people standing apart",
        old_names,
        bot,
        _tags(),
        bot["illust_settings_group"],
        bot["name"],
        multi_char_context={
            "enable": True,
            "char_name_list": old_names,
            "char_inform": ["Old Left, standing", "Old Right, sitting"],
            "background_prompt": "rooftop",
            "composition_prompt": "two people standing apart",
            "mask_fingerprint": snapshot["mask_fingerprint"],
        },
    )
    parsed = llm_prompt_edit.bind_scene_characters({
        "plan": "교체",
        "scene_setup": "sunset rooftop",
        "scene_char": "waving | laughing",
        "scene_supplement": "two people facing each other",
    }, new_names, old_names)
    blocks = llm_prompt_edit.parse_blocks(source_positive)
    trigger_data = llm_prompt_edit.collect_character_triggers(bot, new_names)
    reassembled, scene = llm_prompt_edit.reassemble(
        source_positive,
        blocks,
        {
            "anima": trigger_data["anima_ordered"],
            "sdxl": trigger_data["sdxl_ordered"],
        },
        parsed,
    )
    effective_positive, effective_negative = llm_prompt_edit.rebuild_v3_character_identity(
        reassembled,
        "old negative",
        bot_name=bot["name"],
        bot=bot,
        bot_root=bot_root,
        tags=_tags(),
        character_names=new_names,
        scene=scene,
        multi_char_snapshot=remapped,
    )
    workflow = {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [source_positive]},
            {"title": "부정프롬프트", "widgets_values": ["source negative"]},
        ]
    }
    backup_name = "identity-remap"
    (tmp_path / f"{backup_name}.json").write_text(
        json.dumps(workflow, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / f"{backup_name}_info.json").write_text(
        json.dumps({
            "provider": "comfy",
            "bot_name": bot["name"],
            "illustration_multi_char": snapshot,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    captured = {}

    async def fake_add_item(item_type, label, params, priority=0):
        captured.update({
            "item_type": item_type,
            "label": label,
            "params": params,
            "priority": priority,
        })
        future = asyncio.get_running_loop().create_future()
        future.set_result({"success": True, "generation_time": 0.1})
        return SimpleNamespace(id="identity-regeneration-item", completion_future=future)

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: bot_root)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setitem(server.app_config, "bot_selected", bot["name"])

    response = await server.handle_api_reschedule_with_modified_prompt(
        _JsonRequest({
            "name": backup_name,
            "positive": effective_positive,
            "negative": effective_negative,
            "identity_edit": {
                "bot_name": bot["name"],
                "character_names": new_names,
            },
        })
    )

    response_payload = json.loads(response.text)
    assert response.status == 202
    assert response_payload["queued"] is True
    assert response_payload["job_id"] == "identity-regeneration-item"
    assert captured["item_type"] == "regenerate"
    assert captured["params"]["bot_name"] == bot["name"]
    assert captured["params"]["illustration_multi_char"]["character_order"] == new_names
    assert captured["params"]["illustration_multi_char"]["mask_fingerprint"] == remapped[
        "mask_fingerprint"
    ]
    assert captured["params"]["illustration_visual_states"] == {
        "New Left": {"visual_profile_id": "card_1", "profile_embedding": False},
        "New Right": {"visual_profile_id": "card_1", "profile_embedding": False},
    }


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _QueryRequest:
    def __init__(self, query):
        self.query = query


@pytest.mark.asyncio
async def test_hybrid_local_two_character_backup_hides_identity_and_rejects_api_bypass(
    tmp_path,
    monkeypatch,
):
    names = ["Old Left", "Old Right"]
    bot = _bot(*[
        _character(name, name.lower().replace(" ", "_"), group=True)
        for name in names
    ])
    positive = IllustPromptBuilder().build_positive_prompt(
        "rooftop",
        "Old Left, standing | Old Right, sitting",
        "two people standing apart",
        names,
        bot,
        _tags(),
        bot["illust_settings_group"],
        bot["name"],
    )
    backup_name = "hybrid-local-two-without-snapshot"
    _write_backup(
        tmp_path,
        backup_name,
        positive,
        info={
            "provider": "comfy",
            "bot_name": bot["name"],
            "illustration_workflow_type": "chansub_v3_anima",
        },
    )
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))

    capability_response = await server.handle_api_llm_edit_capability(
        _QueryRequest({"name": backup_name})
    )
    capability = json.loads(capability_response.text)

    assert capability_response.status == 200
    assert capability["enabled"] is False
    assert capability["character_names"] == names
    assert capability["requires_multi_char_snapshot"] is True
    assert "고정 마스크 스냅샷이 없어" in capability["reason"]

    bypass_response = await server.handle_api_llm_edit_prompt(_JsonRequest({
        "name": backup_name,
        "positive": positive,
        "negative": "source negative",
        "direction": "캐릭터를 바꿔줘",
        "characters": names,
    }))
    bypass_payload = json.loads(bypass_response.text)

    assert bypass_response.status == 409
    assert "고정 마스크 스냅샷이 없어" in bypass_payload["error"]


@pytest.mark.asyncio
async def test_easy_edit_api_rebuilds_single_character_identity_from_active_bot(
    tmp_path,
    monkeypatch,
):
    old = _character("Old", "old_trigger")
    new = _character("New", "new_trigger")
    new["use_image_name_tag"] = True
    new["image_name_tag"] = "new canonical character"
    bot = _bot(old, new)
    bot_root = {
        "bots": [bot],
        "positive_whitelist": [],
        "positive_blacklist": [],
    }
    source_positive = IllustPromptBuilder().build_positive_prompt(
        "daytime street",
        "Old, standing",
        "looking at viewer",
        ["Old"],
        bot,
        _tags(),
        bot["illust_settings_solo"],
        bot["name"],
    )
    backup_name = "easy-edit-single"
    workflow = {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [source_positive]},
            {"title": "부정프롬프트", "widgets_values": ["source negative"]},
        ]
    }
    (tmp_path / f"{backup_name}.json").write_text(
        json.dumps(workflow, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / f"{backup_name}_info.json").write_text(
        json.dumps({"provider": "comfy", "bot_name": bot["name"]}),
        encoding="utf-8",
    )

    async def fake_llm_task(*args, **kwargs):
        return json.dumps({
            "plan": "밤 장면과 새 캐릭터로 변경",
            "scene_setup": "night street",
            "scene_char": "Old, smiling, standing",
            "scene_supplement": "looking at viewer",
        }, ensure_ascii=False)

    async def ignore_notify(*args, **kwargs):
        return None

    import importlib

    lighbd_module = importlib.import_module("modes.lighbd_service")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: bot_root)
    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_llm_task)
    monkeypatch.setattr(server, "notify_frontend", ignore_notify)
    monkeypatch.setattr(lighbd_module, "_log_lighbd_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "apply_word_replacements", lambda p, n, *_args, **_kwargs: (p, n))
    monkeypatch.setattr(server.asset_mode, "_tags", _tags())
    monkeypatch.setitem(server.app_config, "bot_selected", bot["name"])

    response = await server.handle_api_llm_edit_prompt(_JsonRequest({
        "name": backup_name,
        "positive": source_positive,
        "negative": "source negative",
        "direction": "캐릭터를 바꾸고 밤으로 변경",
        "characters": ["New"],
    }))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["identity_edit"] == {
        "bot_name": bot["name"],
        "character_names": ["New"],
        "visual_profile_ids": {"New": "card_1"},
    }
    llm_prompt_edit.validate_v3_character_identity(payload["positive"], ["New"])
    content_blocks = llm_prompt_edit.parse_blocks(payload["positive"])
    assert "Old" not in content_blocks["ANIMA_CONTENT"]
    assert "new canonical character" in content_blocks["ANIMA_CONTENT"]
    assert "new canonical character" in content_blocks["ANIMA_ALL"]
    assert "new canonical character" in content_blocks["SDXL"]
    assert "New" not in [
        tag.strip() for tag in content_blocks["ANIMA_CONTENT"].split(",")
    ]
    assert "new-negative" in payload["negative"]


@pytest.mark.asyncio
async def test_easy_edit_applies_selected_visual_profile_to_scene_and_control_blocks(
    tmp_path,
    monkeypatch,
):
    old = _character("Old", "old_trigger")
    old["visual_cards"] = [{
        "id": "base",
        "label": "카드 1",
        "selection_guide": "기본 외형",
        "aliases": [],
        "appearance": [{"tag": "black hair"}],
        "default_outfit": [{"tag": "school uniform"}],
        "gender_tag": old["gender_tag"],
        "face_tags": old["face_tags"],
        "eye_tags": old["eye_tags"],
        "character_negative": old["character_negative"],
        "loras_solo": old["loras_solo"],
        "loras_group": old["loras_group"],
        "face_loras": old["face_loras"],
        "style_loras": old["style_loras"],
        "use_profile_embedding": False,
    }]
    new = _character("New", "new_trigger")
    awakened_lora = {
        "lora_path": "new-awakened-body.safetensors",
        "trigger": "new_awakened",
        "strength": 0.85,
        "BASE": "anima",
    }
    new["visual_cards"] = [{
        "id": "base",
        "label": "카드 1",
        "selection_guide": "기본 외형",
        "aliases": [],
        "appearance": [{"tag": "brown hair"}],
        "default_outfit": [{"tag": "school uniform"}],
        "gender_tag": "1girl",
        "face_tags": "brown hair, small nose",
        "eye_tags": "brown eyes",
        "character_negative": "base-negative",
        "loras_solo": [new["loras_solo"][0]],
        "loras_group": [],
        "face_loras": new["face_loras"],
        "style_loras": new["style_loras"],
        "use_profile_embedding": False,
    }, {
        "id": "awakened",
        "label": "각성",
        "selection_guide": "각성한 뒤에 사용",
        "aliases": [],
        "appearance": [{"tag": "white hair"}, {"tag": "glowing skin"}],
        "default_outfit": [{"tag": "black armor"}],
        "gender_tag": "1girl",
        "face_tags": "white hair, glowing skin",
        "eye_tags": "gold eyes",
        "character_negative": "awakened-negative",
        "loras_solo": [awakened_lora],
        "loras_group": [],
        "face_loras": [{
            "lora_path": "new-awakened-face.safetensors",
            "trigger": "new_awakened_face",
            "strength": 0.7,
            "BASE": "anima",
        }],
        "style_loras": [],
        "use_profile_embedding": True,
    }]
    source_bot = _bot(old)
    source_bot["name"] = "Source Bot"
    active_bot = _bot(new)
    bot_root = {
        "bots": [source_bot, active_bot],
        "positive_whitelist": [],
        "positive_blacklist": [],
    }
    source_positive = IllustPromptBuilder().build_positive_prompt(
        "daytime street",
        "Old, black hair, standing",
        "looking at viewer",
        ["Old"],
        source_bot,
        _tags(),
        source_bot["illust_settings_solo"],
        source_bot["name"],
    )
    backup_name = "easy-edit-profile"
    _write_backup(
        tmp_path,
        backup_name,
        source_positive,
        info={
            "provider": "comfy",
            "bot_name": source_bot["name"],
            "illustration_visual_states": {
                "Old": {"visual_profile_id": "base", "profile_embedding": False},
            },
        },
    )
    captured_messages = []
    llm_responses = iter([
        {
            "plan": "각성 프로필로 교체",
            "scene_setup": "night street",
            "scene_char": "Old, black hair, standing",
            "scene_supplement": "looking at viewer",
        },
        {
            "plan": "기본 프로필로 교체",
            "scene_setup": "night street",
            "scene_char": "New, white hair, glowing skin, standing",
            "scene_supplement": "looking at viewer",
        },
    ])

    async def fake_llm_task(_task_key, messages, **_kwargs):
        captured_messages.extend(messages)
        return json.dumps(next(llm_responses), ensure_ascii=False)

    async def ignore_notify(*_args, **_kwargs):
        return None

    import importlib

    lighbd_module = importlib.import_module("modes.lighbd_service")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: bot_root)
    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_llm_task)
    monkeypatch.setattr(server, "notify_frontend", ignore_notify)
    monkeypatch.setattr(lighbd_module, "_log_lighbd_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "apply_word_replacements", lambda p, n, *_args, **_kwargs: (p, n))
    monkeypatch.setattr(server.asset_mode, "_tags", _tags())
    monkeypatch.setitem(server.app_config, "bot_selected", active_bot["name"])

    response = await server.handle_api_llm_edit_prompt(_JsonRequest({
        "name": backup_name,
        "positive": source_positive,
        "negative": "source negative",
        "direction": "각성 프로필로 바꿔줘",
        "characters": ["New"],
        "visual_profile_ids": {"New": "awakened"},
    }))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["identity_edit"]["visual_profile_ids"] == {"New": "awakened"}
    blocks = llm_prompt_edit.parse_blocks(payload["positive"])
    assert "white hair" in blocks["ANIMA_CONTENT"]
    assert "glowing skin" in blocks["ANIMA_CONTENT"]
    assert "black hair" not in blocks["ANIMA_CONTENT"]
    assert _json_block(payload["positive"], "CACHE_PATH")["list"][0]["emb_path"].endswith(
        "/New/_visual_profiles/awakened/cache.pt"
    )
    assert _json_block(payload["positive"], "FACE_ID_DIR")["list"][0]["ipa_path"].endswith(
        "/New/_visual_profiles/awakened/cache.ipadpt"
    )
    assert "new-awakened-body.safetensors" in json.dumps(
        _json_block(payload["positive"], "LORA_DATA")
    )
    assert "new-awakened-face.safetensors" in json.dumps(
        _json_block(payload["positive"], "FACE_LORA_DATA")
    )
    assert "awakened-negative" in payload["negative"]
    assert any("selected visual profile=각성" in str(message) for message in captured_messages)

    second_response = await server.handle_api_llm_edit_prompt(_JsonRequest({
        "name": backup_name,
        "positive": payload["positive"],
        "negative": payload["negative"],
        "direction": "기본 프로필로 다시 바꿔줘",
        "characters": ["New"],
        "visual_profile_ids": {"New": "base"},
        "previous_identity": payload["identity_edit"],
    }))
    second_payload = json.loads(second_response.text)

    assert second_response.status == 200
    assert second_payload["identity_edit"] == {
        "bot_name": active_bot["name"],
        "character_names": ["New"],
        "visual_profile_ids": {"New": "base"},
    }
    second_blocks = llm_prompt_edit.parse_blocks(second_payload["positive"])
    assert "brown hair" in second_blocks["ANIMA_CONTENT"]
    assert "white hair" not in second_blocks["ANIMA_CONTENT"]
    assert "glowing skin" not in second_blocks["ANIMA_CONTENT"]
    second_cache_path = _json_block(
        second_payload["positive"], "CACHE_PATH"
    )["list"][0]["emb_path"]
    assert second_cache_path.endswith("/New/cache.pt")
    assert "/_visual_profiles/" not in second_cache_path
    assert "base-negative" in second_payload["negative"]
    assert "awakened-negative" not in second_payload["negative"]
