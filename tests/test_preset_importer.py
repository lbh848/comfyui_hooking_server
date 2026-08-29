import copy
import json

import pytest

from modes import preset_importer


def _source_document():
    return {
        "name": "가져오기 테스트",
        "version": 1,
        "library": {
            "공용": {
                "name": "공용",
                "pieces": [
                    {"name": "eyes", "prompt": "blue eyes", "multi": False},
                ],
            },
        },
        "scenes": {
            "scene-key": {
                "name": "표정과 구도",
                "slots": [
                    [{"id": "piece-a", "prompt": "<공용.eyes>, smile"}],
                    [
                        {"id": "piece-b", "prompt": "(cowboy shot, from front)"},
                        {"id": "piece-c", "prompt": "close-up"},
                    ],
                ],
            },
        },
        "presets": {
            "SDImageGenEasy": [
                {
                    "name": "기본 스타일",
                    "frontPrompt": "artist:test",
                    "backPrompt": "best quality",
                    "uc": "bad hands, watermark",
                },
            ],
        },
    }


def _assign_all_fragments(analysis):
    draft = {
        "import_id": analysis["import_id"],
        "items": [],
    }
    for item in analysis["items"]:
        edited = {
            "id": item["id"],
            "selected": item["selected"],
            "target_name": item["target_name"],
            "fragments": copy.deepcopy(item["fragments"]),
        }
        for fragment in edited["fragments"]:
            if fragment["category"] != "unassigned":
                continue
            if item["source_kind"] == "generation_preset":
                fragment["category"] = (
                    "artist_presets"
                    if fragment["source_field"] == "frontPrompt"
                    else "quality_presets"
                )
            else:
                fragment["category"] = "composition_presets"
        draft["items"].append(edited)
    return draft


def test_analyze_preserves_group_boundaries_variants_and_weighted_fragment():
    analysis = preset_importer.analyze_document("source.json", _source_document())

    assert analysis["format"] == "sdstudio_session"
    assert analysis["summary"]["scene_group_count"] == 1
    assert analysis["summary"]["scene_item_count"] == 2
    assert analysis["summary"]["generation_group_count"] == 1
    assert len(analysis["items"]) == 3

    scene_items = [item for item in analysis["items"] if item["source_kind"] == "scene"]
    assert scene_items[0]["group_id"] == scene_items[1]["group_id"]
    assert scene_items[0]["target_name"].endswith("/표정과 구도")
    assert scene_items[1]["target_name"].endswith("/표정과 구도_v1")
    assert [fragment["text"] for fragment in scene_items[0]["fragments"]] == [
        "blue eyes",
        "smile",
        "\\(cowboy shot, from front\\)",
    ]

    generation = next(item for item in analysis["items"] if item["source_kind"] == "generation_preset")
    negative = [fragment for fragment in generation["fragments"] if fragment["source_field"] == "uc"]
    assert [fragment["category"] for fragment in negative] == [
        "negative_presets",
        "negative_presets",
    ]
    assert all(not fragment["llm_eligible"] for fragment in negative)


def test_importer_exposes_exact_nai_weight_and_one_decimal_anima_output():
    document = _source_document()
    document["library"]["공용"]["pieces"][0]["prompt"] = "1.255::blue eyes::"

    analysis = preset_importer.analyze_document("source.json", document)
    scene_item = next(item for item in analysis["items"] if item["source_kind"] == "scene")
    fragment = scene_item["fragments"][0]

    assert analysis["target"]["max_abs_weight"] == 1.5
    assert analysis["target"]["weight_quantum"] == 0.1
    assert analysis["target"]["weight_rounding"] == "ROUND_HALF_UP"
    assert fragment["original_text"] == "1.255::blue eyes::"
    assert fragment["import_text"] == "(blue eyes:1.3)"
    assert fragment["normalization"]["raw_weight"] == "1.255"
    assert fragment["normalization"]["weight"] == "1.3"


def test_llm_contract_requires_every_exact_id_and_never_returns_rewritten_text():
    analysis = preset_importer.analyze_document("source.json", _source_document())
    selected_item = analysis["items"][0]
    selected_fragment = selected_item["fragments"][0]
    targets = [{
        "item_id": selected_item["id"],
        "fragment_ids": [selected_fragment["id"]],
    }]
    payload = preset_importer.build_classification_payload(analysis["import_id"], targets)
    item = payload["items"][0]
    parsed = {
        "items": [{
            "item_id": item["item_id"],
            "assignments": [
                {"fragment_id": fragment["fragment_id"], "category": "composition_presets"}
                for fragment in item["fragments"]
            ],
        }],
    }

    valid, reason = preset_importer.validate_classification_response(parsed, payload)
    assert valid is True
    assert reason == ""
    assert all("text" not in assignment for assignment in parsed["items"][0]["assignments"])

    invalid = copy.deepcopy(parsed)
    invalid["items"][0]["assignments"].pop()
    valid, reason = preset_importer.validate_classification_response(invalid, payload)
    assert valid is False
    assert "누락" in reason

    assert payload["target_fragment_count"] == 1
    assert payload["source_prompt_syntax"] == "NAI"
    assert payload["target_adapter"] == "ANIMA"
    assert item["fragments"][0]["source_nai"] == selected_fragment["original_text"]
    assert item["fragments"][0]["text"] == selected_fragment["import_text"]


def test_llm_batch_limit_counts_target_fragments_not_items():
    analysis = preset_importer.analyze_document("source.json", _source_document())
    item = analysis["items"][0]
    session_item = preset_importer.get_analysis_session(analysis["import_id"])["items_by_id"][item["id"]]
    template = copy.deepcopy(session_item["fragments"][0])
    session_item["fragments"] = []
    fragment_ids = []
    for index in range(31):
        fragment = copy.deepcopy(template)
        fragment["id"] = f"{item['id']}-synthetic-{index}"
        session_item["fragments"].append(fragment)
        fragment_ids.append(fragment["id"])

    with pytest.raises(preset_importer.PresetImportError, match="30개"):
        preset_importer.build_classification_payload(
            analysis["import_id"],
            [{"item_id": item["id"], "fragment_ids": fragment_ids}],
        )

    payload = preset_importer.build_classification_payload(
        analysis["import_id"],
        [{"item_id": item["id"], "fragment_ids": fragment_ids[:30]}],
    )
    assert payload["target_fragment_count"] == 30


def test_scene_negative_and_workflow_shared_prompts_are_not_silently_dropped():
    document = _source_document()
    document["scenes"]["scene-key"]["sceneCharacterUC"] = "wrong eyes, extra fingers"
    document["presetShareds"] = {
        "SDImageGenEasy": {
            "characterPrompt": "1girl, blue hair",
            "backgroundPrompt": "sunlit room",
            "uc": "text",
        },
    }

    analysis = preset_importer.analyze_document("source.json", document)
    scene_item = next(item for item in analysis["items"] if item["source_kind"] == "scene")
    scene_uc = [
        fragment for fragment in scene_item["fragments"]
        if fragment["source_field"] == "sceneCharacterUC"
    ]
    shared = next(item for item in analysis["items"] if item["source_kind"] == "workflow_shared")

    assert [fragment["category"] for fragment in scene_uc] == [
        "character_negative_presets",
        "character_negative_presets",
    ]
    assert all(not fragment["llm_eligible"] for fragment in scene_uc)
    assert analysis["summary"]["shared_group_count"] == 1
    assert analysis["summary"]["shared_item_count"] == 1
    assert [fragment["source_field"] for fragment in shared["fragments"]] == [
        "characterPrompt",
        "characterPrompt",
        "backgroundPrompt",
        "uc",
    ]
    assert shared["fragments"][-1]["category"] == "negative_presets"


def test_validate_requires_explicit_exclusion_instead_of_source_fragment_deletion():
    analysis = preset_importer.analyze_document("source.json", _source_document())
    draft = _assign_all_fragments(analysis)
    first = draft["items"][0]
    removed = first["fragments"].pop()

    invalid = preset_importer.validate_draft(draft, {}, {})
    assert invalid["success"] is False
    assert any(error["code"] == "missing_fragments" for error in invalid["errors"])

    first["fragments"].append({**removed, "excluded": True})
    valid = preset_importer.validate_draft(draft, {}, {})
    assert valid["success"] is True
    assert valid["summary"]["excluded_fragment_count"] == 1


def test_commit_backs_up_existing_data_and_writes_separate_manifest(tmp_path, monkeypatch):
    asset_dir = tmp_path / "asset_data"
    backup_dir = asset_dir / "backup"
    tags_file = asset_dir / "tags.json"
    hidden_file = asset_dir / "hidden_tags.json"
    manifest_file = asset_dir / "preset_import_manifests.json"
    asset_dir.mkdir()

    active = {"composition_presets": {"기존": ["old"]}}
    hidden = {"expressions": {"숨김": ["hidden"]}}
    tags_file.write_text(json.dumps(active, ensure_ascii=False), encoding="utf-8")
    hidden_file.write_text(json.dumps(hidden, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(preset_importer, "ASSET_DATA_DIR", str(asset_dir))
    monkeypatch.setattr(preset_importer, "TAGS_FILE", str(tags_file))
    monkeypatch.setattr(preset_importer, "HIDDEN_TAGS_FILE", str(hidden_file))
    monkeypatch.setattr(preset_importer, "MANIFEST_FILE", str(manifest_file))
    monkeypatch.setattr(preset_importer, "BACKUP_DIR", str(backup_dir))

    analysis = preset_importer.analyze_document("source.json", _source_document())
    draft = _assign_all_fragments(analysis)
    result = preset_importer.commit_draft(draft, [], active, hidden)

    assert result["success"] is True
    assert result["target_count"] > 0
    assert tags_file.is_file()
    assert hidden_file.is_file()
    assert manifest_file.is_file()
    assert len(list(backup_dir.glob("preset_import_tags_*.json"))) == 1
    assert len(list(backup_dir.glob("preset_import_hidden_tags_*.json"))) == 1

    saved_tags = json.loads(tags_file.read_text(encoding="utf-8"))
    saved_hidden = json.loads(hidden_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert saved_tags["composition_presets"]["기존"] == ["old"]
    assert saved_hidden == hidden
    assert manifest["imports"][0]["id"] == result["manifest_id"]
    assert manifest["imports"][0]["source"]["filename"] == "source.json"
    assert manifest["imports"][0]["target"]["adapter"] == "anima"
    assert manifest["imports"][0]["items"][0]["canonical_prompts"]
    assert manifest["imports"][0]["items"][0]["fragments"][0]["import_text"]


def test_commit_requires_resolution_for_each_real_conflict(tmp_path, monkeypatch):
    analysis = preset_importer.analyze_document("source.json", _source_document())
    draft = _assign_all_fragments(analysis)
    first_name = draft["items"][0]["target_name"]
    active = {"composition_presets": {first_name: ["different"]}}
    validation = preset_importer.validate_draft(draft, active, {})
    assert validation["success"] is True
    assert validation["conflicts"]

    monkeypatch.setattr(preset_importer, "TAGS_FILE", str(tmp_path / "tags.json"))
    monkeypatch.setattr(preset_importer, "HIDDEN_TAGS_FILE", str(tmp_path / "hidden.json"))
    monkeypatch.setattr(preset_importer, "MANIFEST_FILE", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(preset_importer, "BACKUP_DIR", str(tmp_path / "backup"))

    with pytest.raises(preset_importer.PresetImportError, match="충돌 해결값"):
        preset_importer.commit_draft(draft, [], active, {})


def test_scene_chain_slots_follow_scene_order_and_connect_anima_fields():
    analysis = preset_importer.analyze_document("source.json", _source_document())
    draft = _assign_all_fragments(analysis)
    scene_items = [item for item in draft["items"] if item["id"].startswith("scene-")]
    for item in scene_items:
        for fragment in item["fragments"]:
            text = fragment["text"]
            if text == "blue eyes":
                fragment["category"] = "appearances"
            elif text == "smile":
                fragment["category"] = "expressions"
            else:
                fragment["category"] = "composition_presets"

    validation = preset_importer.validate_draft(draft, {}, {})
    assert validation["success"] is True
    targets = [
        {
            "item_id": record["item_id"],
            "source_kind": record["source_kind"],
            "category": record["category"],
            "target_name": record["name"],
            "target_state": "active",
        }
        for record in validation["records"]
    ]
    # 최종 충돌 해결로 카테고리별 이름이 달라져도 실제 저장명을 연결해야 한다.
    first_expression = next(
        target for target in targets
        if target["item_id"] == scene_items[0]["id"]
        and target["category"] == "expressions"
    )
    first_expression["target_name"] += "_이름변경"

    plan = preset_importer.build_scene_chain_slots(draft, targets)

    assert plan["slot_count"] == 2
    assert len(plan["chains"]) == 2
    assert plan["chains"][0]["appearance"] == scene_items[0]["target_name"]
    assert plan["chains"][0]["expression"].endswith("_이름변경")
    assert plan["chains"][0]["composition_preset"] == scene_items[0]["target_name"]
    assert plan["chains"][0]["anima_quality_preset"] == ""
    assert all("기본 스타일" not in slot["composition_preset"] for slot in plan["chains"])


def test_scene_chain_slots_leave_hidden_kept_presets_unlinked():
    analysis = preset_importer.analyze_document("source.json", _source_document())
    draft = _assign_all_fragments(analysis)
    validation = preset_importer.validate_draft(draft, {}, {})
    scene_record = next(
        record for record in validation["records"] if record["source_kind"] == "scene"
    )
    targets = [{
        "item_id": scene_record["item_id"],
        "source_kind": "scene",
        "category": scene_record["category"],
        "target_name": scene_record["name"],
        "target_state": "hidden",
    }]
    for item in draft["items"]:
        item["selected"] = item["id"] == scene_record["item_id"]

    plan = preset_importer.build_scene_chain_slots(draft, targets)

    assert plan["slot_count"] == 1
    assert plan["hidden_omitted_count"] == 1
    assert plan["chains"][0]["composition_preset"] == ""
