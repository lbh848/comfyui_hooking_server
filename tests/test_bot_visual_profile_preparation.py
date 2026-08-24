import importlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch


bot_mode = importlib.import_module("modes.bot_mode")


def _card(card_id, label, rep_name, *, use_profile_embedding=False):
    return {
        "id": card_id,
        "label": label,
        "selection_guide": f"{label}을 사용해야 하는 모습",
        "aliases": [],
        "appearance": [],
        "default_outfit_id": "default",
        "outfits": [{
            "id": "default",
            "label": "기본 복장",
            "selection_guide": "기본 복장",
            "tags": [],
        }],
        "rep_images": [rep_name],
        "face_tags": f"face {card_id}",
        "eye_tags": f"eyes {card_id}",
        "absolute_tags": f"absolute {card_id}",
        "use_profile_embedding": use_profile_embedding,
    }


@pytest.fixture
def visual_bot(tmp_path, monkeypatch):
    bot_root = tmp_path / "bot"
    char_dir = bot_root / "sample-bot" / "alice"
    char_dir.mkdir(parents=True)
    (char_dir / "primary.webp").write_bytes(b"primary")
    (char_dir / "alternate.webp").write_bytes(b"alternate")
    cards = [
        _card("primary", "카드 1", "primary.webp"),
        # The preparation pipeline must not depend on this generation-only flag.
        _card("alternate", "카드 2", "alternate.webp", use_profile_embedding=False),
    ]
    character = {
        "name": "alice",
        "rep_images": ["primary.webp"],
        "visual_cards": cards,
    }
    data = {"bots": [{"name": "sample-bot", "characters": [character]}]}
    monkeypatch.setattr(bot_mode, "BOT_DIR", str(bot_root))
    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    return tmp_path, bot_root, char_dir, data


def test_representative_and_face_paths_cover_every_visual_card(visual_bot):
    _tmp_path, _bot_root, char_dir, _data = visual_bot

    reps = bot_mode.get_bot_visual_rep_paths("sample-bot", "alice")

    assert [item["visual_card_id"] for item in reps] == ["primary", "alternate"]
    assert [item["visual_card_index"] for item in reps] == [1, 2]
    assert bot_mode.bot_visual_artifact_dir(
        "sample-bot", "alice", "primary"
    ) == str(char_dir)
    assert bot_mode.bot_visual_artifact_dir(
        "sample-bot", "alice", "alternate"
    ) == str(char_dir / "_visual_profiles" / "alternate")


@pytest.mark.asyncio
async def test_image_list_replaces_primary_face_with_requested_profile_face(
    visual_bot,
):
    _tmp_path, _bot_root, char_dir, _data = visual_bot
    (char_dir / "_face_image.webp").write_bytes(b"primary-face")
    (char_dir / "_face_image_prompt.json").write_text(
        json.dumps({"prompt": "primary prompt", "negative": "primary negative"}),
        encoding="utf-8",
    )
    profile_dir = char_dir / "_visual_profiles" / "alternate"
    profile_dir.mkdir(parents=True)
    (profile_dir / "_face_image.webp").write_bytes(b"alternate-face")
    (profile_dir / "_face_image_prompt.json").write_text(
        json.dumps({"prompt": "alternate prompt", "negative": "alternate negative"}),
        encoding="utf-8",
    )

    class Request:
        query = {
            "bot": "sample-bot",
            "character": "alice",
            "visual_card_id": "alternate",
        }

    response = await bot_mode.BotMode().handle_get_images(Request())
    payload = json.loads(response.text)
    faces = [
        image for image in payload["images"]
        if image["filename"] == "_face_image.webp"
    ]

    assert payload["visual_card_id"] == "alternate"
    assert len(faces) == 1
    assert faces[0]["prompt"] == "alternate prompt"
    assert faces[0]["negative"] == "alternate negative"
    assert faces[0]["visual_card_id"] == "alternate"
    assert faces[0]["url"].endswith(
        "/_face_image.webp?visual_card_id=alternate"
    )
    assert {image["filename"] for image in payload["images"]} >= {
        "primary.webp",
        "alternate.webp",
    }


@pytest.mark.asyncio
async def test_image_list_without_profile_id_keeps_primary_face_compatibility(
    visual_bot,
):
    _tmp_path, _bot_root, char_dir, _data = visual_bot
    (char_dir / "_face_image.webp").write_bytes(b"primary-face")
    (char_dir / "_face_image_prompt.json").write_text(
        json.dumps({"prompt": "primary prompt"}),
        encoding="utf-8",
    )

    class Request:
        query = {"bot": "sample-bot", "character": "alice"}

    response = await bot_mode.BotMode().handle_get_images(Request())
    payload = json.loads(response.text)
    face = next(
        image for image in payload["images"]
        if image["filename"] == "_face_image.webp"
    )

    assert payload["visual_card_id"] == ""
    assert face["prompt"] == "primary prompt"
    assert face["visual_card_id"] == ""
    assert "visual_card_id=" not in face["url"]


@pytest.mark.asyncio
async def test_prompt_update_writes_only_requested_profile_prompt(visual_bot):
    _tmp_path, _bot_root, char_dir, _data = visual_bot
    root_prompt = char_dir / "_face_image_prompt.json"
    root_prompt.write_text(
        json.dumps({"prompt": "primary prompt", "negative": "keep root"}),
        encoding="utf-8",
    )
    profile_dir = char_dir / "_visual_profiles" / "alternate"
    profile_dir.mkdir(parents=True)
    profile_prompt = profile_dir / "_face_image_prompt.json"
    profile_prompt.write_text(
        json.dumps({"prompt": "old alternate", "negative": "keep alternate"}),
        encoding="utf-8",
    )

    class Request:
        async def json(self):
            return {
                "bot": "sample-bot",
                "character": "alice",
                "visual_card_id": "alternate",
                "filename": "_face_image.webp",
                "prompt": "new alternate",
            }

    response = await bot_mode.BotMode().handle_update_prompt(Request())
    payload = json.loads(response.text)

    assert payload["updated"] is True
    assert payload["visual_card_id"] == "alternate"
    assert json.loads(root_prompt.read_text(encoding="utf-8")) == {
        "prompt": "primary prompt",
        "negative": "keep root",
    }
    assert json.loads(profile_prompt.read_text(encoding="utf-8")) == {
        "prompt": "new alternate",
        "negative": "keep alternate",
    }


@pytest.mark.asyncio
async def test_data_patch_copies_every_card_even_without_profile_embedding_flag(
    visual_bot, monkeypatch
):
    tmp_path, _bot_root, _char_dir, _data = visual_bot
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps({"comfy_input_dir": str(comfy_input)}), encoding="utf-8"
    )
    monkeypatch.setattr(bot_mode, "BASE_DIR", str(tmp_path))

    class Request:
        async def json(self):
            return {"bot_name": "sample-bot", "char_names": ["alice"]}

    response = await bot_mode.BotDataPatcher().handle_data_patch(Request())
    payload = json.loads(response.text)

    assert "error" not in payload
    primary = comfy_input / "soya_bot" / "sample-bot" / "alice" / "representation.webp"
    alternate = (
        comfy_input
        / "soya_bot"
        / "sample-bot"
        / "alice"
        / "_visual_profiles"
        / "alternate"
        / "representation.webp"
    )
    assert primary.read_bytes() == b"primary"
    assert alternate.read_bytes() == b"alternate"
    assert [item["visual_card_id"] for item in payload["visual_targets"]] == [
        "primary",
        "alternate",
    ]


def test_face_tag_save_updates_only_requested_card(visual_bot, monkeypatch):
    _tmp_path, _bot_root, _char_dir, data = visual_bot
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    result = bot_mode.save_char_face_tags(
        "sample-bot",
        "alice",
        face_tags="alternate face updated",
        eye_tags="alternate eyes updated",
        absolute_tags="alternate absolute updated",
        visual_card_id="alternate",
    )

    assert result["success"] is True
    character = data["bots"][0]["characters"][0]
    assert character["visual_cards"][0]["face_tags"] == "face primary"
    assert character["visual_cards"][1]["face_tags"] == "alternate face updated"
    assert character["face_tags"] == "face primary"
    assert len(saved) == 1


def test_frontend_one_click_and_manual_actions_use_visual_card_targets():
    source = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "function _botVisualTargets(" in source
    assert "visual_card_id: target.visual_card_id" in source
    assert "patchResult.visual_targets || run.visualTargets" in source
    assert "visual_targets: faceTargets.map" in source
    assert "data-visual-card-id" in source
    assert "프로필 ${run.visualTargets?.length || 0}개" in source


@pytest.mark.asyncio
async def test_bulk_main_rep_replaces_all_representative_candidates_only_on_requested_card(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, char_dir, data = visual_bot
    (char_dir / "replacement.webp").write_bytes(b"replacement")
    data["bots"][0]["characters"][0]["visual_cards"][1]["rep_images"] = [
        "alternate.webp",
        "alternate-candidate.webp",
    ]
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "push",
        "items": [{
            "char_name": "alice",
            "visual_card_id": "alternate",
            "filename": "replacement.webp",
        }],
    })
    payload = json.loads(response.text)
    character = data["bots"][0]["characters"][0]

    assert payload["updated"] == [{
        "char_name": "alice",
        "visual_card_id": "alternate",
        "filename": "replacement.webp",
        "created_profile": False,
    }]
    assert character["visual_cards"][0]["rep_images"] == ["primary.webp"]
    assert character["visual_cards"][1]["rep_images"] == ["replacement.webp"]
    assert character["rep_images"] == ["primary.webp"]
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_bulk_main_rep_creates_profile_from_selected_source_on_apply(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, char_dir, data = visual_bot
    (char_dir / "transformed.webp").write_bytes(b"transformed")
    source_before = deepcopy(data["bots"][0]["characters"][0]["visual_cards"][1])
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "filename": "transformed.webp",
            "create_profile": True,
            "source_visual_card_id": "alternate",
            "profile_label": "변신 상태",
        }],
    })
    payload = json.loads(response.text)
    character = data["bots"][0]["characters"][0]
    created = character["visual_cards"][2]

    assert len(character["visual_cards"]) == 3
    assert created["id"].startswith("card_")
    assert created["label"] == "변신 상태"
    assert created["rep_images"] == ["transformed.webp"]
    assert created["face_tags"] == source_before["face_tags"]
    assert created["outfits"] == source_before["outfits"]
    assert created["selection_guide"] == ""
    assert created["aliases"] == []
    assert created["use_profile_embedding"] is True
    assert character["rep_images"] == ["primary.webp"]
    assert payload["updated"][0]["visual_card_id"] == created["id"]
    assert payload["updated"][0]["created_profile"] is True
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_bulk_main_rep_protects_each_profile_independently(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, char_dir, data = visual_bot
    (char_dir / "replacement.webp").write_bytes(b"replacement")
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "visual_card_id": "alternate",
            "filename": "replacement.webp",
        }],
    })
    payload = json.loads(response.text)

    assert payload["updated"] == []
    assert payload["skipped"] == [{
        "char_name": "alice",
        "visual_card_id": "alternate",
        "reason": "이미 대표 있음",
    }]
    assert data["bots"][0]["characters"][0]["visual_cards"][1]["rep_images"] == [
        "alternate.webp"
    ]
    assert saved == []


@pytest.mark.asyncio
async def test_bulk_main_rep_protect_mode_allows_explicit_manual_override(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, char_dir, data = visual_bot
    (char_dir / "replacement.webp").write_bytes(b"replacement")
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "visual_card_id": "alternate",
            "filename": "replacement.webp",
            "manual_override": True,
        }],
    })
    payload = json.loads(response.text)

    assert payload["skipped"] == []
    assert payload["updated"] == [{
        "char_name": "alice",
        "visual_card_id": "alternate",
        "filename": "replacement.webp",
        "created_profile": False,
    }]
    assert data["bots"][0]["characters"][0]["visual_cards"][1]["rep_images"] == [
        "replacement.webp"
    ]
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_bulk_main_rep_rejects_profile_creation_at_card_limit(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, _char_dir, data = visual_bot
    character = data["bots"][0]["characters"][0]
    source = character["visual_cards"][0]
    for index in range(3, bot_mode.MAX_VISUAL_CARDS + 1):
        extra = deepcopy(source)
        extra["id"] = f"extra_{index}"
        extra["label"] = f"카드 {index}"
        character["visual_cards"].append(extra)
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "filename": "primary.webp",
            "create_profile": True,
            "source_visual_card_id": "primary",
        }],
    })
    payload = json.loads(response.text)

    assert payload["updated"] == []
    assert payload["skipped"][0]["reason"] == "프로필 최대 10개 초과"
    assert len(character["visual_cards"]) == bot_mode.MAX_VISUAL_CARDS
    assert saved == []


@pytest.mark.asyncio
async def test_bulk_main_rep_removes_requested_non_primary_card_on_apply(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, _char_dir, data = visual_bot
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "visual_card_id": "alternate",
            "remove_profile": True,
        }],
    })
    payload = json.loads(response.text)
    character = data["bots"][0]["characters"][0]

    assert payload["updated"] == []
    assert payload["removed"] == [{
        "char_name": "alice",
        "visual_card_id": "alternate",
        "profile_label": "카드 2",
    }]
    assert payload["skipped"] == []
    assert [card["id"] for card in character["visual_cards"]] == ["primary"]
    assert character["rep_images"] == ["primary.webp"]
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_bulk_main_rep_refuses_to_remove_primary_card(
    visual_bot, monkeypatch
):
    _tmp_path, _bot_root, _char_dir, data = visual_bot
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "sample-bot",
        "mode": "protect",
        "items": [{
            "char_name": "alice",
            "visual_card_id": "primary",
            "remove_profile": True,
        }],
    })
    payload = json.loads(response.text)

    assert payload["updated"] == []
    assert payload["removed"] == []
    assert payload["skipped"] == [{
        "char_name": "alice",
        "visual_card_id": "primary",
        "reason": "기본 프로필은 삭제할 수 없음",
    }]
    assert [
        card["id"]
        for card in data["bots"][0]["characters"][0]["visual_cards"]
    ] == ["primary", "alternate"]
    assert saved == []


@pytest.mark.asyncio
async def test_legacy_bulk_main_rep_payload_does_not_force_card_migration(
    tmp_path, monkeypatch
):
    bot_root = tmp_path / "bot"
    char_dir = bot_root / "legacy-bot" / "alice"
    char_dir.mkdir(parents=True)
    (char_dir / "alice.webp").write_bytes(b"alice")
    character = {"name": "alice"}
    data = {"bots": [{"name": "legacy-bot", "characters": [character]}]}
    saved = []
    monkeypatch.setattr(bot_mode, "BOT_DIR", str(bot_root))
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(value))

    response = await bot_mode.BotMode()._bulk_set_main_rep(data, {
        "bot_name": "legacy-bot",
        "mode": "protect",
        "items": [{"char_name": "alice", "filename": "alice.webp"}],
    })
    payload = json.loads(response.text)

    assert payload["updated"][0]["visual_card_id"] == "card_1"
    assert character["rep_images"] == ["alice.webp"]
    assert "visual_cards" not in character
    assert len(saved) == 1


def test_representative_batch_frontend_supports_profile_drafts():
    source = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "function _repBatchAddDraftProfile(" in source
    assert "function _repBatchCharacterPalette(" in source
    assert "const hue = (hash >>> 0) % 360" in source
    assert "function _repBatchSelectManualTarget(" in source
    assert "function _repBatchResolvedTarget(" in source
    assert 'data-role="rep-batch-manual"' in source
    assert 'aria-pressed="' in source
    assert "대표로 지정" in source
    assert "manualTarget" in source
    assert "profile.isNew || profile.manualTarget" in source
    assert "profile.rep0 && !profile.manualTarget" in source
    assert "보호 모드 프로필 수동 후보 선택 거부" not in source
    assert "manual_override: !!profile.manualTarget" in source
    assert "skippedNoImage: skipNoImage" in source
    assert "create_profile = true" in source
    assert "source_visual_card_id" in source
    assert "visual_card_id: profile.profileId" in source
    assert "＋ 프로필 추가" in source
    assert "function _repBatchToggleRemoveProfile(" in source
    assert "profile.removePending" in source
    assert "remove_profile: true" in source
    assert "카드 빼기" in source
    assert "빼기 취소" in source
    assert "[1] 기본 카드는 뺄 수 없습니다." in source
    assert "대체: 기존 대표도 교체" in source
    assert "밀어내기: 기존도 교체" not in source
    assert "기존 대표와 후보는 제거" in source


def test_bubble_matching_uses_best_face_across_character_cards():
    from modes.bubble_match import match_speakers_to_faces

    segments = [{"speaker": "alice", "text": "hello", "type": "speech"}]
    image = Image.new("RGB", (100, 100), "white")
    faces = [{"box": (20, 20, 60, 60), "conf": 0.9, "image": image}]
    root_embedding = np.asarray([0.0, 1.0], dtype=np.float32)
    profile_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    face_embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    with patch(
        "modes.bubble_match._project_character_name_map",
        return_value={"alice": "alice"},
    ), patch(
        "modes.face_embedder.get_char_embedding", return_value=root_embedding
    ), patch(
        "modes.face_embedder.get_char_appearance", return_value=None
    ), patch(
        "modes.face_embedder.get_char_profile_prototypes",
        return_value=[{
            "visual_card_id": "alternate",
            "visual_card_index": 2,
            "embedding": profile_embedding,
            "appearance": None,
        }],
    ), patch(
        "modes.face_embedder.embed_face_crop", return_value=face_embedding
    ):
        results = match_speakers_to_faces(
            segments,
            faces,
            "sample-bot",
            match_thres=0.9,
            appearance_weight=0.0,
        )

    assert results[0]["face_box"] == faces[0]["box"]
    assert results[0]["sim"] == pytest.approx(1.0)
