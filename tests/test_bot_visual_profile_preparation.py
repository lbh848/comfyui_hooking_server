import importlib
import json
import os
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
