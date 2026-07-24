from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_chat_history as history


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    root = tmp_path / "workflow_backup" / "illustration_chat_history"
    monkeypatch.setattr(history, "HISTORY_ROOT", str(root))
    monkeypatch.setattr(history, "RECORDS_DIR", str(root / "records"))
    monkeypatch.setattr(history, "TRASH_DIR", str(root / "trash"))
    monkeypatch.setattr(history, "SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setattr(
        history,
        "REQUIREMENTS_BACKUP_DIR",
        str(tmp_path / "요구사항" / "illustration_chat_history_backups"),
    )
    return root


def _chat(role: str, text: str) -> dict:
    return {"role": role, "data": text}


def test_normal_continuation_appends_only_new_delta(isolated_history):
    first_chats = [
        _chat("user", "A" * 120),
        _chat("char", "B" * 120),
    ]
    first = history.prepare_history(first_chats, 1, "bot-a")
    assert first["operation"] == "new"
    history.finalize_history(first, {"call2_output": "first"})

    second_chats = [
        _chat("user", "A" * 120),
        _chat("char", "B" * 120),
        _chat("user", "C" * 120),
        _chat("char", "D" * 120),
    ]
    second = history.prepare_history(second_chats, 3, "bot-a")
    assert second["history_id"] == first["history_id"]
    assert second["operation"] == "append"
    assert [item["content"] for item in second["proposed_messages"]] == [
        "A" * 120,
        "B" * 120,
        "C" * 120,
        "D" * 120,
    ]


def test_continuation_accepts_high_similarity_tail_with_one_exact_anchor(isolated_history):
    stable_user = "The old library remains quiet. " * 8
    saved_reply = "Alice walks past the tall shelves and watches the dusty window. " * 6
    first = history.prepare_history([
        _chat("user", stable_user),
        _chat("char", saved_reply),
    ], 1, "bot-a")
    history.finalize_history(first, {"call2_output": "first"})

    lightly_edited_reply = saved_reply.replace("dusty window", "rainy window", 1)
    continued = history.prepare_history([
        _chat("user", stable_user),
        _chat("char", lightly_edited_reply),
        _chat("user", "What does Alice see next? " * 5),
        _chat("char", "Alice notices a blue dress reflected in the glass. " * 5),
    ], 3, "bot-a")

    assert continued["history_id"] == first["history_id"]
    assert continued["operation"] == "append"
    assert continued["match"]["similarity"] < 1.0
    assert continued["match"]["similarity"] >= 0.90


def test_same_past_different_current_is_reroll_and_rolls_back_state(isolated_history):
    chats = [
        _chat("user", "same past " * 15),
        _chat("char", "first current " * 12),
    ]
    first = history.prepare_history(chats, 1, "bot-a")
    first_state = {
        "alice": {
            "canonical_name": "Alice",
            "current_wardrobe": {"body_state": "nude", "worn": []},
        }
    }
    history.finalize_history(first, {"character_states_after": first_state})

    rerolled_chats = [
        _chat("user", "same past " * 15),
        _chat("char", "second current " * 12),
    ]
    reroll = history.prepare_history(rerolled_chats, 1, "bot-a")
    assert reroll["history_id"] == first["history_id"]
    assert reroll["operation"] == "reroll"
    assert reroll["state_before"] == {}
    assert [item["content"] for item in reroll["proposed_messages"]] == [
        "same past " * 15,
        "second current " * 12,
    ]

    clothed = {
        "alice": {
            "canonical_name": "Alice",
            "current_wardrobe": {"body_state": "clothed", "worn": ["blue dress"]},
        }
    }
    saved = history.finalize_history(reroll, {"character_states_after": clothed})
    assert saved["characters"] == clothed
    assert saved["active_turn"]["reroll_index"] == 1
    assert len(saved["reroll_archive"]) == 1
    assert "first current" in saved["reroll_archive"][0]["current_content"]


def test_same_past_and_current_is_duplicate(isolated_history):
    chats = [
        _chat("user", "past message " * 12),
        _chat("char", "same current " * 12),
    ]
    first = history.prepare_history(chats, 1, "bot-a")
    history.finalize_history(first, {"call2_output": "first"})

    duplicate = history.prepare_history(chats, 1, "bot-a")
    assert duplicate["history_id"] == first["history_id"]
    assert duplicate["operation"] == "duplicate"
    assert len(duplicate["proposed_messages"]) == 2


def test_revision_conflict_forks_instead_of_overwriting_newer_history(isolated_history):
    base_chats = [
        _chat("user", "shared user base " * 10),
        _chat("char", "shared char base " * 10),
    ]
    initial = history.prepare_history(base_chats, 1, "bot-a")
    history.finalize_history(initial, {"call2_output": "initial"})

    first_branch = history.prepare_history(base_chats + [
        _chat("user", "first branch user " * 8),
        _chat("char", "first branch current " * 8),
    ], 3, "bot-a")
    second_branch = history.prepare_history(base_chats + [
        _chat("user", "second branch user " * 8),
        _chat("char", "second branch current " * 8),
    ], 3, "bot-a")

    saved_first = history.finalize_history(first_branch, {"call2_output": "first"})
    saved_second = history.finalize_history(second_branch, {"call2_output": "second"})
    assert saved_first["history_id"] == initial["history_id"]
    assert saved_second["history_id"] != initial["history_id"]
    assert "first branch current" in saved_first["messages"][-1]["content"]
    assert "second branch current" in saved_second["messages"][-1]["content"]
    assert len(history.list_histories()) == 2


def test_storage_limit_cuts_oldest_content_and_settings_clamp_call_limits(isolated_history):
    settings = history.save_settings({
        "storage_max_chars": 1_000,
        "call1_history_chars": 5_000,
        "call2_fallback_history_chars": 2_000,
        "call3_fallback_history_chars": 500,
        "reroll_archive_limit": 2,
    })
    assert settings["call1_history_chars"] == 1_000
    assert settings["call2_fallback_history_chars"] == 1_000
    assert settings["call3_fallback_history_chars"] == 500

    chats = [
        _chat("user", "A" * 800),
        _chat("char", "B" * 800),
    ]
    plan = history.prepare_history(chats, 1, "bot-a")
    saved = history.finalize_history(plan, {"call2_output": "ok"})
    assert saved["stored_chars"] == 1_000
    assert sum(len(item["content"]) for item in saved["messages"]) == 1_000
    assert saved["messages"][-1]["content"] == "B" * 800


def test_search_and_soft_delete(isolated_history):
    chats = [
        _chat("user", "과거 질문" * 20),
        _chat("char", "파란 원피스를 입은 Alice" * 10),
    ]
    plan = history.prepare_history(chats, 1, "bot-a")
    history.finalize_history(plan, {"call2_output": "ok"})

    results = history.list_histories("Alice")
    assert [item["history_id"] for item in results] == [plan["history_id"]]
    destination = history.delete_history(plan["history_id"])
    assert os.path.isfile(destination)
    assert history.list_histories("Alice") == []
    assert history.get_history(plan["history_id"]) is None


def test_settings_trim_prunes_character_state_outside_retained_history(isolated_history):
    chats = [
        _chat("char", "A" * 600),
        _chat("user", "B" * 600),
        _chat("char", "C" * 600),
    ]
    plan = history.prepare_history(chats, 2, "bot-a")
    old_message_id = plan["proposed_messages"][0]["id"]
    history.finalize_history(plan, {
        "character_states_after": {
            "alice": {
                "canonical_name": "Alice",
                "last_seen_message_id": old_message_id,
                "current_wardrobe": {"body_state": "clothed", "worn": ["dress"]},
            },
        },
    })

    history.save_settings({
        "storage_max_chars": 1_000,
        "call1_history_chars": 1_000,
        "call2_fallback_history_chars": 1_000,
        "call3_fallback_history_chars": 1_000,
    })
    saved = history.get_history(plan["history_id"])
    assert saved is not None
    assert saved["stored_chars"] == 1_000
    assert saved["characters"] == {}
