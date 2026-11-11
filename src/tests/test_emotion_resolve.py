"""Emotion workshop: restored PROFILE-RESOLVE plus the anchor-level EMOTION-RESOLVE stage."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch
import unittest


def profile_cards():
    return {
        "Aria": {
            "name": "Aria",
            "default_visual_profile_id": "card_1",
            "profiles": [
                {"id": "card_1", "label": "카드 1", "appearance": ["red hair"], "default_outfit": ["white shirt"]},
                {"id": "card_2", "label": "카드 2", "appearance": ["red hair"], "default_outfit": ["black dress"]},
            ],
        },
        "Mina": {
            "name": "Mina",
            "default_visual_profile_id": "card_3",
            "profiles": [
                {"id": "card_3", "label": "카드 1", "appearance": ["blue hair"], "default_outfit": ["school uniform"]},
            ],
        },
    }


SLOTTED = (
    "Aria opens the door in the morning.\n\n[Slot 0]"
    "Mina waits by the gate, frightened.\n\n[Slot 1]"
    "Aria storms out of the hall in anger.\n\n[Slot 2]"
)


def previous_state():
    return {
        "aria": {
            "canonical_name": "Aria",
            "call1_status": json.dumps({"outfit": "winter coat"}),
        }
    }


def character_response():
    return {
        "characters": [{"name": "Aria"}, {"name": "Mina"}],
        "scene_plan": [
            {"anchor_segment": "C001", "character": "Aria", "emotion": "joy"},
            {"anchor_segment": "C002", "character": "Mina", "emotion": "fear"},
            {"anchor_segment": "C003", "character": "Aria", "emotion": "anger"},
        ],
    }


def profile_response():
    return {
        "characters": [{
            "name": "Aria",
            "profile_timeline": [
                {"at": "START", "profile_id": "card_1", "reason": "ordinary morning at home"},
                {"at": "C003", "profile_id": "card_2", "reason": "she left the house dressed for the hall"},
            ],
        }],
        "uncertainties": [],
    }


def emotion_response():
    return {
        "C001": {"character": "Aria", "outfit": "sunday dress"},
        "C002": {"character": "Mina", "outfit": "school uniform"},
        "C003": {"character": "Aria", "outfit": "evening gown"},
    }


class EmotionResolveTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        cls.modules = patch.dict(sys.modules, {"modes": package})
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")
        cls.scenes = importlib.import_module("modes.illustration_scene_pipeline")
        cls.prompts = {
            key: (Path(__file__).resolve().parents[1] / "prompts/lighbd" / name)
            .read_text(encoding="utf-8")
            for key, name in cls.pipeline.PROMPT_FILES.items()
        }

    async def resolve(self, *, workshop="emotion"):
        calls = []

        async def llm(name, messages, *args, **kwargs):
            calls.append((name, messages))
            if name == "CHARACTER-RESOLVE":
                return json.dumps(character_response())
            if name == "PROFILE-RESOLVE":
                # Multi-profile characters are judged one job at a time.
                content = "\n".join(str(message["content"]) for message in messages)
                self.assertIn("# TARGET CURRENT MULTI-PROFILE CHARACTER", content)
                self.assertIn("card_2", content)
                return json.dumps(profile_response())
            if name == "EMOTION-RESOLVE":
                return json.dumps(emotion_response())
            raise AssertionError(f"Unexpected stage: {name}")

        with patch.object(self.pipeline, "_call_pipeline_llm", llm), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts), \
             patch.object(self.pipeline, "_emotion_outfit_catalog", return_value={
                 "Aria": {
                     "card_1": ["white shirt", "sunday dress"],
                     "card_2": ["black dress", "evening gown"],
                 },
             }):
            _, result = await self.pipeline.resolve_profiles_before_generation(
                payload={
                    "session_id": "emotion-test",
                    "target_slotted": SLOTTED,
                    "chats": [{"role": "char", "data": SLOTTED}],
                },
                toggles={"profile_context_translate": False},
                history_plan={"state_before": previous_state()},
                visual_profiles=profile_cards(),
                profile_translation_namespace="Story",
                workshop=workshop,
            )
        return calls, result

    async def test_profile_stage_only_judges_multi_profile_characters(self):
        calls, result = await self.resolve()
        profile_calls = [messages for name, messages in calls if name == "PROFILE-RESOLVE"]
        self.assertEqual(len(profile_calls), 1)
        for character in result["characters"]:
            self.assertNotIn("outfit", character)
            self.assertNotIn("character_outfits", character)
        characters = {item["name"]: item for item in result["characters"]}
        self.assertEqual([event["profile_id"] for event in result["profile_events"]], ["card_1", "card_2"])
        self.assertTrue(all(event["character"] == "Aria" for event in result["profile_events"]))
        self.assertEqual(characters["Mina"]["profile_timeline"], [])

    async def test_emotion_stage_runs_once_with_anchor_level_context(self):
        calls, result = await self.resolve()
        emotion_calls = [messages for name, messages in calls if name == "EMOTION-RESOLVE"]
        self.assertEqual(len(emotion_calls), 1)
        prompt = "\n".join(str(message["content"]) for message in emotion_calls[0])
        for expected in (
            "C001", "C002", "C003", "joy", "fear", "anger",
            "card_1", "card_2", "card_3",
            "sunday dress", "evening gown",  # saved outfits for the active profile
            "winter coat",  # previous CURRENT outfit carried from tracked state
            "white shirt",  # profile default outfit
        ):
            self.assertIn(expected, prompt)
        self.assertEqual(result["anchor_outfits"], {
            "Aria": {"C001": "sunday dress", "C003": "evening gown"},
            "Mina": {"C002": "school uniform"},
        })
        self.assertEqual(result["character_outfits"], {
            "Aria": "sunday dress",  # earliest selected anchor
            "Mina": "school uniform",
        })
        self.assertEqual(result["outfit_last_anchor"], {"Aria": "C003", "Mina": "C002"})
        table = result["emotion_state_table"]
        self.assertEqual(table["slots"], ["C001", "C002", "C003"])
        self.assertEqual(table["changes"]["C003"]["Aria"], {"outfit": "evening gown"})
        # Inherited rows keep the previous anchor's outfit in the display table.
        self.assertEqual(table["states"]["C002"]["Aria"], {"outfit": "sunday dress"})
        self.assertEqual(table["states"]["C003"]["Mina"], {"outfit": "school uniform"})
        # The profile timeline itself is untouched; only its outfits are injected.
        events = {(event["character"], event["segment_id"]): event["outfit"]
                  for event in result["profile_events"]}
        self.assertEqual(events[("Aria", "START")], "sunday dress")
        self.assertEqual(events[("Aria", "C003")], "evening gown")

    async def test_missing_anchor_and_character_mismatch_fall_back_without_raising(self):
        parsed = self.pipeline._parse_emotion_resolution(
            json.dumps({
                "C001": {"character": "Aria", "outfit": "sunday dress"},
                "C003": {"character": "Somebody Else", "outfit": "evening gown"},
            }),
            character_response()["scene_plan"],
            {"C001": {"text": "one"}, "C002": {"text": "two"}, "C003": {"text": "three"}},
            ["Aria", "Mina"],
            previous_state(),
            {"profile_events": []},
            profile_cards(),
        )
        self.assertEqual(parsed["anchor_outfits"]["Aria"],
                         {"C001": "sunday dress", "C003": "winter coat"})
        self.assertNotIn("C002", parsed["anchor_outfits"].get("Mina", {}))
        self.assertEqual(len(parsed["validation_warnings"]), 2)
        self.assertTrue(any("C002" in warning for warning in parsed["validation_warnings"]))
        self.assertTrue(any("불일치" in warning for warning in parsed["validation_warnings"]))

    async def test_unregistered_scene_character_is_kept_without_crashing(self):
        scene_plan = [{"anchor_segment": "C001", "character": "Porter", "emotion": "fear"}]
        parsed = self.pipeline._parse_emotion_resolution(
            json.dumps({"C001": {"character": "Porter", "outfit": "blue coat"}}),
            scene_plan,
            {"C001": {"text": "one"}},
            ["Aria"],
            {},
            {"profile_events": []},
            profile_cards(),
        )
        self.assertEqual(parsed["anchor_outfits"], {"Porter": {"C001": "blue coat"}})
        self.assertEqual(parsed["validation_warnings"], [])

    async def test_active_profile_uses_the_outfit_of_each_anchor(self):
        _, result = await self.resolve()
        order = {"C001": 1, "C002": 2, "C003": 3}
        first = self.scenes.active_profile("Aria", "C001", result, profile_cards(), order)
        third = self.scenes.active_profile("Aria", "C003", result, profile_cards(), order)
        self.assertEqual(first["outfit"], [{"tag": "sunday dress"}])
        self.assertEqual(third["outfit"], [{"tag": "evening gown"}])
        # Same character, same profile card: only the anchor decides the outfit.
        self.assertEqual(first["visual_profile_id"], "card_1")
        self.assertEqual(third["visual_profile_id"], "card_2")

    async def test_persistence_saves_the_last_selected_anchor_outfit(self):
        _, result = await self.resolve()
        last_anchor = self.scenes._emotion_last_anchor(result, "Aria")
        self.assertEqual(last_anchor, "C003")
        self.assertEqual(
            self.scenes._emotion_outfit_override(result, "Aria", last_anchor),
            "evening gown",
        )

    async def test_normal_illustration_does_not_run_emotion_stage(self):
        calls, result = await self.resolve(workshop="illustration")
        self.assertNotIn("EMOTION-RESOLVE", [name for name, _ in calls])
        self.assertNotIn("anchor_outfits", result)
        self.assertNotIn("character_outfits", result)
        self.assertNotIn("emotion_state_table", result)
        for character in result["characters"]:
            self.assertNotIn("outfit", character)


if __name__ == "__main__":
    unittest.main()
