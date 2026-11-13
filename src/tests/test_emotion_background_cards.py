"""Place-based background reuse and deletion of individual saved cards."""
import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from emotion_background_bots import EmotionBackgroundBots
from emotion_composition import background_crop_box, character_box, normalize_composition_settings
from emotion_background_cards import (
    EmotionBackgroundCards, background_bot_key, compose_background, composition_preview, parse_card_predictions,
    register_background_cards, selected_background_cards, selection_instruction,
)
from settings_illustration import allowed_illustration_path


def scene(anchor, location, description, **extra):
    return {"anchor_segment": anchor, "background": {"location": location, "description": description, **extra}}


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


class PlaceSelectionTests(unittest.TestCase):
    def test_each_scene_keeps_its_point_code_and_view_center_without_splitting_a_place(self):
        context = {"composition_settings": normalize_composition_settings()}
        scenes = [scene("C001", "Library", "stone walls", view_center={"x": .2, "y": .4}, character_position="R"),
                  scene("C002", "Library", "stone walls", view_center={"x": .8, "y": .6}, character_position="L")]
        by_anchor, requests = selected_background_cards(scenes, context)
        self.assertEqual(len(requests), 1)
        self.assertNotIn("character_position", requests[0])
        self.assertNotIn("view_center", requests[0])
        self.assertEqual(by_anchor["C001"]["character_position"], "R")
        self.assertEqual(by_anchor["C002"]["character_position"], "L")
        self.assertEqual(by_anchor["C002"]["view_center"], {"x": .8, "y": .6})
        for invalid in (None, "right", "C", 50, {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                selected_background_cards([scene("C001", "Library", "stone walls", view_center={"x": .5, "y": .5}, character_position=invalid)], context)
        with self.assertRaises(ValueError):
            selected_background_cards([scene("C001", "Library", "stone walls", view_center={"x": .5, "y": .5})], context)
        instruction = selection_instruction(context)
        self.assertIn('"character_positions": {"R": 75, "M": 50, "L": 25}', instruction)
        self.assertNotIn('"visible_center_x"', instruction)

    def test_same_place_shares_card_despite_description_time_weather_and_case(self):
        by_anchor, requests = selected_background_cards([
            scene("C001", "Castle Hall", "Stone arches in morning light", name="Morning hall"),
            scene("C002", "castle_hall", "Rain at the windows at night", name="Rainy hall"),
        ], {"world": "castle", "cards": []})
        self.assertEqual(len(requests), 1)
        self.assertEqual(by_anchor["C001"], by_anchor["C002"])
        self.assertEqual(requests[0]["name"], "Castle Hall")

    def test_identity_is_stable_between_requests_but_distinct_for_other_places(self):
        first = selected_background_cards([scene("C001", "Library", "Daylight")], {})[1][0]
        later = selected_background_cards([scene("C008", "Library", "Candlelight")], {})[1][0]
        courtyard = selected_background_cards([scene("C002", "Courtyard", "Daylight")], {})[1][0]
        self.assertEqual(first["id"], later["id"])
        self.assertNotEqual(first["id"], courtyard["id"])

    def test_existing_legacy_card_is_reused_by_location_without_new_prediction(self):
        existing = {"id": "a" * 24, "name": "Library", "description": "Daylight", "positive": "stone library"}
        by_anchor, requests = selected_background_cards([
            scene("C001", "library", "Later at night"),
            {"anchor_segment": "C002", "background": {"id": existing["id"]}},
        ], {"cards": [existing]})
        self.assertEqual(requests, [])
        self.assertEqual(by_anchor["C001"], existing)
        self.assertEqual(by_anchor["C002"], existing)

    def test_editing_world_reference_does_not_create_another_version_of_a_place(self):
        first = selected_background_cards([scene("C001", "Library", "stone walls")], {"world": "castle"})[1][0]
        later = selected_background_cards([scene("C002", "Library", "stone arches")], {"world": "castle with arches"})[1][0]
        self.assertEqual(first["id"], later["id"])

    def test_prediction_preserves_location_and_each_anchor_receives_its_place(self):
        by_anchor, requests = selected_background_cards([
            scene("C001", "Library", "stone walls"), scene("C002", "Courtyard", "oak trees"),
        ], {})
        predicted = parse_card_predictions(json.dumps({"cards": [
            {"id": row["id"], "positive": "empty " + row["name"]} for row in requests
        ]}), requests)
        self.assertEqual(predicted[by_anchor["C001"]["id"]]["location"], "Library")
        self.assertEqual(predicted[by_anchor["C002"]["id"]]["positive"], "empty Courtyard")


class BackgroundCardStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_anchors_render_once_and_deleted_place_can_be_generated_again(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EmotionBackgroundCards(directory)
            card = selected_background_cards([scene("C001", "Library", "stone walls")], {})[1][0]
            card["positive"] = "empty library"
            render = AsyncMock(return_value=image_bytes())
            await asyncio.gather(*(store.ensure("b" * 24, card, "world", render) for _ in range(3)))
            self.assertEqual(render.await_count, 1)
            self.assertEqual(len(store.catalog("b" * 24)), 1)
            await store.delete("b" * 24, card["id"])
            self.assertEqual(store.catalog("b" * 24), [])
            self.assertFalse(store.image_path("b" * 24, card["id"]).exists())
            await store.ensure("b" * 24, card, "world", render)
            self.assertEqual(render.await_count, 2)


class CompositionPreviewTests(unittest.TestCase):
    def test_all_three_saved_center_points_drive_real_composition_and_keep_y_centered(self):
        settings = normalize_composition_settings({"character_positions": {"R": 64, "M": 38, "L": 9},
                                                  "character_height": 50, "character_offset_x": 500,
                                                  "character_offset_y": 500, "character_align_y": 0})
        with tempfile.TemporaryDirectory() as directory:
            background = Path(directory) / "background.png"
            Image.new("RGBA", (120, 80), "blue").save(background)
            cutout = Image.new("RGBA", (60, 100))
            cutout.paste("red", (20, 10, 40, 90))
            source = BytesIO()
            cutout.save(source, format="PNG")
            for code, left in (("R", 72), ("M", 41), ("L", 6)):
                with self.subTest(code=code):
                    self.assertEqual(character_box((20, 80), (120, 80), settings, position=code), (left, 20, 10, 40))
                    rendered = Image.open(BytesIO(compose_background(source.getvalue(), str(background), settings, character_position=code)))
                    self.assertEqual(rendered.getpixel((left, 20)), (255, 0, 0, 255))
                    self.assertEqual(rendered.getpixel((left, 19)), (0, 0, 255, 255))
                    self.assertEqual(rendered.getpixel((left + 10, 20)), (0, 0, 255, 255))

    def test_position_settings_are_independent_and_invalid_points_are_rejected(self):
        first = normalize_composition_settings()
        first["character_positions"]["R"] = 90
        self.assertEqual(normalize_composition_settings()["character_positions"], {"R": 75, "M": 50, "L": 25})
        for points in ({"R": 60, "M": 50}, {"R": 101, "M": 50, "L": 10},
                       {"R": True, "M": 50, "L": 10}, {"R": float("nan"), "M": 50, "L": 10}):
            with self.subTest(points=points), self.assertRaises(ValueError):
                normalize_composition_settings({"character_positions": points})

    def test_saved_alignment_is_preserved_and_same_size_character_can_move_outside(self):
        saved = normalize_composition_settings({"character_align_x": 25, "character_align_y": 75})
        self.assertEqual(character_box((20, 80), (120, 80), saved), (25, 0, 20, 80))
        moved = normalize_composition_settings({**saved, "character_offset_x": -50, "character_offset_y": 25})
        self.assertEqual(character_box((20, 80), (120, 80), moved), (-35, 20, 20, 80))
        same_size = normalize_composition_settings({"character_offset_x": 25, "character_offset_y": -50})
        self.assertEqual(character_box((120, 80), (120, 80), same_size), (30, -40, 120, 80))

    def test_scene_background_position_is_free_from_the_editor_preview_position(self):
        settings = normalize_composition_settings({"visible_width": 40, "visible_height": 50})
        predicted = {"x": .2, "y": .5}
        self.assertEqual(background_crop_box((1000, 800), settings, predicted), (0, 200, 400, 600))
        moved = normalize_composition_settings({**settings, "visible_center_x": 80, "visible_center_y": 50})
        self.assertEqual(background_crop_box((1000, 800), moved), (600, 200, 1000, 600))
        self.assertEqual(background_crop_box((1000, 800), moved, predicted), (0, 200, 400, 600))

    def test_render_keeps_outside_character_and_adds_transparent_space_around_cropped_background(self):
        with tempfile.TemporaryDirectory() as directory:
            background = Path(directory) / "background.png"
            Image.new("RGBA", (120, 80), "blue").save(background)
            cutout = Image.new("RGBA", (60, 100))
            cutout.paste("red", (20, 10, 40, 90))
            source = BytesIO()
            cutout.save(source, format="PNG")
            settings = {"character_align_x": 0, "character_offset_x": -10, "character_offset_y": 25}
            rendered = Image.open(BytesIO(compose_background(source.getvalue(), str(background), settings)))
            self.assertEqual(rendered.size, (132, 100))
            self.assertEqual(rendered.getpixel((0, 0)), (0, 0, 0, 0))
            self.assertEqual(rendered.getpixel((12, 0)), (0, 0, 255, 255))
            self.assertEqual(rendered.getpixel((0, 20)), (255, 0, 0, 255))
            self.assertEqual(rendered.getpixel((20, 20)), (0, 0, 255, 255))
            self.assertEqual(rendered.getpixel((0, 99)), (255, 0, 0, 255))
            self.assertEqual(rendered.getpixel((20, 99)), (0, 0, 0, 0))
            outside = Image.open(BytesIO(compose_background(source.getvalue(), str(background),
                                                            {**settings, "character_offset_x": -200})))
            self.assertEqual(outside.size, (360, 100))
            self.assertEqual(outside.getpixel((0, 99)), (255, 0, 0, 255))
            self.assertEqual(outside.getpixel((20, 20)), (0, 0, 0, 0))
            self.assertEqual(outside.getpixel((240, 20)), (0, 0, 255, 255))

    def test_oversized_figure_is_complete_at_every_point_with_alpha_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            background = Path(directory) / "background.png"
            Image.new("RGBA", (120, 80), "blue").save(background)
            cutout = Image.new("RGBA", (60, 100))
            cutout.paste((255, 0, 0, 128), (20, 10, 40, 90))
            source = BytesIO()
            cutout.save(source, format="PNG")
            settings = {"character_height": 200, "character_positions": {"R": 100, "M": 50, "L": 0}}
            for code, output_width, person_x, background_x in (("R", 140, 100, 0), ("M", 120, 40, 0), ("L", 140, 0, 20)):
                with self.subTest(code=code):
                    rendered = Image.open(BytesIO(compose_background(source.getvalue(), str(background), settings, character_position=code)))
                    self.assertEqual(rendered.size, (output_width, 160))
                    self.assertEqual(rendered.getpixel((person_x, 0)), (255, 0, 0, 128))
                    self.assertEqual(rendered.getpixel((person_x + 39, 159)), (255, 0, 0, 128))
                    empty_x = output_width - 1 if code == "L" else 0
                    self.assertEqual(rendered.getpixel((empty_x, 0)), (0, 0, 0, 0))
                    self.assertEqual(rendered.getpixel((background_x + (119 if code == "L" else 0), 40)), (0, 0, 255, 255))

    def test_editable_layers_preserve_transparency_and_match_final_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            background = Path(directory) / "background.png"
            Image.new("RGBA", (120, 80), "blue").save(background)
            cutout = Image.new("RGBA", (60, 100))
            cutout.paste((255, 0, 0, 128), (20, 10, 40, 90))
            source = BytesIO()
            cutout.save(source, format="PNG")
            settings = {"visible_width": 50, "visible_height": 50, "character_height": 300,
                        "character_align_x": 25, "character_align_y": 75}
            result = composition_preview({"base": base64.b64encode(source.getvalue()).decode(),
                                          "composition_settings": settings}, background)
            self.assertEqual(result["character_size"], {"width": 20, "height": 80})
            layer = Image.open(BytesIO(base64.b64decode(result["character_image"].split(",")[1])))
            self.assertEqual(layer.size, (20, 80))
            self.assertEqual(layer.getpixel((0, 0)), (255, 0, 0, 128))
            rendered = Image.open(BytesIO(base64.b64decode(result["image"].split(",")[1])))
            final = Image.open(BytesIO(compose_background(source.getvalue(), str(background), settings)))
            self.assertEqual(rendered.size, (60, 120))
            self.assertEqual(result["output_bounds"], {"x": 0, "y": -60, "width": 60, "height": 120})
            self.assertEqual((result["background_width"], result["background_height"]), (120, 80))
            self.assertEqual(rendered.getpixel((0, 0)), (0, 0, 0, 0))
            self.assertEqual(rendered.getpixel((8, 119)), (255, 0, 0, 128))
            self.assertEqual(rendered.tobytes(), final.tobytes())
            empty = composition_preview({}, background)
            self.assertIsNone(empty["character_image"])
            self.assertIsNone(empty["character_size"])


class BackgroundCardApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        builtin = self.root / "builtin.json"
        builtin.write_text(json.dumps({"version": 1, "bots": [{"name": "Shipped"}]}), encoding="utf-8")
        self.bots = EmotionBackgroundBots(self.root / "config.json", builtin)
        self.bots.change({"action": "create", "name": "User"})
        self.store = EmotionBackgroundCards(self.root / "cards")
        self.keys = {bot["name"]: background_bot_key(bot) for bot in self.bots.snapshot()["bots"]}
        self.card = {"id": "a" * 24, "name": "Library", "description": "stone walls", "positive": "library"}
        for name, key in self.keys.items():
            await self.store.ensure(key, self.card, "world", AsyncMock(return_value=image_bytes()))
        app = web.Application()
        with patch("emotion_background_cards.EmotionBackgroundCards", return_value=self.store):
            register_background_cards(SimpleNamespace(app=app), self.bots)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.path = "/api/illustration/background-cards?scope=emotion"

    async def test_corner_delete_removes_only_target_card_and_image_from_target_preset(self):
        second = {**self.card, "id": "c" * 24, "name": "Courtyard"}
        await self.store.ensure(self.keys["User"], second, "world", AsyncMock(return_value=image_bytes()))
        response = await self.client.post(self.path, json={"action": "delete", "bot": "User", "id": self.card["id"]})
        self.assertEqual(response.status, 200)
        self.assertEqual([card["name"] for card in (await response.json())["cards"]], ["Courtyard"])
        self.assertFalse(self.store.image_path(self.keys["User"], self.card["id"]).exists())
        self.assertEqual(len(self.store.catalog(self.keys["Shipped"])), 1)

    async def test_shipped_card_deletion_requires_its_background_unlock_token(self):
        body = {"action": "delete", "bot": "Shipped", "id": self.card["id"]}
        self.assertEqual((await self.client.post(self.path, json=body)).status, 400)
        token = self.bots.change({"action": "unlock", "confirmed": True})["builtin_unlock_token"]
        response = await self.client.post(self.path, json={**body, "builtin_unlock_token": token})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.store.catalog(self.keys["Shipped"]), [])

    async def test_scope_gateway_identifiers_and_missing_cards(self):
        self.assertTrue(allowed_illustration_path("/api/illustration/background-cards", "POST"))
        for body, status in (
            ([], 400), ({"action": "delete", "id": self.card["id"]}, 400),
            ({"action": "delete", "bot": "User", "id": "../outside"}, 400),
            ({"action": "delete", "bot": "User", "id": "d" * 24}, 404),
            ({"action": "delete", "bot": "Missing", "id": self.card["id"]}, 404),
        ):
            with self.subTest(body=body):
                self.assertEqual((await self.client.post(self.path, json=body)).status, status)
        self.assertEqual((await self.client.post("/api/illustration/background-cards", json={})).status, 400)
        self.assertEqual(len(self.store.catalog(self.keys["User"])), 1)


if __name__ == "__main__":
    unittest.main()
