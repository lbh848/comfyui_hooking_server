"""NPC registration, gender choices, STEP 2 handoff and 30-flow expiry."""
import asyncio
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image


class NpcGenerationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Avoid application startup in modes.__init__; load the real modules.
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        cls.modules = patch.dict(sys.modules, {"modes": package})
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        cls.npc = importlib.import_module("modes.illustration_npc")
        cls.storage = importlib.import_module("modes.bot_mode")
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")
        cls.scenes = importlib.import_module("modes.illustration_scene_pipeline")
        cls.builders = importlib.import_module("modes.illustration_prompt_builders")
        root = Path(__file__).resolve().parents[1]

        def prompt_text(name):
            # 감정 삽화 전용 지침은 defaults/emotion/prompts 에만 배포된다.
            for folder in (root / "prompts/lighbd", root / "defaults/emotion/prompts"):
                path = folder / name
                if path.is_file():
                    return path.read_text(encoding="utf-8")
            raise FileNotFoundError(name)

        cls.prompts = {key: prompt_text(name)
                       for key, name in cls.pipeline.PROMPT_FILES.items()}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "bot.json"
        self.extra = []
        self.write({"bots": [{"name": "Story", "characters": [{"name": "Manual"}],
                              "character_auto_settings_solo": {"img_w": 960, "seed": 17}}]})
        for field, value in (
            ("BOT_DIR", str(self.directory / "bot")),
            ("comfy_input_root", lambda base: self.directory / "input"),
            ("_load_bot_data", self.read), ("_save_bot_data", self.write),
            ("_load_lb_extra", lambda name: deepcopy(self.extra)),
            ("_save_lb_extra", self.save_extra),
            ("_load_patch_settings", lambda name: {"face_crop_top": 1.0, "face_crop_bottom": 1.0}),
        ):
            patcher = patch.object(self.storage, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = self.storage.BotMode()
        self.npc._bot_locks.clear()
        self.candidates = [{"name": "Porter", "aliases": ["짐꾼"], "description": "The porter handing over a key.", "anchors": ["C001"]}]
        self.response = {"characters": [{"name": "Porter", "gender_tag": "1male",
            "appearance": ["brown hair", "broad shoulders"], "default_outfit": ["blue coat"],
            "face_tags": ["square face", "brown hair"], "eye_tags": ["green eyes"],
            "visual_context": "A broad-shouldered porter wearing a blue coat."}]}
        image = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(image, format="PNG")
        self.pixels = image.getvalue()

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def save_extra(self, name, data):
        self.extra = deepcopy(data)

    def root(self, name="Porter"):
        return next(entry for entry in self.read()["bots"][0]["characters"] if entry["name"] == name)

    async def generate(self, render=None):
        return await self.npc.generate_cards(manager=self.manager, bot_name="Story", candidates=self.candidates,
            prompts=self.prompts, past_text="", segmented="[C001] A porter hands over a key.",
            toggles={}, stream_notify=None, render_image=render or AsyncMock(return_value={
                "image_bytes": self.pixels, "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]}))

    async def test_gender_dropdown_values_are_enforced(self):
        for gender in self.npc.NPC_GENDER_TAGS:
            self.response["characters"][0]["gender_tag"] = gender
            root = self.npc.parse_cards(self.response, self.candidates)[0]
            self.assertEqual(root["gender_tag"], gender)
            self.assertEqual(root["visual_cards"][0]["gender_tag"], gender)
        for invalid in ("unknown", "female", "", None):
            self.response["characters"][0]["gender_tag"] = invalid
            with self.assertRaises(ValueError):
                self.npc.parse_cards(self.response, self.candidates)

    async def test_partial_npc_card_response_still_registers_valid_cards(self):
        self.candidates = [*self.candidates,
                           {"name": "Guide", "aliases": ["안내인"], "description": "A guide waiting outside.", "anchors": ["C001"]}]
        llm = AsyncMock(return_value=json.dumps(self.response))
        with patch.object(self.pipeline, "_call_pipeline_llm", llm):
            result = await self.generate()
        self.assertEqual(result["created"], ["Porter"])
        self.assertEqual([entry["name"] for entry in self.read()["bots"][0]["characters"]], ["Manual", "Porter"])
        self.assertFalse((self.directory / "bot/Story/Guide").exists())

    async def test_registers_only_after_representative_success_and_reuses_card(self):
        async def render(root, bot):
            self.assertNotIn("Porter", [entry["name"] for entry in self.read()["bots"][0]["characters"]])
            settings = self.storage.resolve_illust_settings(bot, generation_mode="character_auto")
            self.assertEqual((settings["img_w"], settings["seed"]), (960, 17))
            from modes.illust_prompt_builder import IllustPromptBuilder
            builder = IllustPromptBuilder()
            self.assertEqual(builder.build_cache_path(["Porter"], "Story", bot["characters"]), {"list": []})
            self.assertEqual(builder.build_face_id_dir(["Porter"], "Story", settings, bot["characters"]), {"list": []})
            return {"image_bytes": self.pixels, "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]}
        llm = AsyncMock(return_value=json.dumps(self.response))
        with patch.object(self.pipeline, "_call_pipeline_llm", llm):
            result = await self.generate(render)
            root = self.root()
            self.assertEqual(result["created"], ["Porter"])
            self.assertEqual((root["npc_generated"], root["npc_idle_calls"]), (True, 0))
            directory = self.directory / "bot/Story/Porter"
            with Image.open(directory / root["rep_images"][0]) as representative:
                self.assertEqual(representative.size, (16, 16))
            with Image.open(directory / "_face_image.webp") as face:
                self.assertEqual(face.size, (8, 8))
            self.assertEqual(self.extra[0]["outfit"], [{"tag": "blue coat"}])
            visible = self.storage._character_card_data(root, None)["character"]
            self.assertEqual((visible["npc_generated"], visible["npc_lifetime"]), (True, 30))
            again = await self.generate(AsyncMock(side_effect=AssertionError("must reuse representative")))
            self.assertEqual(again["created"], [])
            self.assertEqual(llm.await_count, 1)

    async def test_rename_generated_character_moves_directory_and_portable_entry(self):
        data = self.read()
        data["bots"][0]["characters"].append({"name": "Porter", "npc_generated": True, "npc_idle_calls": 0})
        self.write(data)
        (self.directory / "bot/Story/Porter").mkdir(parents=True)
        self.extra = [{"name": "Porter", "outfit": [{"tag": "blue coat"}]}]
        response = await self.manager._rename_character(self.read(), {
            "bot_name": "Story", "old_name": "Porter", "new_name": "Courier",
        })
        self.assertEqual(response.status, 200)
        self.assertEqual([entry["name"] for entry in self.read()["bots"][0]["characters"]], ["Manual", "Courier"])
        self.assertFalse((self.directory / "bot/Story/Porter").exists())
        self.assertTrue((self.directory / "bot/Story/Courier").is_dir())
        self.assertEqual(self.extra[0]["name"], "Courier")

    async def test_representative_failure_does_not_register_partial_character(self):
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                await self.generate(AsyncMock(side_effect=RuntimeError("render failed")))
        self.assertEqual(self.read()["bots"][0]["characters"], [{"name": "Manual"}])
        self.assertFalse((self.directory / "bot/Story/Porter").exists())

    async def test_face_uses_tracking_of_its_own_representative_without_replacing_it(self):
        import hashlib
        image = Image.new("RGB", (100, 80), "red")
        image.paste("blue", (50, 0, 100, 80))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        faces = [{"name": "Other", "bbox": [.1, .25, .4, .75]},
                 {"name": "Porter", "bbox": [.6, .25, .9, .75]}]
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate(AsyncMock(return_value={"image_bytes": stream.getvalue(), "face_positions": faces}))
        root = self.root()
        directory = self.directory / "bot/Story/Porter"
        rep = directory / root["rep_images"][0]
        before = rep.read_bytes()
        with Image.open(rep) as result:
            self.assertEqual(result.size, (100, 80))
            self.assertEqual(result.tobytes(), image.tobytes())
        with Image.open(directory / "_face_image.webp") as result:
            self.assertEqual(result.size, (30, 40))
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
        record = json.loads((directory / "_face_image_source.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source_filename"], rep.name)
        self.assertEqual(record["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(record["face_sha256"], hashlib.sha256((directory / "_face_image.webp").read_bytes()).hexdigest())
        self.assertEqual(record["bbox"], faces[1]["bbox"])
        self.assertEqual(len(list(directory.glob("npc_*.png"))), 1)
        self.assertTrue(self.npc.is_generated_image(rep.name, {"source": "character_auto"}, npc_generated=True))
        self.assertEqual(rep.read_bytes(), before)

    async def test_missing_tracking_keeps_representative_without_separate_face_job(self):
        render = AsyncMock(return_value={"image_bytes": self.pixels, "face_positions": []})
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate(render)
        render.assert_awaited_once()
        directory = self.directory / "bot/Story/Porter"
        self.assertTrue((directory / self.root()["rep_images"][0]).exists())
        self.assertFalse((directory / "_face_image.webp").exists())

    async def test_invalid_tracks_do_not_crop_and_clear_stale_face_only(self):
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate()
        directory = self.directory / "bot/Story/Porter"
        filename = self.root()["rep_images"][0]
        before = (directory / filename).read_bytes()
        for box in ([0, 0, float("nan"), 1], [1, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 1]):
            self.assertFalse(self.npc.save_tracked_face(directory=directory, source_filename=filename,
                character_name="Porter", face_positions=[{"name": "Porter", "bbox": box}], settings={}))
        self.assertFalse((directory / "_face_image.webp").exists())
        self.assertEqual((directory / filename).read_bytes(), before)
        self.assertEqual(self.root()["rep_images"], [filename])

    async def test_representative_tracking_context_keeps_solo_mask_enabled(self):
        from modes.multi_char_mask import normalize_multi_char_snapshot, should_skip_solo_region_mask, render_region_mask
        root = self.npc.parse_cards(self.response, self.candidates)[0]
        context = self.npc.character_tracking_context(root)
        snapshot = normalize_multi_char_snapshot(context)
        self.assertEqual(snapshot["character_order"], ["Porter"])
        self.assertGreaterEqual(snapshot["layout"]["mask_render_version"], 5)
        self.assertFalse(should_skip_solo_region_mask(context["layout"], context))
        mask = render_region_mask(context["layout"], 16, 16)
        self.assertEqual(mask.getpixel((8, 8)), (255, 0, 0))
        self.assertEqual(context["characters"], self.npc.character_image_descriptor(root)["characters"])

    async def test_representative_descriptor_can_carry_preset_emotion_tags(self):
        root = self.npc.parse_cards(self.response, self.candidates)[0]
        default = self.npc.character_image_descriptor(root)["characters"][0]["positive"]
        tagged = self.npc.character_image_descriptor(
            root, emotion_tags=["smile-tag", "pose-tag"])["characters"][0]["positive"]
        self.assertIn("smile-tag", tagged)
        self.assertIn("pose-tag", tagged)
        self.assertNotIn("smile-tag", default)

    async def test_generated_face_is_embedded_alongside_original_representative(self):
        import shutil
        from modes.illust_prompt_builder import IllustPromptBuilder
        from illustration_json import build_illustration_json
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate()
        root = self.root()
        source = self.directory / "bot/Story/Porter" / root["rep_images"][0]
        staged = self.directory / "input/soya_bot/Story/Porter"
        staged.mkdir(parents=True)
        shutil.copy2(source, staged / "representation.png")
        builder = IllustPromptBuilder()
        self.assertEqual(builder.build_cache_path(["Porter"], "Story", [root]), {"list": []})
        refs = builder.build_face_id_dir(["Porter"], "Story", {}, [root])
        prompt = json.dumps({"schema": "soya.illustration.v2", "fields": {
            "CHAR_LIST": ["Porter"], "FACE_ID_DIR": refs,
            "SNAPREF": {"enabled": True, "secondary_reference": "face"}}, "assets": {}})
        transport = json.loads(build_illustration_json(prompt, "", input_dir=str(self.directory / "input"),
                                                       bot_dir=str(self.directory / "bot")))
        references = transport["fields"]["SNAPREF"]["references"]
        self.assertEqual(len(references), 2)
        self.assertNotEqual(references[0]["asset"], references[0]["face_asset"])
        self.assertEqual(references[0]["face_asset"], references[1]["asset"])
        self.assertEqual(source.read_bytes(), (staged / "representation.png").read_bytes())

    async def test_queue_prepares_representative_mask_before_generation(self):
        from queue_manager import QueueManager
        (self.directory / "input").mkdir()
        root = self.npc.parse_cards(self.response, self.candidates)[0]
        received = []
        async def generate(prompt_id, prompt_data, raw_body, **kwargs):
            received.append(deepcopy(raw_body))
            mask_path = self.directory / "input/region_mask/region_mask.png"
            self.assertTrue(mask_path.is_file())
            context = raw_body["illustration_multi_char"]
            self.assertEqual(context["characters"][0]["name"], root["name"])
            self.assertFalse(context["single_char_mask_skip_enabled"])
        queue = SimpleNamespace(get_config=lambda: {"comfy_input_dir": str(self.directory / "input"),
                "illustration_context_toggles": {"single_char_mask_skip_enabled": True}},
            process_prompt_full=generate)
        item = SimpleNamespace(id="representative", params={"prompt_id": "representative", "npc_tracking_root": root,
            "raw_body": {"illustration_provider": "comfy", "illustration_prompt_format": "v3"}})
        result = await QueueManager._handle_illustration(queue, item)
        self.assertTrue(result["success"])
        self.assertEqual(len(received), 1)

    async def test_render_handoff_preserves_its_own_tracking_before_discard(self):
        import ast
        import copy
        from test_illustration_submission import TREE, SOURCE
        root = self.npc.parse_cards(self.response, self.candidates)[0]
        render_bot = {"characters": [root]}
        faces = [{"name": "Porter", "bbox": [.1, .2, .4, .6]}]
        prompts = {"representative": {"_deferred_finalize": {"conversion_info": {"face_positions": faces}}}}
        enqueue = AsyncMock(return_value=("representative", SimpleNamespace()))
        def discard(*args):
            faces.clear()
            prompts.clear()
        namespace = {"illustration_context_pipeline": self.pipeline,
            "runtime_snapshot": {"workshop": "illustration"},
            "illust_toggles": {"prompt_format": "v3"}, "_enqueue_child": enqueue,
            "_await_child": AsyncMock(return_value=(self.pixels, "", False)),
            "prompts": prompts, "copy": copy, "asyncio": asyncio,
            "_discard_deferred_illustration_prompt": discard}
        node = copy.deepcopy(next(node for node in ast.walk(TREE)
                                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "render_npc_image"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        with patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            result = await namespace["render_npc_image"](root, render_bot)
        self.assertEqual(result["image_bytes"], self.pixels)
        self.assertEqual(result["face_positions"], [{"name": "Porter", "bbox": [.1, .2, .4, .6]}])
        self.assertEqual(enqueue.await_args.kwargs["npc_render_bot"], render_bot)
        self.assertEqual(prompts, {})

    async def test_render_injects_preset_representative_emotion_in_emotion_workshop(self):
        import ast
        import copy
        from test_illustration_submission import TREE, SOURCE
        root = self.npc.parse_cards(self.response, self.candidates)[0]
        render_bot = {"characters": [root]}
        enqueue = AsyncMock(return_value=("representative", SimpleNamespace()))
        namespace = {"illustration_context_pipeline": self.pipeline,
            "runtime_snapshot": {"workshop": "emotion"},
            "illust_toggles": {"prompt_format": "v3", "emotion_representative": {
                "preset": "smile", "name": "기쁨", "rating": "sfw"}},
            "_enqueue_child": enqueue,
            "_await_child": AsyncMock(return_value=(self.pixels, "", False)),
            "prompts": {}, "copy": copy, "asyncio": asyncio,
            "_discard_deferred_illustration_prompt": lambda *args: None}
        node = copy.deepcopy(next(node for node in ast.walk(TREE)
                                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "render_npc_image"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        with patch.object(self.pipeline, "_emotion_expression_catalog",
                          return_value={"smile": ["smile-tag", "pose-tag"]}), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            await namespace["render_npc_image"](root, render_bot)
        descriptor = enqueue.await_args.args[0]
        positive = descriptor["characters"][0]["positive"]
        self.assertIn("smile-tag", positive)
        self.assertIn("pose-tag", positive)

    async def test_npc_preparation_stages_representative_without_face_utility(self):
        import ast
        import copy
        from aiohttp import web
        from test_illustration_submission import TREE, SOURCE
        stage = AsyncMock(return_value=web.json_response({"visual_targets": [{"visual_card_id": "card_1"}]}))
        queue = SimpleNamespace(add_item=AsyncMock(side_effect=AssertionError("no separate FACE generation")))
        namespace = {"illustration_provider_snapshot": "comfy", "active_bot": "Story",
                     "data_patcher": SimpleNamespace(handle_data_patch=stage), "queue_manager": queue}
        node = deepcopy(next(node for node in ast.walk(TREE)
                             if isinstance(node, ast.AsyncFunctionDef) and node.name == "prepare_npc_character"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        await namespace["prepare_npc_character"]({"name": "Porter"})
        stage.assert_awaited_once()
        queue.add_item.assert_not_awaited()

    async def test_explicit_regeneration_refreshes_face_from_same_new_image(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from settings_illustration_regeneration import register_character_image_regeneration
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate()
        data = self.read()
        data["bots"][0]["characters"][-1].pop("npc_face_source")
        self.write(data)
        filename = self.root()["rep_images"][0]
        rendered = io.BytesIO()
        Image.new("RGB", (32, 24), "red").save(rendered, format="PNG")
        server = SimpleNamespace(app=web.Application(), prompts={}, bot_mode=self.manager, app_config={},
            data_patcher=SimpleNamespace(handle_data_patch=AsyncMock(return_value=web.json_response(
                {"visual_targets": [{"visual_card_id": "card_1"}]}))),
            _capture_illustration_runtime_snapshot=lambda config: {"provider": "comfy",
                "illustration_workflow_type": "v3_anima", "illustration_context_toggles": {}},
            _discard_deferred_illustration_prompt=lambda key, reason: None)
        async def enqueue(kind, label, params, **kwargs):
            self.assertEqual(kind, "illustration", "regeneration must not enqueue a separate FACE utility")
            self.assertEqual(params["npc_tracking_root"]["name"], "Porter")
            server.prompts[params["prompt_id"]].update(_deferred_image_bytes=rendered.getvalue(),
                _deferred_finalize={"conversion_info": {"face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]}})
            future = asyncio.get_running_loop().create_future()
            future.set_result({"success": True})
            return SimpleNamespace(id="regenerated", completion_future=future, status="completed")
        server.queue_manager = SimpleNamespace(add_item=AsyncMock(side_effect=enqueue))
        register_character_image_regeneration(server)
        client = TestClient(TestServer(server.app))
        await client.start_server()
        try:
            with patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
                response = await client.post("/api/bot_mode/regenerate_image", json={"bot_name": "Story",
                    "char_name": "Porter", "filename": filename, "visual_card_id": "card_1"})
                self.assertEqual(response.status, 202)
                for _ in range(100):
                    status = await (await client.get("/api/bot_mode/regenerate_image?bot_name=Story&char_name=Porter")).json()
                    if status["run"]["status"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(.01)
            self.assertEqual(status["run"]["status"], "completed", status)
            self.assertEqual(server.queue_manager.add_item.await_count, 1)
            self.assertEqual(self.root()["npc_face_source"], "representative_mask_tracking")
            directory = self.directory / "bot/Story/Porter"
            with Image.open(directory / filename) as representative:
                self.assertEqual(representative.size, (32, 24))
            with Image.open(directory / "_face_image.webp") as face:
                self.assertEqual(face.size, (16, 12))
                self.assertEqual(face.getpixel((0, 0)), (255, 0, 0))
        finally:
            await client.close()

    async def test_expiry_at_30_resets_on_use_and_preserves_manual_and_other_bots(self):
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(self.response))):
            await self.generate()
        prepared = self.directory / "input/soya_bot/Story/Porter"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "cache.pt").write_bytes(b"npc reference cache")
        data = self.read()
        data["bots"].append({"name": "Other", "characters": [{"name": "OtherNpc", "npc_generated": True, "npc_idle_calls": 29}]})
        self.write(data)
        for index in range(29):
            self.assertEqual(await self.npc.advance_lifetimes(self.manager, "Story", [], str(index)), [])
        self.assertEqual(self.root()["npc_idle_calls"], 29)
        await self.npc.advance_lifetimes(self.manager, "Story", [], "28")
        self.assertEqual(self.root()["npc_idle_calls"], 29)
        # Reappearance on the expiry boundary resets the full lifetime.
        await self.npc.advance_lifetimes(self.manager, "Story", ["Porter"], "reappeared")
        self.assertEqual(self.root()["npc_idle_calls"], 0)
        for index in range(29):
            await self.npc.advance_lifetimes(self.manager, "Story", [], f"again-{index}")
        self.assertEqual(await self.npc.advance_lifetimes(self.manager, "Story", [], "expired"), ["Porter"])
        self.assertEqual(self.read()["bots"][0]["characters"], [{"name": "Manual"}])
        self.assertEqual(self.read()["bots"][1]["characters"][0]["npc_idle_calls"], 29)
        self.assertFalse((self.directory / "bot/Story/Porter").exists())
        self.assertFalse(prepared.exists())
        self.assertEqual(self.extra, [])

    def test_npc_candidate_anchor_typos_are_normalized(self):
        segments = {"C001": {}, "C045": {}}
        candidates = self.npc.parse_candidates(
            [{"name": "Rose", "description": "A maid serving the table.", "anchors": ["c45", "C045"]}],
            [], segments)
        self.assertEqual(candidates[0]["anchors"], ["C045"])

    def test_repair_mode_drops_candidates_without_valid_anchors(self):
        candidates = self.npc.parse_candidates(
            [{"name": "Rose", "description": "A maid serving the table.", "anchors": ["C999"]}],
            [], {"C001": {}}, repair=True)
        self.assertEqual(candidates, [])
        with self.assertRaises(ValueError):
            self.npc.parse_candidates(
                [{"name": "Rose", "description": "A maid serving the table.", "anchors": ["C999"]}],
                [], {"C001": {}})

    def test_npc_contract_is_sent_once_in_the_user_message(self):
        toggles = {"character_limit": 1, "nsfw": False, "illustration_output_mode": "illustration"}
        selection = self.builders.selection_messages(self.prompts, "role context", "2", toggles, {}, "", "")
        messages = self.builders.resolution_messages(False, ["Manual"], {}, "", "current", "", "", "character system",
            selection=selection, anchors=["C001"])
        self.assertNotIn("# NPC REGISTRATION EXTENSION", messages[0]["content"])
        self.assertIn("npc_candidates is a required top-level array", messages[0]["content"])
        self.assertEqual(messages[1]["content"].count("# NPC REGISTRATION EXTENSION"), 1)
        self.assertIn("Return only a JSON object with characters, npc_candidates and scene_plan arrays.",
                      messages[1]["content"])

    async def test_character_resolution_salvages_invalid_npc_candidates(self):
        raw = {"characters": [], "npc_candidates": [
                   self.candidates[0],
                   {"name": "Bad Name", "description": "An invalid candidate.", "anchors": ["C001"]}],
               "scene_plan": [{"anchor_segment": "C001", "characters": ["Porter", "Bad Name"],
                               "scene_brief": "The porter introduces a companion."}]}
        payload = {"session_id": "npc-test", "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                   "chats": [{"role": "char", "data": "A porter hands over a key."}]}
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(raw))), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            _, result = await self.pipeline.resolve_profiles_before_generation(payload=payload,
                toggles={"npc_generation_enabled": True}, history_plan=None, visual_profiles={})
        self.assertEqual(result["npc_candidates"], self.candidates)
        self.assertEqual(result["scene_plan"][0]["characters"], ["Porter"])

    async def test_character_resolution_preserves_npcs_through_profile_handoff(self):
        raw = {"characters": [], "npc_candidates": self.candidates,
               "scene_plan": [{"anchor_segment": "C001", "characters": ["Porter"], "scene_brief": "The porter hands over a key."}]}
        payload = {"session_id": "npc-test", "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                   "chats": [{"role": "char", "data": "A porter hands over a key."}]}
        with patch.object(self.pipeline, "_call_pipeline_llm", AsyncMock(return_value=json.dumps(raw))), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            _, result = await self.pipeline.resolve_profiles_before_generation(payload=payload,
                toggles={"npc_generation_enabled": True}, history_plan=None, visual_profiles={})
        self.assertEqual(result["npc_candidates"], self.candidates)
        self.assertEqual(result["current_characters"][0]["name"], "Porter")
        self.assertEqual(result["scene_plan"][0]["characters"], ["Porter"])

    async def test_step2_parallel_join_passes_new_card_to_detail_with_plan_off(self):
        await self.check_step2_handoff(False)

    async def test_plan_and_detail_continue_during_representative_generation(self):
        await self.check_step2_handoff(True)

    async def check_step2_handoff(self, plan_enabled):
        started = set()
        render_done = asyncio.Event()
        detail_done = asyncio.Event()
        prepared = asyncio.Event()
        dispatched = []
        trackers_done = asyncio.Event()
        result = self.pipeline._empty_profile_result()
        result.update(npc_candidates=self.candidates, current_characters=[{"name": "Porter", "confidence": 1.0}],
            characters=[{"name": "Porter", "in_history": False, "profile_timeline": []}],
            scene_plan=[{"anchor_segment": "C001", "characters": ["Porter"], "scene_brief": "The porter hands over a key."}])
        async def render(root, bot):
            await asyncio.wait_for(trackers_done.wait(), 3)
            await asyncio.wait_for(detail_done.wait(), 3)
            render_done.set()
            return {"image_bytes": self.pixels, "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]}
        async def prepare(root):
            self.assertTrue(render_done.is_set())
            self.assertTrue(self.root()["npc_generated"])
            self.assertTrue((self.directory / "bot/Story/Porter/_face_image.webp").exists())
            prepared.set()
        async def dispatch(ready):
            self.assertTrue(prepared.is_set(), "scene GPU waits for representative and FACE from the same generation")
            dispatched.extend(ready["items"])
        async def llm(name, messages, *args, **kwargs):
            if name in ("OUTFIT", "TIME & PLACE", "NPC 생성 노드"):
                started.add(name)
                if len(started) == 3:
                    trackers_done.set()
                await asyncio.wait_for(trackers_done.wait(), 3)
            if name == "NPC 생성 노드":
                return json.dumps(self.response)
            if name == "OUTFIT":
                return json.dumps({"SLOT0": {"Porter": {"outfit": "blue coat", "hair": "brown hair"}}})
            if name == "TIME & PLACE":
                return json.dumps({"SLOT0": {"Porter": {"time": "day", "place": "street", "lighting": "sunlight"}}})
            if name.startswith("CALL2-PLAN"):
                self.assertFalse(render_done.is_set())
                return json.dumps({"scene_plan": result["scene_plan"]})
            if name.startswith("CALL2-DETAIL"):
                self.assertFalse(render_done.is_set())
                self.assertNotIn("Porter", [entry["name"] for entry in self.read()["bots"][0]["characters"]])
                content = "\n".join(str(message["content"]) for message in messages)
                for expected in ("green eyes", "square face", "1male", "blue coat"):
                    self.assertIn(expected, content)
                detail_done.set()
                return json.dumps({"scenes": [{"slot": 0, "camera": "cowboy shot", "scene": "street", "supplement": "",
                    "characters": [{"name": "Porter", "positive": "1male, brown hair, green eyes, blue coat", "negative": ""}]}]})
            raise AssertionError(f"Unexpected stage: {name}")
        with patch.object(self.pipeline, "_call_pipeline_llm", llm), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            output = await asyncio.wait_for(self.pipeline.build_from_context(
                {"session_id": "npc-test", "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                 "chats": [{"role": "char", "data": "A porter hands over a key."}]},
                {"npc_generation_enabled": True, "outfit_enabled": True, "time_place_enabled": True,
                 "call2_plan_enabled": plan_enabled, "key_visual": False, "speak_enabled": False,
                 "multi_char_mask_enabled": False, "call2_detail_parallel_slow_retry_enabled": False}, "",
                visual_profiles={}, profile_translation_namespace="Story", pre_resolved_profile_result=result,
                npc_manager=self.manager, render_npc_image=render, prepare_npc_character=prepare,
                on_call2_ready=dispatch), 10)
        self.assertEqual(started, {"OUTFIT", "TIME & PLACE", "NPC 생성 노드"})
        self.assertEqual(output["npc_generation_result"]["created"], ["Porter"])
        self.assertEqual(output["items"][0]["characters"][0]["name"], "Porter")
        self.assertEqual(len(dispatched), 1)

    async def test_npc_representative_comfy_waits_for_render_gate(self):
        gate = asyncio.Event()
        render = AsyncMock(return_value={
            "image_bytes": self.pixels,
            "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]})
        predicted = []

        async def on_profiles_ready(result):
            predicted.extend(result["characters"])

        with patch.object(self.pipeline, "_call_pipeline_llm",
                          AsyncMock(return_value=json.dumps(self.response))):
            task = asyncio.create_task(self.npc.generate_cards(
                manager=self.manager, bot_name="Story", candidates=self.candidates,
                prompts=self.prompts, past_text="",
                segmented="[C001] A porter hands over a key.",
                toggles={}, stream_notify=None, render_image=render,
                on_profiles_ready=on_profiles_ready, render_gate=gate))
            for _ in range(100):
                if predicted:
                    break
                await asyncio.sleep(0)
            self.assertEqual([profile["name"] for profile in predicted], ["Porter"])
            render.assert_not_awaited()
            self.assertEqual(
                [entry["name"] for entry in self.read()["bots"][0]["characters"]],
                ["Manual"])
            gate.set()
            result = await asyncio.wait_for(task, 5)
        self.assertEqual(result["created"], ["Porter"])
        render.assert_awaited_once()

    async def test_npc_node_runs_before_emotion_asset_with_same_card_tags(self):
        order = []
        captured = {}
        character_response = {
            "characters": [],
            "npc_candidates": self.candidates,
            "scene_plan": [{"anchor_segment": "C001", "character": "Porter", "emotion": "weary"}],
        }

        async def llm(name, messages, *args, **kwargs):
            order.append(name)
            if name == "CHARACTER-RESOLVE":
                return json.dumps(character_response)
            if name == "NPC 생성 노드":
                return json.dumps(self.response)
            if name == "EMOTION-RESOLVE":
                return json.dumps({"C001": {"character": "Porter", "outfit": "blue coat"}})
            if name == "EMOTION-ASSET":
                captured["asset"] = "\n".join(
                    str(message.get("content") or "") for message in messages)
                return json.dumps({"C001": {"emotion": "", "rating": "sfw"}})
            raise AssertionError(f"Unexpected stage: {name}")

        render = AsyncMock(return_value={
            "image_bytes": self.pixels,
            "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]})
        payload = {"session_id": "emotion-order",
                   "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                   "chats": [{"role": "char", "data": "A porter hands over a key."}]}
        with patch.object(self.pipeline, "_call_pipeline_llm", llm), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            _, result = await self.pipeline.resolve_profiles_before_generation(
                payload=payload,
                toggles={"npc_generation_enabled": True,
                         "npc_generation_deferred": True,
                         "illustration_enabled": True},
                history_plan=None, visual_profiles={},
                profile_translation_namespace="Story", workshop="emotion",
                npc_manager=self.manager, render_npc_image=render)
        self.assertLess(order.index("NPC 생성 노드"), order.index("EMOTION-RESOLVE"))
        self.assertLess(order.index("NPC 생성 노드"), order.index("EMOTION-ASSET"))
        # 예측한 NPC 카드 태그가 등록 캐릭터와 같은 입력 레벨로 EMOTION-ASSET에 들어간다.
        self.assertIn("1male", captured["asset"])
        self.assertIn("brown hair", captured["asset"])
        token = result["_npc_registration_token"]
        run = self.pipeline.take_emotion_npc_run(token)
        self.assertIsNotNone(run)
        render.assert_not_awaited()
        run["gate"].set()
        registered = await asyncio.wait_for(run["task"], 5)
        self.assertEqual(registered["created"], ["Porter"])
        self.assertTrue(self.root()["npc_generated"])

    async def test_deferred_emotion_npc_renders_before_asset_and_sets_reference(self):
        result = self.pipeline._empty_profile_result()
        result.update(
            npc_candidates=self.candidates,
            scene_plan=[{"anchor_segment": "C001", "character": "Porter", "emotion": "weary"}],
            emotion_assets=[{
                "anchor": "C001", "character": "Porter", "profile_id": "card_1",
                "gender": "", "appearance": [], "default_outfit": [],
                "outfit": "default outfit", "outfit_folder": "default outfit",
                "outfit_tags": ["blue coat"], "new_outfit_tags": False,
                "rating": "sfw", "emotion_id": "E1", "emotion_name": "weary",
                "emotion_tags": ["tired"], "emotion_folder": "weary",
                "target_dir": str(self.directory / "asset"), "reuse_path": "",
            }],
            current_characters=[{"name": "Porter", "confidence": 1.0}],
            characters=[{"name": "Porter", "in_history": False, "profile_timeline": []}],
        )
        render = AsyncMock(return_value={
            "image_bytes": self.pixels,
            "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]})
        with patch.object(self.pipeline, "_call_pipeline_llm",
                          AsyncMock(return_value=json.dumps(self.response))), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            output = await asyncio.wait_for(self.pipeline.build_from_context(
                {"session_id": "emotion-npc",
                 "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                 "chats": [{"role": "char", "data": "A porter hands over a key."}]},
                {"npc_generation_enabled": True, "npc_generation_deferred": True,
                 "illustration_output_mode": "illustration"},
                "", visual_profiles={}, profile_translation_namespace="Story",
                pre_resolved_profile_result=result, npc_manager=self.manager,
                render_npc_image=render, prepare_npc_character=AsyncMock(),
                workshop="emotion"), 10)
        self.assertEqual(output["npc_generation_result"]["created"], ["Porter"])
        asset = output["items"][0]["emotion_asset"]
        self.assertEqual(asset["profile_id"], "card_1")
        self.assertIn("brown hair", asset["appearance"])
        self.assertIn("broad shoulders", asset["appearance"])
        self.assertEqual(
            output["items"][0]["visual_base_snapshot"],
            {"Porter": {"visual_profile_id": "card_1"}})
        render.assert_awaited_once()

    async def test_emotion_npc_registers_even_without_deferred_render(self):
        """npc_generation_deferred가 꺼져 있어도 감정 에셋용 NPC는 생성됨으로 등록된다."""
        result = self.pipeline._empty_profile_result()
        result.update(
            npc_candidates=self.candidates,
            scene_plan=[{"anchor_segment": "C001", "character": "Porter", "emotion": "weary"}],
            emotion_assets=[{
                "anchor": "C001", "character": "Porter", "profile_id": "card_1",
                "gender": "", "appearance": [], "default_outfit": [],
                "outfit": "default outfit", "outfit_folder": "default outfit",
                "outfit_tags": ["blue coat"], "new_outfit_tags": False,
                "rating": "sfw", "emotion_id": "E1", "emotion_name": "weary",
                "emotion_tags": ["tired"], "emotion_folder": "weary",
                "target_dir": str(self.directory / "asset"), "reuse_path": "",
            }],
            current_characters=[{"name": "Porter", "confidence": 1.0}],
            characters=[{"name": "Porter", "in_history": False, "profile_timeline": []}],
            emotion_asset_updates=[{
                "character": "Porter", "profile_id": "card_1",
                "label": "default outfit", "tags": ["blue coat"],
            }],
        )
        render = AsyncMock(return_value={
            "image_bytes": self.pixels,
            "face_positions": [{"name": "Porter", "bbox": [.25, .25, .75, .75]}]})
        with patch.object(self.pipeline, "_call_pipeline_llm",
                          AsyncMock(return_value=json.dumps(self.response))), \
             patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
            output = await asyncio.wait_for(self.pipeline.build_from_context(
                {"session_id": "emotion-npc-immediate",
                 "target_slotted": "A porter hands over a key.\n\n[Slot 0]",
                 "chats": [{"role": "char", "data": "A porter hands over a key."}]},
                {"npc_generation_enabled": True, "npc_generation_deferred": False,
                 "illustration_output_mode": "illustration"},
                "", visual_profiles={}, profile_translation_namespace="Story",
                pre_resolved_profile_result=result, npc_manager=self.manager,
                render_npc_image=render, prepare_npc_character=AsyncMock(),
                workshop="emotion"), 10)
        self.assertEqual(output["npc_generation_result"]["created"], ["Porter"])
        root = self.root()
        self.assertTrue(root["npc_generated"])
        self.assertEqual(root["npc_idle_calls"], 0)
        self.assertEqual(root["gender_tag"], "1male")
        presets = {
            str(preset.get("label") or "").casefold(): [
                entry.get("tag") for entry in preset.get("tags") or []
            ]
            for preset in root["visual_cards"][0].get("outfit_presets") or []
        }
        self.assertEqual(presets.get("default outfit"), ["blue coat"])
        asset = output["items"][0]["emotion_asset"]
        self.assertEqual(asset["gender"], "1male")
        self.assertIn("brown hair", asset["appearance"])
        render.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
