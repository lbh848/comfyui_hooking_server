"""Maker proposals never apply implicitly; restores preserve both history and settings."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from asset_character_maker import AssetCharacterMaker, generation_job, register_character_maker_runtime
from asset_character_presets import AssetCharacterPresets
from json_store import write_json_atomic
from settings_assets import allowed_asset_path


def settings_fixture():
    return {"defaults": {"raw_appearance_tags": ["default"], "appearance": "appearance"}, "values": {
        "appearance": "appearance", "outfit": "outfit", "expression": "expression",
        "raw_appearance_tags": ["black hair"], "raw_outfit_tags": ["white shirt"],
        "raw_expression_tags": ["smile"], "raw_composition_tags": ["portrait"],
        "raw_character_negative_tags": ["bad hands"], "natural_language": "Before.",
        "lora_enabled": True, "lora_list": [{"name": "character.safetensors", "strength": .7}],
        "snapref_images": [{"name": "reference.png", "local_path": "reference.png"}],
    }}


def proposal_fixture():
    return {"fields": {"appearance": ["blue hair"], "outfit": ["coat"], "expression": ["smile"], "composition": ["portrait"]},
            "natural_language": "After.", "assistant_message": "Changed hair and outfit."}


class CharacterMakerTests(unittest.TestCase):
    def test_proposal_merges_composition_into_expression_and_keeps_background(self):
        self.settings["values"]["raw_background_tags"] = ["garden"]
        self.save(self.settings)
        pending, before, _ = self.store.begin(self.body("propose", message="Change expression"))
        self.store.finish("sample", pending["pending"]["id"], before, proposal_fixture())
        proposal = self.store.read("sample")["versions"][-1]["after"]["values"]
        self.assertEqual(proposal["raw_expression_tags"], ["smile", "portrait"])
        self.assertNotIn("raw_composition_tags", proposal)
        self.assertEqual(proposal["raw_background_tags"], ["garden"])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.characters = AssetCharacterPresets(self.root / "tags.json", self.root / "builtin.json")
        self.characters.change({"action": "create", "name": "sample"})
        self.settings = settings_fixture()
        self.save(self.settings)
        self.store = AssetCharacterMaker(self.characters, self.root / "versions")

    def save(self, settings):
        self.characters.change({"action": "save", "name": "sample", "fields": {"settings": settings}})

    def body(self, action, **extra):
        current = self.store.read("sample")
        return {"name": "sample", "action": action, "settings": self.characters.catalog()["characters"]["sample"]["settings"],
                "revision": current["revision"], "character_revision": current["character_revision"], **extra}

    def propose(self, **extra):
        state, before, history = self.store.begin(self.body("propose", message="Change the outfit", **extra))
        self.store.finish("sample", state["pending"]["id"], before, proposal_fixture())
        return self.store.read("sample")["versions"][-1], history

    def test_proposal_is_separate_and_survives_restart(self):
        version, _ = self.propose()
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], self.settings)
        self.assertEqual(version["before"], self.settings)
        self.assertEqual(version["after"]["values"]["raw_appearance_tags"], ["blue hair"])
        for key in ("lora_list", "snapref_images", "raw_character_negative_tags"):
            self.assertEqual(version["after"]["values"][key], self.settings["values"][key])
        restarted = AssetCharacterMaker(self.characters, self.root / "versions")
        self.assertEqual(restarted.read("sample")["versions"][-1], version)

    def test_apply_rollback_and_roll_forward_preserve_all_versions(self):
        version, _ = self.propose()
        first = self.store.read("sample")["versions"][0]
        self.store.change(self.body("apply", version_id=version["id"]))
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], version["after"])
        restored = self.store.change(self.body("rollback", version_id=first["id"]))
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], self.settings)
        self.assertEqual([v["kind"] for v in restored["versions"]], ["snapshot", "proposal", "apply", "rollback"])
        forward = self.store.change(self.body("rollback", version_id=version["id"]))
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], version["after"])
        self.assertEqual(len(forward["versions"]), 5)

    def test_manual_changes_during_ai_are_preserved_before_explicit_apply(self):
        state, before, _ = self.store.begin(self.body("propose", message="change"))
        edited = copy.deepcopy(self.settings)
        edited["values"]["lora_list"] = []
        self.save(edited)
        self.store.finish("sample", state["pending"]["id"], before, proposal_fixture())
        result = self.store.read("sample")
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], edited)
        applied = self.store.change(self.body("apply", version_id=result["versions"][-1]["id"]))
        self.assertEqual(applied["versions"][-2]["after"], edited)
        self.store.change(self.body("rollback", version_id=applied["versions"][-2]["id"]))
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], edited)

    def test_stale_revision_and_local_draft_are_rejected(self):
        stale = self.body("snapshot")
        edited = copy.deepcopy(self.settings)
        edited["values"]["natural_language"] = "Another tab"
        self.save(edited)
        with self.assertRaisesRegex(ValueError, "변경"):
            self.store.change(stale)
        with self.assertRaisesRegex(ValueError, "다릅니다"):
            self.store.change(self.body("snapshot", settings=self.settings))

    def test_legacy_character_background_does_not_block_current_maker_drafts(self):
        legacy = copy.deepcopy(self.settings)
        for part in legacy.values():
            part.update(background_preset="garden", raw_background_tags=["garden"])
        self.save(legacy)
        result = self.store.change(self.body("snapshot", settings=self.settings))
        self.assertEqual(result["versions"][-1]["after"], self.settings)
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], legacy)

    def test_chat_context_follows_selected_branch_only(self):
        first, _ = self.propose()
        second, history = self.propose(base_id=first["id"])
        self.assertEqual(len(history), 2)
        self.assertEqual(second["before"], first["after"])
        _, history = self.propose()  # Current applied value excludes unaccepted proposals.
        self.assertEqual(history, [])
        self.store.change(self.body("apply", version_id=second["id"]))
        _, history = self.propose()
        self.assertEqual(len(history), 4)

    def test_reset_clears_conversation_and_versions_but_keeps_settings_and_other_characters(self):
        version, _ = self.propose()
        self.store.change(self.body("apply", version_id=version["id"]))
        pending, before, _ = self.store.begin(self.body("propose", message="failed request"))
        self.store.finish("sample", pending["pending"]["id"], before, error="Failed")
        self.characters.change({"action": "create", "name": "other"})
        self.characters.change({"action": "save", "name": "other", "fields": {"settings": self.settings}})
        other = self.store.read("other")
        self.store.change({**other, "action": "snapshot", "settings": self.settings})
        other = self.store.read("other")
        catalog = self.characters.catalog()

        reset = self.store.change(self.body("reset"))
        self.assertEqual(reset["versions"], [])
        self.assertEqual(reset["active_version_id"], "")
        self.assertEqual(reset["error"], "")
        self.assertIsNone(reset["pending"])
        self.assertEqual(self.characters.catalog(), catalog)
        self.assertEqual(self.store.read("other"), other)
        self.assertEqual(AssetCharacterMaker(self.characters, self.root / "versions").read("sample"), reset)
        fresh, history = self.propose()
        self.assertEqual(history, [])
        self.assertEqual(fresh["before"], version["after"])
        self.assertEqual(fresh["number"], 2)

    def test_reset_rejects_stale_revision_and_running_requests(self):
        stale = self.body("reset")
        self.propose()
        with self.assertRaisesRegex(ValueError, "변경"):
            self.store.change(stale)
        pending, before, _ = self.store.begin(self.body("propose", message="running"))
        for generating in (False, True):
            if generating:
                self.store.finish("sample", pending["pending"]["id"], before, proposal_fixture(), generate=True)
            current = self.store.read("sample")
            with self.assertRaisesRegex(ValueError, "기다린"):
                self.store.change(self.body("reset"))
            self.assertEqual(self.store.read("sample"), current)

    def test_failed_reset_write_keeps_history_and_settings(self):
        self.propose()
        before = self.store.read("sample")
        catalog = self.characters.catalog()
        with patch("asset_character_maker.write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.change(self.body("reset"))
        self.assertEqual(self.store.read("sample"), before)
        self.assertEqual(self.characters.catalog(), catalog)

    def test_failed_or_cancelled_ai_can_be_retried_and_restart_recovers(self):
        state, before, _ = self.store.begin(self.body("propose", message="change"))
        with self.assertRaises(ValueError):
            self.store.begin(self.body("propose", message="duplicate"))
        self.store.finish("sample", state["pending"]["id"], before, error="Cancelled")
        self.assertEqual(len(self.store.read("sample")["versions"]), 1)
        self.store.begin(self.body("propose", message="retry"))
        restarted = AssetCharacterMaker(self.characters, self.root / "versions")
        result = restarted.read("sample")
        self.assertIsNone(result["pending"])
        self.assertIn("재시작", result["error"])

    def test_failed_write_keeps_history_and_applied_values(self):
        version, _ = self.propose()
        original = self.store.read("sample")
        def failing(path, value):
            if path == self.characters.path:
                raise OSError("disk full")
            write_json_atomic(path, value)
        from asset_preset_scope import write_scope_documents
        with patch("asset_character_maker.write_scope_documents", side_effect=lambda docs: write_scope_documents(docs, writer=failing)):
            with self.assertRaises(OSError):
                self.store.change(self.body("apply", version_id=version["id"]))
        self.assertEqual(self.store.read("sample"), original)
        self.assertEqual(self.characters.catalog()["characters"]["sample"]["settings"], self.settings)

    def test_builtin_is_readonly_and_character_histories_are_isolated(self):
        version, _ = self.propose()
        token = self.characters.change({"action": "unlock", "confirmed": True})["builtin_unlock_token"]
        self.characters.change({"action": "scope", "name": "sample", "scope": "builtin", "revision": self.characters.catalog()["revision"], "builtin_unlock_token": token})
        with self.assertRaisesRegex(ValueError, "읽기 전용"):
            self.store.change(self.body("apply", version_id=version["id"]))
        self.store.change(self.body("apply", version_id=version["id"], builtin_unlock_token=token))
        with self.assertRaises(ValueError):
            self.characters.change({"action": "create", "name": "../../other"})
        self.characters.change({"action": "create", "name": "other"})
        self.assertEqual(self.store.read("other")["versions"], [])
        self.assertEqual(self.store.path("../../other").parent, self.root / "versions")

    def test_images_follow_selected_version_and_survive_apply_and_restart(self):
        from PIL import Image
        image = self.root / "result.png"
        Image.new("RGB", (8, 8)).save(image)
        pending, before, _ = self.store.begin(self.body("propose", message="draw"))
        proposal = self.store.finish("sample", pending["pending"]["id"], before, proposal_fixture(), generate=True)
        rendering = self.store.read("sample")
        self.assertEqual(rendering["pending"]["phase"], "generate")
        self.assertEqual(rendering["pending"]["version_id"], proposal["id"])
        self.store.finish_image("sample", pending["pending"]["id"], {"success": True, "local_path": str(image)})
        version = self.store.read("sample")["versions"][-1]
        self.assertIn("image_url", version)
        self.assertNotIn("image_path", version)
        self.assertEqual(self.store.image_path("sample", version["id"]), str(image))
        restarted = AssetCharacterMaker(self.characters, self.root / "versions")
        self.assertEqual(restarted.read("sample")["versions"][-1], version)
        feedback, _, history = self.store.begin(self.body("propose", message="fix image", base_id=version["id"]))
        self.assertEqual(feedback["pending"]["image_version_id"], version["id"])
        self.assertEqual(len(history), 2)
        self.store.finish("sample", feedback["pending"]["id"], before, error="cancelled")
        unrelated, _, _ = self.store.begin(self.body("propose", message="current values"))
        self.assertNotIn("image_version_id", unrelated["pending"])
        self.store.finish("sample", unrelated["pending"]["id"], before, error="cancelled")
        applied = self.store.change(self.body("apply", version_id=version["id"]))
        current, _, _ = self.store.begin(self.body("propose", message="applied image"))
        self.assertEqual(current["pending"]["image_version_id"], applied["active_version_id"])
        self.store.finish("sample", current["pending"]["id"], before, error="cancelled")
        image.unlink()
        with self.assertRaisesRegex(ValueError, "기준 이미지"):
            self.store.begin(self.body("propose", message="missing image", base_id=version["id"]))
        text_only, _, _ = self.store.begin(self.body("propose", message="text only", base_id=version["id"], use_image=False))
        self.assertNotIn("image_version_id", text_only["pending"])

    def test_image_failure_keeps_proposal_and_retry_uses_the_same_version(self):
        pending, before, _ = self.store.begin(self.body("propose", message="draw"))
        version = self.store.finish("sample", pending["pending"]["id"], before, proposal_fixture(), generate=True)
        self.store.finish("sample", pending["pending"]["id"], before, error="GPU failed")
        result = self.store.read("sample")
        self.assertIsNone(result["pending"])
        self.assertEqual(result["versions"][-1]["image_error"], "GPU failed")
        retry, values, _ = self.store.begin(self.body("generate", base_id=version["id"]))
        self.assertEqual(retry["pending"]["version_id"], version["id"])
        self.assertEqual(values, version["after"])
        self.assertEqual(len(retry["versions"]), 2)
        restarted = AssetCharacterMaker(self.characters, self.root / "versions").read("sample")
        self.assertIsNone(restarted["pending"])
        self.assertIn("재시작", restarted["error"])
        self.assertEqual(restarted["versions"][-1]["id"], version["id"])

    def test_generation_uses_proposal_text_and_preserves_style_and_controls(self):
        template = {"prompt_json": {"fields": {"ANIMA_CONTENT": "artist, garden", "SDXL_CONTENT": "quality, artist, garden",
                    "SNAPREF": {"enabled": True, "references": ["ref"]}, "LORA_DATA": {"list": ["character"]}, "SEED": 42}}}
        original = copy.deepcopy(template)
        job = generation_job(template, self.settings["values"])
        self.assertEqual(template, original)
        fields = json.loads(job["positive_prompt"])["fields"]
        self.assertEqual(fields["ANIMA_CONTENT"], "artist, garden, black hair, smile, portrait, white shirt, Before.")
        self.assertNotIn("Before.", fields["SDXL_CONTENT"])
        for key in ("SNAPREF", "LORA_DATA", "SEED"):
            self.assertEqual(fields[key], template["prompt_json"]["fields"][key])


class CharacterMakerAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_adapter_sends_image_to_vision_with_conversation(self):
        from modes.character_maker_mode import CharacterMakerService
        editor = CharacterMakerService.__new__(CharacterMakerService)
        editor._encode_vision_image = lambda path: ("image-bytes", "image/png")
        editor._call_revision_llm = AsyncMock(return_value=proposal_fixture())
        await editor.suggest_asset_values("sample", settings_fixture()["values"], "fix eyes",
                                          [{"role": "user", "content": "previous request", "accepted": True}], image_path="result.png")
        task, messages, images = editor._call_revision_llm.await_args.args
        self.assertEqual(task, "character_maker_feedback")
        self.assertEqual(images, [("image-bytes", "image/png")])
        self.assertIn("CURRENT", messages[1]["content"])
        self.assertIn("previous request", messages[1]["content"])

    async def test_ai_adapter_keeps_legacy_session_untouched(self):
        from modes.character_maker_mode import CharacterMakerService
        editor = CharacterMakerService.__new__(CharacterMakerService)
        editor.sessions = {"default": {"fields": {"appearance": ["legacy"]}}}
        before = copy.deepcopy(editor.sessions)
        editor._call_revision_llm = AsyncMock(return_value=proposal_fixture())
        result = await editor.suggest_asset_values("sample", settings_fixture()["values"], "change", [])
        self.assertEqual(result, proposal_fixture())
        self.assertEqual(editor.sessions, before)
        task, messages, images = editor._call_revision_llm.await_args.args
        self.assertEqual(task, "character_maker_draft")
        self.assertIn("black hair", messages[1]["content"])
        self.assertIsNone(images)

    async def test_queue_routes_maker_callback_and_preserves_legacy_route(self):
        from queue_manager import QueueManager
        queue = QueueManager.__new__(QueueManager)
        queue.character_maker = SimpleNamespace(revise=AsyncMock(return_value={"legacy": True}))
        queue._notify_progress = AsyncMock()
        callback = AsyncMock(return_value={"maker": True})
        item = SimpleNamespace(_runtime_handler=callback, params={"character": "sample"})
        self.assertEqual(await queue._handle_character_maker(item), {"maker": True})
        queue.character_maker.revise.assert_not_awaited()
        self.assertEqual(await queue._handle_character_maker(SimpleNamespace(params={"session_id": "default", "payload": {"message": "change"}})), {"legacy": True})
        queue.character_maker.revise.assert_awaited_once_with("default", {"message": "change"})


class CharacterMakerApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_generates_image_and_next_feedback_receives_it(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            characters = AssetCharacterPresets(root / "tags.json", root / "builtin.json")
            characters.change({"action": "create", "name": "sample"})
            settings = settings_fixture()
            characters.change({"action": "save", "name": "sample", "fields": {"settings": settings}})
            calls = []
            async def add_item(*args, **kwargs):
                item = SimpleNamespace(completion_future=asyncio.get_running_loop().create_future())
                calls.append((args, kwargs, item))
                return item
            app = web.Application()
            editor = SimpleNamespace(suggest_asset_values=AsyncMock(return_value=proposal_fixture()))
            server = SimpleNamespace(app=app, CONFIG_FILE=root / "settings.json", queue_manager=SimpleNamespace(add_item=add_item),
                                     asset_mode=SimpleNamespace(character_presets=characters, load_tags=lambda: None), character_maker=editor)
            register_character_maker_runtime(server, lambda job: {"model": "captured"})
            async with TestClient(TestServer(app)) as client:
                async def get():
                    return await (await client.get("/api/asset/character-maker?name=sample")).json()
                async def post(**extra):
                    current = await get()
                    response = await client.post("/api/asset/character-maker", json={"action": "propose", "name": "sample", "settings": settings,
                        "revision": current["revision"], "character_revision": current["character_revision"], "message": "draw", **extra})
                    self.assertEqual(response.status, 200, await response.text())
                    return await response.json()
                template = {"prompt_json": {"fields": {"ANIMA_CONTENT": "style"}}}
                await post(generation_job=template)
                args, kwargs, item = calls[0]
                self.assertEqual(args, ("character_maker",))
                item.completion_future.set_result(await kwargs["runtime_handler"](item))
                generating = await get()
                self.assertEqual(generating["pending"]["phase"], "generate")
                args, kwargs, image_item = calls[1]
                self.assertEqual(args, ("asset_generation",))
                self.assertEqual(kwargs["params"]["asset_model_preset"], {"model": "captured"})
                self.assertRegex(kwargs["params"]["body"]["storage_session"], r"^asset-[0-9a-f]{64}$")
                self.assertIn("blue hair", kwargs["params"]["body"]["positive_prompt"])
                self.assertNotIn("black hair", kwargs["params"]["body"]["positive_prompt"])
                image = root / "generated.png"
                Image.new("RGB", (8, 8)).save(image)
                image_item.completion_future.set_result({"success": True, "local_path": str(image)})
                await asyncio.sleep(0)
                result = await get()
                self.assertIsNone(result["pending"])
                version = result["versions"][-1]
                response = await client.get(version["image_url"].replace("/api/asset/asset/", "/api/asset/", 1))
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.read(), image.read_bytes())
                self.assertNotIn("image_path", version)
                self.assertEqual(characters.catalog()["characters"]["sample"]["settings"], settings)
                await post(base_id=version["id"])
                _, kwargs, item = calls[-1]
                item.completion_future.set_result(await kwargs["runtime_handler"](item))
                self.assertEqual(editor.suggest_asset_values.await_args.kwargs["image_path"], str(image))
                self.assertEqual(len(editor.suggest_asset_values.await_args.args[3]), 2)
                await post(action="generate", base_id=version["id"], generation_job=template)
                calls[-1][2].completion_future.cancel()
                await asyncio.sleep(0)
                failed = await get()
                self.assertIsNone(failed["pending"])
                self.assertIn("취소", failed["error"])
                self.assertEqual(failed["versions"][1]["image_url"], version["image_url"])
            self.assertTrue(allowed_asset_path("/api/asset/character-maker/image", "GET"))
            self.assertFalse(allowed_asset_path("/api/asset/character-maker/image", "POST"))

    async def test_queue_submission_polling_and_apply(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            characters = AssetCharacterPresets(root / "tags.json", root / "builtin.json")
            characters.change({"action": "create", "name": "sample"})
            settings = settings_fixture()
            characters.change({"action": "save", "name": "sample", "fields": {"settings": settings}})
            calls = []
            async def add_item(*args, **kwargs):
                item = SimpleNamespace(completion_future=asyncio.get_running_loop().create_future())
                calls.append((args, kwargs, item))
                return item
            app = web.Application()
            server = SimpleNamespace(app=app, CONFIG_FILE=root / "settings.json", queue_manager=SimpleNamespace(add_item=add_item),
                                     asset_mode=SimpleNamespace(character_presets=characters, load_tags=lambda: None),
                                     character_maker=SimpleNamespace(suggest_asset_values=AsyncMock(return_value=proposal_fixture())))
            register_character_maker_runtime(server)
            async with TestClient(TestServer(app)) as client:
                async def get():
                    return await (await client.get("/api/asset/character-maker?name=sample")).json()
                initial = await get()
                response = await client.post("/api/asset/character-maker", json={"action": "propose", "name": "sample", "settings": settings,
                    "revision": initial["revision"], "character_revision": initial["character_revision"], "message": "change"})
                self.assertEqual(response.status, 200)
                self.assertIsNotNone((await response.json())["pending"])
                args, kwargs, item = calls[0]
                self.assertEqual(args, ("character_maker",))
                item.completion_future.set_result(await kwargs["runtime_handler"](item))
                result = await get()
                self.assertIsNone(result["pending"])
                self.assertEqual(result["versions"][-1]["kind"], "proposal")
                self.assertEqual(characters.catalog()["characters"]["sample"]["settings"], settings)
                response = await client.post("/api/asset/character-maker", json={"action": "apply", "name": "sample", "settings": settings,
                    "revision": result["revision"], "character_revision": result["character_revision"], "version_id": result["versions"][-1]["id"]})
                self.assertEqual(response.status, 200)
                self.assertEqual(characters.catalog()["characters"]["sample"]["settings"]["values"]["raw_outfit_tags"], ["coat"])
                applied_settings = characters.catalog()["characters"]["sample"]["settings"]
                current = await get()
                response = await client.post("/api/asset/character-maker", json={"action": "reset", "name": "sample", "settings": applied_settings,
                    "revision": current["revision"], "character_revision": current["character_revision"]})
                self.assertEqual(response.status, 200, await response.text())
                self.assertEqual((await response.json())["versions"], [])
                self.assertEqual((await get())["versions"], [])
                self.assertEqual(characters.catalog()["characters"]["sample"]["settings"], applied_settings)
            self.assertTrue(allowed_asset_path("/api/asset/character-maker", "GET"))
            self.assertTrue(allowed_asset_path("/api/asset/character-maker", "POST"))
            self.assertFalse(allowed_asset_path("/api/asset/character-maker", "DELETE"))


if __name__ == "__main__":
    unittest.main()
