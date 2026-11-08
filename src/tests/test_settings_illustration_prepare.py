"""Preparation ordering, progress, card persistence and safe cancellation."""
import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from settings_illustration_prepare import register_data_preparation


class PreparationProgressTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("preparation_bot_mode", modes / "bot_mode.py")
        cls.bot_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.bot_module)
            cls.profiles_module = sys.modules["modes.visual_profiles"]
        cls.modes_package = package

    async def asyncSetUp(self):
        def representative(character, card, filename):
            return {"character": character, "visual_card_id": card,
                    "visual_card_label": card, "visual_card_index": 1,
                    "filename": filename, "filepath": f"/{character}/{filename}"}

        self.images = [representative("Alice", "card_1", "shared.png"),
                       representative("Alice", "card_2", "shared.png"),
                       representative("Bob", "card_1", "bob.png"),
                       representative("Carol", "card_1", "carol.png")]
        # A newly added card sharing a filename must not expand the snapshot.
        self.current_images = [*self.images, representative("Alice", "card_new", "shared.png")]
        self.qm = SimpleNamespace(items=[])
        self.batch_calls = 0

        async def enqueue(specs):
            self.batch_calls += 1
            batch_id = f"batch-{self.batch_calls}"
            items = [SimpleNamespace(id=f"{batch_id}-{index}", batch_id=batch_id,
                                     label=spec["label"], params=spec["params"], status="pending",
                                     completion_future=asyncio.get_running_loop().create_future())
                     for index, spec in enumerate(specs)]
            self.qm.items.extend(items)
            return items

        async def cancel(run_id):
            for item in self.qm.items:
                if item.params.get("one_click_run_id") == run_id and item.status == "pending":
                    item.status = "cancelled"
                    item.completion_future.set_exception(RuntimeError("cancelled"))

        self.qm.add_items_batch = enqueue
        async def add_item(item_type, label, params, priority=0):
            item = (await enqueue([{"label": label, "params": params}]))[0]
            item.type = item_type
            return item

        self.qm.add_item = add_item
        self.qm.cancel_one_click_run = AsyncMock(side_effect=cancel)
        self.qm._get_bot_rep_paths = lambda bot, character: [
            image for image in self.current_images if not character or image["character"] == character]
        distribution = SimpleNamespace(request_lock=asyncio.Lock(), _runtime=lambda: {})
        modules = patch.dict(sys.modules, {
            "modes": self.modes_package,
            "modes.visual_profiles": self.profiles_module,
            "modes.bot_mode": self.bot_module,
            "queue_manager": SimpleNamespace(queue_manager=self.qm),
            "illustration_bot_distribution": SimpleNamespace(
                bot_distribution=distribution, bot_distributions={"illust": distribution, "emotion": distribution}),
        })
        modules.start()
        self.addCleanup(modules.stop)
        for name, value in (
            ("_load_patch_settings", Mock(return_value={})),
            ("_load_tag_filter_profiles", Mock(return_value={"profiles": {}})),
            ("get_bot_visual_targets", Mock(return_value=self.current_images)),
        ):
            mock = patch.object(self.bot_module, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.bot = self.bot_module.BotMode()
        self.bot.set_asset_tool(SimpleNamespace(use_builtin_tagger=True))
        self.bot.handle_get_rep_preview = AsyncMock(side_effect=lambda request: web.json_response({"images": self.images}))
        self.data_patcher = SimpleNamespace(handle_data_patch=AsyncMock(return_value=web.json_response({
            "visual_targets": self.images[:3],
        })))
        app = web.Application()
        register_data_preparation(SimpleNamespace(app=app, queue_manager=self.qm, bot_mode=self.bot, data_patcher=self.data_patcher))
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def run_state(self):
        response = await self.client.get("/api/bot_mode/data_preparation?bot=TestBot")
        return (await response.json())["run"]

    async def until(self, predicate):
        async with asyncio.timeout(3):
            while True:
                run = await self.run_state()
                if predicate(run):
                    return run
                await asyncio.sleep(0)

    async def start(self, characters=None, stages=None):
        response = await self.client.post("/api/bot_mode/data_preparation", json={
            "action": "start", "bot_name": "TestBot", "characters": characters or ["Alice", "Bob"],
            "stages": stages or ["tag_analysis"],
        })
        self.assertEqual(response.status, 202)
        return await self.until(lambda run: any(stage["total"] > 0 for stage in run["stages"]) or run["status"] != "running")

    def complete(self, item, error=None, result=None):
        item.status = "failed" if error else "completed"
        if error:
            item.completion_future.set_exception(RuntimeError(error))
        else:
            item.completion_future.set_result(result if result is not None else {"success": True})

    def prepare_cards(self):
        self.card_data = {"bots": [{"name": "TestBot", "characters": [
            {"name": name, "custom_field": "preserve", "visual_cards": [
                {"id": image["visual_card_id"], "label": image["visual_card_id"], "rep_images": [image["filename"]],
                 "appearance": [{"tag": "long hair", "desc": "머리 설명"}],
                 "default_outfit": [{"tag": "dress", "desc": "복장 설명"}], "gender_tag": "1girl",
                 "face_tags": "old face", "loras": [{"name": "keep.safetensors", "strength": 0.8}]}
                for image in self.images if image["character"] == name]}
            for name in ("Alice", "Bob", "Carol")]}]}
        self.card_data["bots"][0]["characters"][0]["visual_cards"].append({"id": "no_rep", "label": "대표 없음", "gender_tag": "1girl"})

        def save(data):
            self.card_data = deepcopy(data)

        for name, value in (
            ("_load_bot_data", Mock(side_effect=lambda: deepcopy(self.card_data))),
            ("_save_bot_data", Mock(side_effect=save)),
            ("_load_lb_extra", Mock(return_value=None)),
            ("_save_lb_extra", Mock()),
        ):
            mock = patch.object(self.bot_module, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def character(self, name):
        return next(character for character in self.card_data["bots"][0]["characters"] if character["name"] == name)

    def appearance_stage(self, run):
        return next(stage for stage in run["stages"] if stage["key"] == "appearance_refine")

    def refine(self, item):
        self.complete(item, result={"success": True, "data": {"appearance": ["long hair", "blue eyes"], "outfit": ["dress"]}})

    async def test_total_includes_later_characters_before_first_completion(self):
        run = await self.start()
        stage = run["stages"][0]
        self.assertEqual((stage["done"], stage["total"], stage["success"]), (0, 3, 0))
        self.assertIn("0/3", stage["detail"])
        self.assertEqual(self.batch_calls, 1)
        self.assertEqual([(item.params["image"]["character"], item.params["image"]["visual_card_id"])
                          for item in self.qm.items], [("Alice", "card_1"), ("Alice", "card_2"), ("Bob", "card_1")])
        self.complete(self.qm.items[0])
        run = await self.until(lambda value: value["stages"][0]["done"] == 1)
        self.assertEqual(run["stages"][0]["total"], 3)
        for item in self.qm.items[1:]:
            self.complete(item)
        run = await self.until(lambda value: value["status"] != "running")
        self.assertEqual(run["status"], "completed")
        self.assertEqual((run["stages"][0]["done"], run["stages"][0]["total"]), (3, 3))

    async def test_failed_image_is_counted_without_changing_total(self):
        await self.start()
        for index, item in enumerate(self.qm.items):
            self.complete(item, "analysis failed" if index == 0 else None)
        run = await self.until(lambda value: value["status"] != "running")
        stage = run["stages"][0]
        self.assertEqual((stage["done"], stage["total"], stage["success"]), (3, 3, 2))
        self.assertEqual(stage["status"], "warning")
        self.assertEqual(len(stage["issues"]), 1)

    async def test_stop_cancels_remaining_selection_and_keeps_full_total(self):
        run = await self.start()
        self.complete(self.qm.items[0])
        response = await self.client.post("/api/bot_mode/data_preparation", json={
            "action": "stop", "bot_name": "TestBot", "run_id": run["id"],
        })
        self.assertEqual(response.status, 200)
        run = await self.until(lambda value: value["status"] != "running")
        stage = run["stages"][0]
        self.assertEqual((stage["done"], stage["total"], stage["success"], stage["cancelled"]), (3, 3, 1, 2))
        self.assertEqual(stage["status"], "cancelled")

    async def test_missing_images_never_enqueue_other_characters(self):
        self.images[:] = [image for image in self.images if image["character"] != "Alice"]
        run = await self.start(["Alice"])
        stage = run["stages"][0]
        self.assertEqual(stage["status"], "warning")
        self.assertEqual(stage["total"], 0)
        self.assertEqual(self.batch_calls, 0)
        self.assertEqual(len(stage["issues"]), 1)

    async def test_appearance_stage_follows_filters_and_precedes_patch_and_faces(self):
        response = await self.client.get("/api/bot_mode/data_preparation?bot=TestBot")
        keys = [stage["key"] for stage in (await response.json())["stages"]]
        self.assertEqual(keys[:5], ["tag_analysis", "tag_filter", "appearance_refine", "negative", "data_patch"])
        self.assertEqual(keys[5:], ["face_tag_analysis", "face_negative", "llm_refine", "face_filter"])
        self.assertLess(keys.index("appearance_refine"), keys.index("llm_refine"))

    async def test_refinement_saves_all_selected_profiles_before_data_patch(self):
        self.prepare_cards()
        before = deepcopy(self.card_data)
        await self.start(stages=["data_patch", "appearance_refine", "tag_analysis"])
        self.data_patcher.handle_data_patch.assert_not_awaited()
        for item in list(self.qm.items):
            self.complete(item)
        run = await self.until(lambda value: self.appearance_stage(value)["total"] == 3)
        stage = self.appearance_stage(run)
        self.assertEqual((stage["done"], stage["total"]), (0, 3))
        self.data_patcher.handle_data_patch.assert_not_awaited()
        items = [item for item in self.qm.items if getattr(item, "type", "") == "bot_lb_extra_refine"]
        self.assertEqual([(item.params["char_name"], item.params["visual_card_id"]) for item in items],
                         [("Alice", "card_1"), ("Alice", "card_2"), ("Bob", "card_1")])
        for item in items:
            self.assertEqual(item.params["one_click_run_id"], run["id"])
            self.assertTrue(item.params["batch_refine"])
            self.assertEqual(item.params["appearance_tags"], ["long hair"])
            self.refine(item)
        run = await self.until(lambda value: next(s for s in value["stages"] if s["key"] == "data_patch")["total"] == 3)
        self.assertEqual(self.appearance_stage(run)["success"], 3)
        self.data_patcher.handle_data_patch.assert_awaited_once()
        for name in ("Alice", "Bob"):
            for card in self.character(name)["visual_cards"]:
                if card["id"] == "no_rep":
                    continue
                self.assertEqual(card["appearance"], [{"tag": "long hair", "desc": "머리 설명"}, {"tag": "blue eyes"}])
                self.assertEqual(card["default_outfit"], [{"tag": "dress", "desc": "복장 설명"}])
                self.assertEqual(card["face_tags"], "old face")
                self.assertEqual(card["loras"], [{"name": "keep.safetensors", "strength": 0.8}])
        self.assertEqual(self.character("Carol"), before["bots"][0]["characters"][2])
        for item in self.qm.items:
            if getattr(item, "type", "") == "data_patch_utility":
                self.complete(item)
        run = await self.until(lambda value: value["status"] != "running")
        self.assertEqual(run["status"], "completed")

    async def test_refinement_excludes_character_and_profile_locks(self):
        self.prepare_cards()
        self.character("Alice")["visual_cards"][1]["skip_lb_extra_batch_refine"] = True
        self.character("Bob")["lb_extra_batch_locked"] = True
        before = deepcopy(self.card_data)
        run = await self.start(stages=["appearance_refine"])
        stage = self.appearance_stage(run)
        self.assertEqual((stage["total"], stage["excluded"]), (1, 2))
        self.assertIn("잠금 제외 2개", stage["detail"])
        self.refine(self.qm.items[0])
        await self.until(lambda value: value["status"] != "running")
        self.assertEqual(self.character("Bob"), before["bots"][0]["characters"][1])
        self.assertEqual(self.character("Alice")["visual_cards"][1]["appearance"], [{"tag": "long hair", "desc": "머리 설명"}])

    async def test_all_locked_refinement_is_skipped_without_queueing(self):
        self.prepare_cards()
        for name in ("Alice", "Bob"):
            self.character(name)["lb_extra_batch_locked"] = True
        run = await self.start(stages=["appearance_refine"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(self.appearance_stage(run)["status"], "skipped")
        self.assertEqual(self.qm.items, [])

    async def test_invalid_response_and_queue_failure_preserve_cards_and_continue(self):
        self.prepare_cards()
        await self.start(stages=["appearance_refine"])
        self.complete(self.qm.items[0], result={"success": True, "data": {"appearance": [12], "outfit": []}})
        self.complete(self.qm.items[1], error="LLM unavailable")
        self.refine(self.qm.items[2])
        run = await self.until(lambda value: value["status"] != "running")
        stage = self.appearance_stage(run)
        self.assertEqual((stage["done"], stage["total"], stage["success"]), (3, 3, 1))
        self.assertEqual(stage["status"], "warning")
        self.assertEqual(len(stage["issues"]), 2)
        for card in self.character("Alice")["visual_cards"][:2]:
            self.assertEqual(card["appearance"], [{"tag": "long hair", "desc": "머리 설명"}])

    async def test_locks_are_rechecked_at_save_and_unrelated_edits_survive(self):
        self.prepare_cards()
        await self.start(stages=["appearance_refine"])
        self.character("Alice")["visual_cards"][1]["skip_lb_extra_batch_refine"] = True
        self.character("Bob")["lb_extra_batch_locked"] = True
        self.character("Alice")["visual_cards"][0]["face_tags"] = "concurrent face edit"
        for item in self.qm.items:
            self.refine(item)
        run = await self.until(lambda value: value["status"] != "running")
        stage = self.appearance_stage(run)
        self.assertEqual(stage["success"], 1)
        self.assertEqual(stage["status"], "warning")
        self.assertEqual(self.character("Alice")["visual_cards"][0]["face_tags"], "concurrent face edit")
        self.assertEqual(self.character("Bob")["visual_cards"][0]["appearance"], [{"tag": "long hair", "desc": "머리 설명"}])

    async def test_refinement_stop_finishes_running_card_and_cancels_pending_work(self):
        self.prepare_cards()
        run = await self.start(stages=["appearance_refine", "data_patch"])
        self.qm.items[0].status = "running"
        response = await self.client.post("/api/bot_mode/data_preparation", json={
            "action": "stop", "bot_name": "TestBot", "run_id": run["id"],
        })
        self.assertEqual(response.status, 200)
        self.refine(self.qm.items[0])
        run = await self.until(lambda value: value["status"] != "running")
        stage = self.appearance_stage(run)
        self.assertEqual((stage["done"], stage["total"], stage["success"], stage["cancelled"]), (3, 3, 1, 2))
        self.assertEqual(run["status"], "cancelled")
        self.data_patcher.handle_data_patch.assert_not_awaited()
        self.assertEqual(self.character("Alice")["visual_cards"][0]["appearance"][-1], {"tag": "blue eyes"})

    async def test_save_failure_is_reported_as_failure_not_refinement_success(self):
        self.prepare_cards()
        await self.start(stages=["appearance_refine"])
        with patch.object(self.bot_module, "_save_bot_data", side_effect=OSError("disk unavailable")):
            for item in self.qm.items:
                self.refine(item)
            run = await self.until(lambda value: value["status"] != "running")
        stage = self.appearance_stage(run)
        self.assertEqual((stage["done"], stage["success"]), (3, 0))
        self.assertEqual(stage["status"], "warning")
        self.assertTrue(all("disk unavailable" in issue for issue in stage["issues"]))

    async def retry(self, run, failure, expected=202):
        response = await self.client.post("/api/bot_mode/data_preparation", json={
            "action": "retry", "bot_name": "TestBot", "run_id": run["id"], "issue_ids": [failure["id"]],
        })
        self.assertEqual(response.status, expected, await response.text())
        return await response.json()

    async def failed_analysis(self):
        await self.start()
        for index, item in enumerate(self.qm.items):
            self.complete(item, "analysis failed" if index == 1 else None)
        return await self.until(lambda run: run["status"] != "running")

    async def test_failure_identifies_profile_image_and_retries_only_that_item(self):
        run = await self.failed_analysis()
        failure = run["stages"][0]["failures"][0]
        self.assertEqual((failure["character"], failure["visual_card_id"], failure["filename"]),
                         ("Alice", "card_2", "shared.png"))
        self.assertTrue(failure["retryable"])
        old_params = deepcopy(self.qm.items[1].params)
        result = await self.retry(run, failure)
        self.assertNotEqual(result["run"]["id"], run["id"])
        self.assertNotIn("_retries", result["run"])
        run = await self.until(lambda run: len(self.qm.items) == 4)
        retried = self.qm.items[-1]
        self.assertEqual(retried.params, {**old_params, "one_click_run_id": run["id"]})
        await self.retry(run, failure, expected=409)
        self.complete(retried)
        run = await self.until(lambda run: run["status"] != "running")
        stage = run["stages"][0]
        self.assertEqual((stage["done"], stage["total"], stage["success"]), (3, 3, 3))
        self.assertEqual(stage["issues"], [])
        self.assertEqual(stage["failures"][0]["status"], "resolved")
        self.assertEqual(stage["failures"][0]["attempts"], 1)
        self.assertEqual(run["status"], "completed")
        await self.retry(run, failure, expected=400)
        self.assertEqual(len(self.qm.items), 4)

    async def test_failed_retry_preserves_other_failures_and_can_retry_again(self):
        await self.start()
        self.complete(self.qm.items[0], "first failure")
        self.complete(self.qm.items[1], "second failure")
        self.complete(self.qm.items[2])
        run = await self.until(lambda run: run["status"] != "running")
        failure = run["stages"][0]["failures"][0]
        await self.retry(run, failure)
        await self.until(lambda run: len(self.qm.items) == 4)
        self.complete(self.qm.items[-1], "retry failed")
        run = await self.until(lambda run: run["status"] != "running")
        self.assertEqual(len(run["stages"][0]["issues"]), 2)
        self.assertEqual(run["stages"][0]["failures"][0]["message"], "retry failed")
        await self.retry(run, failure)
        await self.until(lambda run: len(self.qm.items) == 5)
        self.complete(self.qm.items[-1])
        run = await self.until(lambda run: run["status"] != "running")
        self.assertEqual(run["status"], "warning")
        self.assertEqual(run["stages"][0]["issues"], [run["stages"][0]["failures"][1]["issue"]])
        self.assertEqual(run["stages"][0]["success"], 2)

    async def test_llm_failure_exposes_exact_card_and_face_image(self):
        self.images[1]["visual_card_index"] = 2
        self.bot.handle_get_utility_preview = AsyncMock(return_value=web.json_response({"images": self.images}))
        with patch.dict(sys.modules, {"server": SimpleNamespace(queue_manager=self.qm)}):
            await self.start(stages=["llm_refine"])
        for index, item in enumerate(self.qm.items):
            self.complete(item, "얼굴/눈 태그 JSON 파싱 실패" if index == 1 else None)
        run = await self.until(lambda run: run["status"] != "running")
        stage = next(stage for stage in run["stages"] if stage["key"] == "llm_refine")
        failure = next(failure for failure in stage["failures"] if failure["retryable"])
        self.assertEqual((failure["character"], failure["visual_card_id"], failure["visual_card_index"], failure["filename"]),
                         ("Alice", "card_2", 2, "_face_image.webp"))
        await self.retry(run, failure)
        await self.until(lambda run: len(self.qm.items) == 4)
        self.assertEqual(self.qm.items[-1].type, "bot_llm_face_tag_analysis")
        self.assertEqual(self.qm.items[-1].params["visual_card_id"], "card_2")
        self.complete(self.qm.items[-1])
        run = await self.until(lambda run: run["status"] != "running")
        # The separate warning about a newly added card without FACE remains.
        self.assertEqual(run["status"], "warning")

    async def test_retry_refinement_still_saves_result_and_preserves_other_cards(self):
        self.prepare_cards()
        await self.start(stages=["appearance_refine"])
        self.complete(self.qm.items[0], "LLM failed")
        for item in self.qm.items[1:]:
            self.refine(item)
        run = await self.until(lambda run: run["status"] != "running")
        before = deepcopy(self.character("Bob"))
        failure = self.appearance_stage(run)["failures"][0]
        await self.retry(run, failure)
        await self.until(lambda run: len(self.qm.items) == 4)
        self.refine(self.qm.items[-1])
        run = await self.until(lambda run: run["status"] != "running")
        self.assertEqual(self.appearance_stage(run)["success"], 3)
        self.assertEqual(self.character("Alice")["visual_cards"][0]["appearance"][-1], {"tag": "blue eyes"})
        self.assertEqual(self.character("Bob"), before)

    async def test_unknown_or_stale_retry_does_not_enqueue_work(self):
        run = await self.failed_analysis()
        await self.retry(run, {"id": "unknown"}, expected=400)
        await self.retry({"id": "old-run"}, run["stages"][0]["failures"][0], expected=400)
        self.assertEqual(len(self.qm.items), 3)

    async def test_cancelled_retry_remains_available_for_retry(self):
        run = await self.failed_analysis()
        failure = run["stages"][0]["failures"][0]
        await self.retry(run, failure)
        run = await self.until(lambda run: len(self.qm.items) == 4)
        response = await self.client.post("/api/bot_mode/data_preparation", json={
            "action": "stop", "bot_name": "TestBot", "run_id": run["id"],
        })
        self.assertEqual(response.status, 200)
        run = await self.until(lambda run: run["status"] != "running")
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["stages"][0]["failures"][0]["status"], "failed")
        await self.retry(run, failure)
        await self.until(lambda run: len(self.qm.items) == 5)
        self.complete(self.qm.items[-1])
        run = await self.until(lambda run: run["status"] != "running")
        self.assertEqual(run["status"], "completed")


if __name__ == "__main__":
    unittest.main()
