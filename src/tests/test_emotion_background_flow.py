"""Background rendering must report its actual outcome without reserving an LLM call."""
import ast
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import illustration_flow as flow


def background_callback(store):
    # Load the real server callback without starting the server or its image workers.
    path = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_background_cards_ready")
    wrapper = ast.parse("""
def bind(background_context, emotion_background_cards, illustration_flow, asyncio):
    background_render_task = None
    pass
    return on_background_cards_ready, lambda: background_render_task
""")
    wrapper.body[0].body[1] = deepcopy(callback)
    namespace = {"render_background_image": AsyncMock()}
    exec(compile(ast.fix_missing_locations(wrapper), str(path), "exec"), namespace)
    return namespace["bind"]({"bot": "b" * 24, "world": "castle"}, store, flow, asyncio)


class BackgroundFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.run = {"id": "test", "kind": "illustration", "nodes": {}, "revision": 0}
        for variable, value in ((flow._run, self.run), (flow._frontier, ("selection",)),
                                (flow._reserved, None), (flow._branches, None)):
            token = variable.set(value)
            self.addCleanup(variable.reset, token)
        change = patch.object(flow, "changed")
        change.start()
        self.addCleanup(change.stop)
        self.card = {"id": "a" * 24, "name": "Library", "positive": "empty library"}

    def only_node(self):
        self.assertEqual(len(self.run["nodes"]), 1)
        return next(iter(self.run["nodes"].values()))

    async def test_shared_background_image_keeps_each_scenes_point_and_crop(self):
        store = SimpleNamespace(ensure=AsyncMock(return_value={**self.card, "image_path": "library.webp"}))
        callback, get_task = background_callback(store)
        anchors = {"C001": {**self.card, "view_center": {"x": .2, "y": .5}, "character_position": "R"},
                   "C002": {**self.card, "view_center": {"x": .8, "y": .5}, "character_position": "L"}}
        await callback(anchors)
        ready = await flow.join(get_task())
        self.assertEqual(store.ensure.await_count, 1)
        self.assertNotIn("character_position", store.ensure.call_args.args[1])
        self.assertNotIn("view_center", store.ensure.call_args.args[1])
        for anchor, card in anchors.items():
            self.assertEqual(ready[anchor]["character_position"], card["character_position"])
            self.assertEqual(ready[anchor]["view_center"], card["view_center"])

    async def test_successful_render_reports_processing_then_completed_and_merges_anchors(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def ensure(*args):
            started.set()
            await finish.wait()
            return {**self.card, "image_path": "library.webp"}

        store = SimpleNamespace(ensure=AsyncMock(side_effect=ensure))
        callback, get_task = background_callback(store)
        await callback({"C001": self.card, "C002": self.card})
        task = get_task()
        self.addAsyncCleanup(self.finish_task, task)
        await asyncio.wait_for(started.wait(), timeout=2)
        node = self.only_node()
        self.assertEqual((node["kind"], node["executor"], node["status"]), ("stage", "process", "processing"))
        self.assertEqual(node["dependencies"], ["selection"])
        finish.set()
        result = await flow.join(task)
        self.assertEqual(result["C001"], result["C002"])
        self.assertEqual(store.ensure.await_count, 1)
        self.assertEqual(node["status"], "completed")
        self.assertEqual(node["error"], "")
        self.assertIsNotNone(node["started_at"])
        self.assertEqual(flow._frontier.get(), (node["id"],))

    async def test_cached_result_and_empty_scene_complete_without_an_llm_call(self):
        for anchors in ({"C001": self.card}, {}):
            with self.subTest(anchors=anchors):
                self.run["nodes"].clear()
                store = SimpleNamespace(ensure=AsyncMock(return_value=self.card))
                callback, get_task = background_callback(store)
                await callback(anchors)
                await flow.join(get_task())
                self.assertEqual(self.only_node()["status"], "completed")

    async def test_real_render_error_reports_failed(self):
        callback, get_task = background_callback(SimpleNamespace(ensure=AsyncMock(side_effect=RuntimeError("render failed"))))
        await callback({"C001": self.card})
        with self.assertRaises(ExceptionGroup):
            await flow.join(get_task())
        node = self.only_node()
        self.assertEqual(node["status"], "failed")
        self.assertTrue(node["error"])

    async def test_real_cancellation_reports_cancelled(self):
        started = asyncio.Event()

        async def ensure(*args):
            started.set()
            await asyncio.Event().wait()

        callback, get_task = background_callback(SimpleNamespace(ensure=AsyncMock(side_effect=ensure)))
        await callback({"C001": self.card})
        task = get_task()
        self.addAsyncCleanup(self.finish_task, task)
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await flow.join(task)
        self.assertEqual(self.only_node()["status"], "cancelled")

    async def finish_task(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class BackgroundPredictionTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_background_prompt_does_not_replace_scene_specific_placement(self):
        from emotion_background_cards import selected_background_cards
        from emotion_composition import normalize_composition_settings
        path = Path(__file__).resolve().parents[1] / "modes" / "illustration_context_pipeline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_background_card_prediction")
        scenes = [{"anchor_segment": anchor, "background": {"location": "Library", "description": "stone walls",
                   "view_center": {"x": x, "y": .5}, "character_position": code}}
                  for anchor, x, code in (("C001", .2, "R"), ("C002", .8, "L"))]
        context = {"composition_settings": normalize_composition_settings()}
        requests = selected_background_cards(scenes, context)[1]
        call = AsyncMock(return_value=json.dumps({"cards": [{"id": requests[0]["id"], "positive": "empty library"}]}))
        namespace = {"_normalize_messages": lambda messages: messages,
                     "prompt_builders": SimpleNamespace(background_card_messages=lambda *args: []),
                     "_call_pipeline_llm": call}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        _, anchors = await namespace["_run_background_card_prediction"](
            background_system="", scene_plan=scenes, segmented_current="", background_context=context)
        for scene in scenes:
            card = anchors[scene["anchor_segment"]]
            self.assertEqual(card["positive"], "empty library")
            self.assertEqual(card["character_position"], scene["background"]["character_position"])
            self.assertEqual(card["view_center"], scene["background"]["view_center"])
