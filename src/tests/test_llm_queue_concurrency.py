"""Queued vision work uses provider settings even with empty legacy slot values."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from app_paths import SOURCE_DIR
from llm_settings import RuntimeSettingsReader, migrate_document
from settings_store import SettingsStore
from test_llm_settings import document, load_engine


class LlmQueueConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = load_engine()
        package = types.ModuleType("modes")
        package.__path__ = [str(SOURCE_DIR / "modes")]
        package.llm_service = self.engine
        spec = importlib.util.spec_from_file_location("queue_concurrency_test", SOURCE_DIR / "queue_manager.py")
        self.queue_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package, spec.name: self.queue_module}):
            spec.loader.exec_module(self.queue_module)
        self.manager = self.queue_module.QueueManager()
        self.manager.get_config = lambda: {"llm_model": "", "llm_model2": "", "llm_max_concurrency2": 1}
        self.store = SettingsStore(self.root)
        self.value = migrate_document(document(), self.root / "config")
        for index in (1, 2):
            self.value["draft"]["llms"][index]["model"] = f"vision-{index}"
        self.value["draft"]["llmProviders"][0]["maxConcurrency"] = 3
        self.revision = self.store.save(self.value, '"missing"')
        patches = [patch.object(self.engine, "_settings_reader", RuntimeSettingsReader(self.root / "config"))]
        patches += [patch.object(self.manager, name, AsyncMock()) for name in ("_notify_queue_updated", "_deferred_prune", "_process_loop")]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addAsyncCleanup(self.stop_workers)

    async def stop_workers(self):
        workers = list(self.manager._llm_worker_tasks.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def test_refine_queue_runs_three_requests_but_shares_provider_cap_across_slots(self):
        started = []
        ready, release = asyncio.Event(), asyncio.Event()
        active, peak = 0, 0

        async def refine(item):
            nonlocal active, peak
            async with self.engine._limit_llm_request(item.params["slot"]):
                started.append(item.id)
                active += 1
                peak = max(peak, active)
                if active == 3:
                    ready.set()
                try:
                    await release.wait()
                finally:
                    active -= 1
                return {"success": True}

        self.manager._handle_bot_lb_extra_refine = refine
        items = [self.queue_module.QueueItem(id=str(i), type="bot_lb_extra_refine", label="test",
                 params={"slot": "llm2" if i % 2 else "llm3"}) for i in range(6)]
        for item in items:
            item.completion_future = asyncio.get_running_loop().create_future()
        self.manager.items.extend(items)
        await self.manager._ensure_llm_workers()
        try:
            await asyncio.wait_for(ready.wait(), 2)
            self.assertEqual(len(started), 3)
            self.assertEqual(sum(item.status == "processing" for item in items), 3)
            self.assertEqual(sum(item.status == "waiting" for item in items), 3)
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(*(item.completion_future for item in items)), 2)
        self.assertEqual(peak, 3)
        self.assertTrue(all(item.status == "completed" for item in items))

    async def test_one_click_stop_cancels_gate_waiters_and_finishes_active_refinement(self):
        started = []
        ready, release = asyncio.Event(), asyncio.Event()

        async def refine(item):
            async with self.engine._limit_llm_request(item.params["slot"]):
                started.append(item.id)
                if len(started) == 3:
                    ready.set()
                await release.wait()
                return {"success": True}

        self.manager._handle_bot_lb_extra_refine = refine
        items = [self.queue_module.QueueItem(id=str(i), type="bot_lb_extra_refine", label="test",
                 params={"slot": "llm2" if i % 2 else "llm3", "one_click_run_id": "preparation"})
                 for i in range(6)]
        for item in items:
            item.completion_future = asyncio.get_running_loop().create_future()
            item.completion_future.add_done_callback(self.manager._mark_completion_future_observed)
        self.manager.items.extend(items)
        await self.manager._ensure_llm_workers()
        try:
            await asyncio.wait_for(ready.wait(), 2)
            active = [item for item in items if item.status == "processing"]
            waiting = [item for item in items if item.status == "waiting"]
            self.assertEqual((len(active), len(waiting)), (3, 3))
            self.assertEqual(await self.manager.cancel_one_click_run("preparation"), 3)
            results = await asyncio.wait_for(asyncio.gather(
                *(item.completion_future for item in waiting), return_exceptions=True), 2)
            self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
            self.assertTrue(all(item.status == "cancelled" for item in waiting))
            self.assertTrue(all(not item.completion_future.done() for item in active))
            self.assertEqual(set(started), {item.id for item in active})
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(
                *(item.completion_future for item in items), return_exceptions=True), 2)
        self.assertTrue(all(item.status == "completed" for item in active))
        self.assertTrue(all(item.status == "cancelled" for item in waiting))
        self.assertEqual(set(started), {item.id for item in active})

    async def test_new_provider_limit_is_used_when_workers_are_refreshed(self):
        before = self.manager._target_llm_workers()
        self.value["draft"]["llmProviders"][0]["maxConcurrency"] = 5
        self.store.save(self.value, self.revision)
        await self.manager._ensure_llm_workers()
        self.assertGreater(self.manager._target_llm_workers(), before)
        self.assertEqual(len(self.manager._llm_worker_tasks), self.manager._target_llm_workers())
