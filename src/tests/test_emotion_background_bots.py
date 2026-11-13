"""Automatic background bots do not share manual settings or character bots."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from asset_preset_scope import write_scope_documents
from emotion_background_bots import EmotionBackgroundBots
from json_store import write_json_atomic
from settings_illustration import allowed_illustration_path, register_illustration_config


class BackgroundBotsFixture:
    def setup_store(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.builtin = self.root / "presets" / "emotion_background_bots.json"
        self.builtin.parent.mkdir()
        self.builtin.write_text(json.dumps({
            "version": 1, "bots": [{"name": "배포 배경", "settings": {"seed": 42}}],
        }, ensure_ascii=False), encoding="utf-8")
        self.store = EmotionBackgroundBots(self.root / "settings.json", self.builtin)

    def change(self, action, **body):
        return self.store.change({"action": action, **body})

    def unlock(self):
        return self.change("unlock", confirmed=True)["builtin_unlock_token"]


class BackgroundBotsTests(BackgroundBotsFixture, unittest.TestCase):
    def setUp(self):
        self.setup_store()

    def test_selection_and_user_bots_survive_reload_without_rewriting_distribution(self):
        before = self.builtin.read_bytes()
        result = self.change("create", name="숲 배경")
        self.assertEqual(result["bot_selected"], "숲 배경")
        self.change("create", name="도시 배경")
        self.change("select", name="숲 배경")
        reloaded = EmotionBackgroundBots(self.root / "settings.json", self.builtin)
        self.assertEqual(reloaded.snapshot()["bot_selected"], "숲 배경")
        self.assertEqual(self.builtin.read_bytes(), before)
        self.change("rename", name="숲 배경", new_name="아침 숲")
        self.assertEqual(self.store.snapshot()["bot_selected"], "아침 숲")
        self.change("delete", name="아침 숲")
        self.assertEqual(self.store.snapshot()["bot_selected"], "배포 배경")

    def test_builtin_mutations_require_session_unlock_and_lock_revokes_it(self):
        for action, body in (
            ("create", {"name": "새 배포", "scope": "builtin"}),
            ("rename", {"name": "배포 배경", "new_name": "변경"}),
            ("delete", {"name": "배포 배경"}),
            ("scope", {"name": "배포 배경", "scope": "local"}),
        ):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.change(action, **body)
        with self.assertRaises(ValueError):
            self.change("unlock")
        token = self.unlock()
        self.change("create", name="새 배포", scope="builtin", builtin_unlock_token=token)
        self.change("lock", builtin_unlock_token=token)
        with self.assertRaises(ValueError):
            self.change("delete", name="새 배포", builtin_unlock_token=token)

    def test_locked_builtin_can_be_selected_and_copied_into_independent_user_bot(self):
        self.change("select", name="배포 배경")
        result = self.change("duplicate", name="배포 배경", new_name="내 배경")
        copied = next(bot for bot in result["bots"] if bot["name"] == "내 배경")
        self.assertEqual(copied["scope"], "local")
        self.assertEqual(copied["settings"], {"seed": 42})
        self.change("delete", name="내 배경")
        self.assertEqual(self.store.snapshot()["bots"][0]["name"], "배포 배경")

    def test_scope_transfer_and_failed_transfer_preserve_documents(self):
        self.change("create", name="사용자 배경")
        with self.assertRaises(ValueError):
            self.change("scope", name="사용자 배경", scope="builtin")
        token = self.unlock()
        before = {path: json.loads(path.read_text(encoding="utf-8")) for path in (self.builtin, self.store.local_file)}

        def fail_local(path, document):
            if path == self.store.local_file:
                raise OSError("disk unavailable")
            write_json_atomic(path, document)

        with patch("emotion_background_bots.write_scope_documents",
                   side_effect=lambda documents: write_scope_documents(documents, writer=fail_local)):
            with self.assertRaises(OSError):
                self.change("scope", name="사용자 배경", scope="builtin", builtin_unlock_token=token)
        self.assertEqual(before, {path: json.loads(path.read_text(encoding="utf-8")) for path in before})
        result = self.change("scope", name="사용자 배경", scope="builtin", builtin_unlock_token=token)
        self.assertEqual(next(bot["scope"] for bot in result["bots"] if bot["name"] == "사용자 배경"), "builtin")

    def test_invalid_names_scopes_actions_and_missing_bots_do_not_write(self):
        for name in (None, "", " ", "x" * 101, "line\nbreak", "배포 배경"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.change("create", name=name)
        with self.assertRaises(ValueError):
            self.change("create", name="봇", scope="other")
        with self.assertRaises(ValueError):
            self.change("update", name="배포 배경")
        with self.assertRaises(KeyError):
            self.change("select", name="missing")
        self.assertFalse(self.store.local_file.exists())

    def test_empty_catalog_stays_empty_and_manual_and_character_files_are_untouched(self):
        self.builtin.unlink()
        manual = self.root / "emotion_background_presets.json"
        character = self.root / "bot.json"
        config = self.root / "settings.json"
        for path in (manual, character, config):
            path.write_text('{"sentinel": "unchanged"}', encoding="utf-8")
        self.assertEqual(self.store.snapshot(), {"bots": [], "bot_selected": ""})
        self.change("create", name="자동 배경")
        result = self.change("delete", name="자동 배경")
        self.assertEqual(result, {"bots": [], "bot_selected": ""})
        self.assertEqual(self.store.snapshot(), result)
        for path in (manual, character, config):
            self.assertEqual(path.read_text(encoding="utf-8"), '{"sentinel": "unchanged"}')


class BackgroundBotsApiTests(BackgroundBotsFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.setup_store()
        server = SimpleNamespace(app=web.Application(), CONFIG_FILE=self.root / "settings.json",
                                 app_config={}, DEFAULT_CONFIG={})
        register_illustration_config(server, model_presets=Mock(), background_bot_builtin_file=self.builtin)
        self.client = TestClient(TestServer(server.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_gateway_and_emotion_scope_and_persistent_crud(self):
        path = "/api/illustration/background-bots"
        for method in ("GET", "HEAD", "POST"):
            self.assertTrue(allowed_illustration_path(path, method))
        self.assertFalse(allowed_illustration_path(path, "DELETE"))
        self.assertEqual((await self.client.get(path)).status, 400)
        self.assertEqual((await self.client.post(path, json={"action": "create", "name": "봇"})).status, 400)
        path += "?scope=emotion"
        response = await self.client.post(path, json={"action": "create", "name": "사용자 봇"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual((await response.json())["bot_selected"], "사용자 봇")
        self.assertEqual(self.store.snapshot()["bot_selected"], "사용자 봇")
        self.assertEqual((await self.client.post(path, json={"action": "delete", "name": "배포 배경"})).status, 400)
        self.assertEqual((await self.client.post(path, json={"action": "select", "name": "missing"})).status, 404)
        self.assertEqual((await self.client.post(path, json=[])).status, 400)


if __name__ == "__main__":
    unittest.main()
