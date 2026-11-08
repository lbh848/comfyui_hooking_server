"""Training profile persistence, release locks and settings gateway coverage."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from lora_training_presets import LoraTrainingPresets, TrainingPresetConflict
from settings_internal_lora import register_internal_lora


class TrainingPresetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "src"
        (self.root / "presets").mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "presets/lora_training_presets.json"
        (self.root / "presets/lora_training_presets.json").write_bytes(source.read_bytes())
        self.store = LoraTrainingPresets(self.root)

    def settings(self, profile="anima"):
        return deepcopy(next(item["settings"] for item in self.store.catalog(profile)["presets"] if item["scope"] == "builtin"))

    def change(self, action="create", profile="anima", **values):
        body = {"action": action, "profile": profile, "revision": self.store.catalog(profile)["revision"], **values}
        if action == "create":
            body.setdefault("name", "내 학습 프로필")
            body.setdefault("settings", self.settings(profile))
        return self.store.change(body)

    def test_profiles_and_scopes_persist_independently(self):
        shipped = self.store.builtin_file.read_bytes()
        anima = self.settings()
        anima["step_per_image"] = 321
        self.change(settings=anima)
        self.change(profile="sdxl")
        restarted = LoraTrainingPresets(self.root)
        for profile, steps in (("anima", 321), ("sdxl", 400)):
            entries = restarted.catalog(profile)["presets"]
            self.assertEqual({item["scope"] for item in entries}, {"builtin", "local"})
            self.assertEqual(next(item["settings"]["step_per_image"] for item in entries if item["scope"] == "local"), steps)
        self.assertEqual(self.store.builtin_file.read_bytes(), shipped)
        self.assertEqual(self.store.path.parent, self.root.parent / "config")

    def test_builtin_edits_and_scope_transfer_require_session_unlock(self):
        builtin = self.store.catalog("anima")["presets"][0]
        for action, values in (("save", {"settings": self.settings()}), ("rename", {"name": "수정"}),
                               ("delete", {"confirmed": True}), ("scope", {"scope": "local"}),
                               ("create", {"scope": "builtin"})):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.change(action, id=builtin["id"], **values)
        token = self.change("unlock", confirmed=True)["builtin_unlock_token"]
        moved = self.change("scope", id=builtin["id"], scope="local", builtin_unlock_token=token)
        self.assertEqual(next(item["scope"] for item in moved["presets"] if item["id"] == builtin["id"]), "local")
        self.change("scope", id=builtin["id"], scope="builtin", builtin_unlock_token=token)
        self.change("lock", builtin_unlock_token=token)
        with self.assertRaises(ValueError):
            self.change("rename", id=builtin["id"], name="수정", builtin_unlock_token=token)

    def test_user_edit_rename_delete_and_cross_profile_rejection(self):
        identity = self.change()["saved_id"]
        settings = self.settings()
        settings["upscale"] = True
        self.change("save", id=identity, settings=settings)
        renamed = self.change("rename", id=identity, name="새 이름")
        saved = next(item for item in renamed["presets"] if item["id"] == identity)
        self.assertTrue(saved["settings"]["upscale"])
        self.assertEqual(saved["name"], "새 이름")
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.change("save", profile="sdxl", id=identity, settings=self.settings("sdxl"))
        with self.assertRaises(ValueError):
            self.change("delete", id=identity)
        self.assertEqual(self.store.path.read_bytes(), before)
        remaining = self.change("delete", id=identity, confirmed=True)
        self.assertTrue(all(item["scope"] == "builtin" for item in remaining["presets"]))

    def test_duplicate_names_and_stale_revision_do_not_overwrite(self):
        revision = self.store.catalog("anima")["revision"]
        identity = self.change()["saved_id"]
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.change(id=identity)
        with self.assertRaises(TrainingPresetConflict):
            self.change("save", id=identity, revision=revision, settings=self.settings())
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_invalid_settings_never_replace_saved_file(self):
        identity = self.change()["saved_id"]
        before = self.store.path.read_bytes()
        for key, value in (("profile", "sdxl"), ("step_per_image", 0), ("dim", True), ("alpha", 1.5),
                           ("il_rate", float("nan")), ("gen_w", float("inf")), ("upscale", "false"),
                           ("multi_img_folder_name", "../outside"), ("lora_save_path", "other-project")):
            settings = self.settings()
            settings[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.change("save", id=identity, settings=settings)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.store.path.write_text('{"version": 999}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.change({"action": "create", "profile": "anima"})
        self.assertEqual(self.store.path.read_text(encoding="utf-8"), '{"version": 999}')

    def test_release_restoring_moved_preset_preserves_user_identity(self):
        builtin_bytes = self.store.builtin_file.read_bytes()
        builtin = self.store.catalog("anima")["presets"][0]
        token = self.change("unlock", confirmed=True)["builtin_unlock_token"]
        self.change("scope", id=builtin["id"], scope="local", builtin_unlock_token=token)
        self.store.builtin_file.write_bytes(builtin_bytes)
        catalog = self.store.catalog("anima")
        self.assertEqual(len({item["id"] for item in catalog["presets"]}), 2)
        self.assertEqual(next(item["id"] for item in catalog["presets"] if item["scope"] == "local"), builtin["id"])
        shipped = next(item for item in catalog["presets"] if item["scope"] == "builtin")
        self.change("rename", id=shipped["id"], name="배포 이름", builtin_unlock_token=token)
        self.assertEqual(json.loads(self.store.builtin_file.read_text(encoding="utf-8"))["presets"][0]["id"], builtin["id"])


class TrainingPresetGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_do_not_launch_training_worker_and_validate_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            app = web.Application()
            runtime = register_internal_lora(app, Path(directory), "2")
            async with TestClient(TestServer(app)) as client:
                route = "/api/setting/internal-lora/training-presets"
                headers = {"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
                with patch.object(runtime, "start", new_callable=AsyncMock) as start:
                    self.assertEqual((await client.get(route + "?profile=anima")).status, 403)
                    response = await client.get(route + "?profile=anima", headers=headers)
                    self.assertEqual(response.status, 200)
                    catalog = await response.json()
                    self.assertEqual(catalog["presets"], [])
                    self.assertEqual((await client.get(route + "?profile=invalid", headers=headers)).status, 400)
                    self.assertEqual((await client.get(route + "?profile=anima", headers={**headers, "X-Comfy-Settings-Schema": "1"})).status, 409)
                    self.assertEqual((await client.post(route, headers=headers, json=[])).status, 400)
                    self.assertEqual((await client.post(route, headers=headers, json={"action": "create", "profile": "anima", "revision": "stale"})).status, 409)
                    start.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
