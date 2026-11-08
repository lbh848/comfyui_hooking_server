"""Gallery deletion removes only the confirmed character or outfit/expression folder."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from asset_character_presets import AssetCharacterPresets
from settings_assets import allowed_asset_path, register_assets

asset_mode = importlib.import_module("modes.asset_mode")


class AssetGalleryDeletionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.assets = self.root / "assets"
        asset_dir = patch.object(asset_mode, "ASSET_DIR", str(self.assets))
        asset_dir.start()
        self.addCleanup(asset_dir.stop)
        self.mode = asset_mode.AssetMode()
        self.mode.load_tags = Mock()
        self.mode.character_presets = AssetCharacterPresets(self.root / "tags.json", self.root / "builtin.json")
        self.mode.character_presets.change({"action": "create", "name": "테스트 캐릭터"})
        self.target = self.folder("테스트 캐릭터", "기본 복장", "미소")
        self.sibling = self.folder("테스트 캐릭터", "기본 복장", "화남")
        self.other = self.folder("다른 캐릭터", "기본 복장", "미소")

    def folder(self, *names):
        path = self.assets.joinpath(*names)
        path.mkdir(parents=True)
        (path / "image.webp").write_bytes(b"test-image")
        (path / "image_prompt.json").write_text("{}", encoding="utf-8")
        (path / "_representative.json").write_text(json.dumps({"filename": "image.webp"}), encoding="utf-8")
        (path / "_expr_group.json").write_text(json.dumps({"group_id": "group"}), encoding="utf-8")
        return path

    def test_combination_deletes_images_and_metadata_but_preserves_siblings(self):
        result = self.mode.delete_combination("테스트 캐릭터", "기본 복장", "미소")
        self.assertTrue(result["success"])
        self.assertFalse(self.target.exists())
        self.assertTrue((self.sibling / "image.webp").exists())
        self.assertTrue((self.other / "image.webp").exists())
        self.assertEqual(self.mode.get_outfit_groups("테스트 캐릭터"), {
            "group": [{"outfit": "기본 복장", "expression": "화남"}],
        })
        self.assertIn("테스트 캐릭터", self.mode.character_presets.catalog()["characters"])
        self.assertFalse(self.mode.delete_combination("테스트 캐릭터", "기본 복장", "미소")["success"])

    def test_character_deletes_definition_and_all_its_assets(self):
        loras = {"loras": {"테스트 캐릭터": {}, "다른 캐릭터": {}}}
        save = Mock()
        with patch.dict("sys.modules", {"modes.lora_mode": SimpleNamespace(_load_lora_manage=lambda: loras, _save_lora_manage=save)}):
            result = self.mode.remove_character("테스트 캐릭터")
        self.assertTrue(result["success"])
        self.assertNotIn("테스트 캐릭터", self.mode.character_presets.catalog()["characters"])
        self.assertFalse(self.target.parent.parent.exists())
        self.assertTrue((self.other / "image.webp").exists())
        self.assertEqual(loras, {"loras": {"다른 캐릭터": {}}})
        save.assert_called_once_with(loras)

    def test_locked_builtin_character_keeps_definition_and_assets(self):
        self.mode.character_presets.builtin_file.write_text(json.dumps({"characters": {"다른 캐릭터": {}}}), encoding="utf-8")
        self.assertFalse(self.mode.remove_character("다른 캐릭터")["success"])
        self.assertTrue((self.other / "image.webp").exists())
        self.assertIn("다른 캐릭터", self.mode.character_presets.catalog()["characters"])

    def test_invalid_paths_never_delete_parent_directories(self):
        for invalid in ("", " ", ".", "..", None):
            for index in range(3):
                names = ["테스트 캐릭터", "기본 복장", "미소"]
                names[index] = invalid
                with self.subTest(names=names):
                    self.assertFalse(self.mode.delete_combination(*names)["success"])
                    self.assertTrue((self.target / "image.webp").exists())
        self.mode.character_presets.change({"action": "create", "name": ".."})
        self.assertFalse(self.mode.remove_character("..")["success"])
        self.assertIn("..", self.mode.character_presets.catalog()["characters"])
        self.assertTrue(self.root.exists())


class AssetGalleryDeletionGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_folder_deletion_reaches_runtime_only_with_settings_headers(self):
        received = []

        class Runtime:
            async def request(self, request, *, asset):
                path = "/api/" + request.match_info["path"]
                if not asset or not allowed_asset_path(path, request.method):
                    raise web.HTTPNotFound()
                received.append(await request.json())
                return web.json_response({"success": True})

        app = web.Application()
        register_assets(app, Runtime(), "2")
        async with TestClient(TestServer(app)) as client:
            path = "/api/asset/asset_mode/delete_combination"
            body = {"character": "테스트 캐릭터", "outfit": "기본 복장", "expression": "미소"}
            self.assertEqual((await client.post(path, json=body)).status, 403)
            self.assertEqual((await client.post(path, json=body, headers={"X-Comfy-Settings": "1"})).status, 409)
            self.assertEqual(received, [])
            response = await client.post(path, json=body, headers={"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"})
            self.assertEqual(response.status, 200)
            self.assertEqual(received, [body])
        for method in ("GET", "HEAD", "DELETE", "PUT"):
            self.assertFalse(allowed_asset_path("/api/asset_mode/delete_combination", method))
        self.assertFalse(allowed_asset_path("/api/asset_mode/delete_combination/extra", "POST"))


if __name__ == "__main__":
    unittest.main()
