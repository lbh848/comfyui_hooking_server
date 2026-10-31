"""Regression checks for storage separation and the extracted frontend."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app_paths import SOURCE_DIR, config_root, data_root, initialize_layout, install_root
from comfy_installer.credentials import load_civitai_key, save_civitai_key
from comfy_installer.patch_importer import PatchImportError, _safe_destination
from frontend_auth import FrontendAuthController, FrontendAuthManager
from launch import app_url
from runtime_temp import clear_runtime_temp
from vast_backend.settings import load_key_files


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.install = Path(self.temporary.name)
        self.source = self.install / "src"
        self.source.mkdir()

    def test_roots_and_launch_entries(self):
        self.assertEqual(install_root(self.source), self.install)
        self.assertEqual(data_root(self.source), self.install / "data")
        self.assertEqual(config_root(self.source), self.install / "config")
        for app in ("illustration", "asset", "video"):
            self.assertEqual(app_url(app), f"http://127.0.0.1:8189/?app={app}")
        with self.assertRaises(ValueError):
            app_url("unrecognized")

    def test_seed_defaults_preserves_user_edits_and_adds_missing_files(self):
        prompts = self.source / "prompts"
        prompts.mkdir()
        (prompts / "example.txt").write_text("bundled", encoding="utf-8")
        initialize_layout(self.source)
        saved = data_root(self.source) / "prompts" / "example.txt"
        saved.write_text("user edit", encoding="utf-8")
        (prompts / "new.txt").write_text("new default", encoding="utf-8")
        initialize_layout(self.source)
        self.assertEqual(saved.read_text(encoding="utf-8"), "user edit")
        self.assertEqual((saved.parent / "new.txt").read_text(encoding="utf-8"), "new default")

    def test_credentials_are_shared_in_config(self):
        save_civitai_key(self.source, "test-placeholder")
        self.assertEqual(load_civitai_key(self.source), "test-placeholder")
        self.assertEqual(load_key_files(self.source)["civitai_api_key"], "test-placeholder")
        self.assertTrue((config_root(self.source) / "key" / "civitai_key.json").is_file())
        self.assertFalse((self.source / "key").exists())

    def test_rag_installer_uses_data_for_default_and_custom_index(self):
        # Load the real installer/service without creating every modes singleton.
        package = types.ModuleType("layout_rag")
        package.__path__ = [str(SOURCE_DIR / "modes" / "danbooru_rag")]
        with patch.dict(sys.modules, {"layout_rag": package}):
            spec = importlib.util.spec_from_file_location("layout_rag.installer", Path(package.__path__[0]) / "installer.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            installer = module.DanbooruRagIndexInstaller(project_root=self.source)
            expected = data_root(self.source) / "auto_complete"
            self.assertEqual(installer.data_root, expected / "danbooru_rag_data")
            custom = module.DanbooruRagIndexInstaller(project_root=self.source, data_root=expected / "custom")
            self.assertEqual(custom.index_path, expected / "custom" / "lancedb_b")

    def test_patch_targets_and_temporary_cleanup_stay_in_data(self):
        self.assertEqual(_safe_destination(self.source, "asset/test.png"), data_root(self.source) / "asset" / "test.png")
        with self.assertRaises(PatchImportError):
            _safe_destination(self.source, "../config/config.json")
        keep = self.source / "keep.txt"
        keep.write_text("source", encoding="utf-8")
        temporary = data_root(self.source) / "runtime" / "temp"
        temporary.mkdir(parents=True)
        (temporary / "discard.txt").write_text("temporary", encoding="utf-8")
        clear_runtime_temp(self.source)
        self.assertEqual(list(temporary.iterdir()), [])
        self.assertEqual(keep.read_text(encoding="utf-8"), "source")

    def test_packaged_update_keeps_configuration_and_user_data(self):
        spec = importlib.util.spec_from_file_location("update_payload", SOURCE_DIR / "packaging" / "macos" / "update_payload.py")
        updater = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(updater)
        payload = self.install / "payload"
        (payload / "src").mkdir(parents=True)
        (payload / "src" / "example.py").write_text("v1", encoding="utf-8")
        updater.inventory(payload)
        destination = self.install / "installed"
        updater.update(payload, destination)
        (destination / "config").mkdir()
        settings = destination / "config" / "config.json"
        settings.write_text('{"user": true}', encoding="utf-8")
        (payload / "src" / "example.py").write_text("v2", encoding="utf-8")
        updater.inventory(payload)
        updater.update(payload, destination)
        self.assertEqual((destination / "src" / "example.py").read_text(encoding="utf-8"), "v2")
        self.assertEqual(settings.read_text(encoding="utf-8"), '{"user": true}')
        self.assertTrue((destination / "data" / "backups" / "app_updates").is_dir())
        self.assertFalse((destination / "backups").exists())


class FrontendLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_login_and_extracted_static_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = web.Application()
            frontend = SOURCE_DIR / "frontend"
            controller = FrontendAuthController(
                FrontendAuthManager(Path(temporary) / "config" / "key" / "frontend_auth.json"),
                index_file=frontend / "index.html",
                login_file=frontend / "login.html",
            )
            controller.register_routes(app)
            app.router.add_static("/static/", frontend / "static", show_index=False)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/?app=asset")
                self.assertEqual(response.status, 200)
                self.assertIn("config/key/frontend_auth.json", await response.text())
                response = await client.post("/api/frontend-auth/setup", json={"password": "test-password"})
                self.assertEqual(response.status, 200)
                response = await client.get("/?app=asset")
                body = await response.text()
                self.assertIn('/static/app.js', body)
                self.assertIn('/static/app.css', body)
                for name, content_type in (("app.js", "javascript"), ("app.css", "text/css"), ("entry.js", "javascript")):
                    response = await client.get("/static/" + name)
                    self.assertEqual(response.status, 200)
                    self.assertIn(content_type, response.headers["Content-Type"])
                    self.assertEqual(await response.read(), (frontend / "static" / name).read_bytes())
                response = await client.get("/static/")
                self.assertEqual(response.status, 403)
                response = await client.get("/static/%2e%2e/%2e%2e/server.py")
                self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
