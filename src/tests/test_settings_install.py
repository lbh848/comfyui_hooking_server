"""Clean installation requires consent and only deletes the embedded Comfy root."""
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from comfy_installer.downloader import DownloadCancelled
from comfy_installer.service import ComfyInstallerService, InstallerServiceError
from settings_server import create_app
from settings_store import SettingsStore
from test_llm_settings import document


class CleanInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        self.source.mkdir()
        with patch("comfy_installer.service.git_head", return_value=None):
            self.service = ComfyInstallerService(project_root=self.source)
        self.root = self.service.comfy_root

    def old_file(self):
        path = self.root / "models" / "loras" / "old.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old model")
        return path

    def test_existing_folder_requires_confirmation_before_any_install_work(self):
        old = self.old_file()
        self.assertTrue(self.service.status()["comfy_exists"])
        with patch.object(self.service, "unpack_workflow_pack") as unpack:
            with self.assertRaisesRegex(InstallerServiceError, "삭제를 확인"):
                self.service.start_basic_install(workflow_key="fixture", civitai_key="fixture")
            unpack.assert_not_called()
        self.assertEqual(old.read_bytes(), b"old model")

    def test_confirmation_reaches_install_worker(self):
        self.old_file()
        with patch.object(self.service, "unpack_workflow_pack", return_value={"unpacked": {"release_version": "v1"}}), \
                patch("comfy_installer.service.basic_workflow_item_ids", return_value=["fixture"]), \
                patch.object(self.service, "set_civitai_key"), \
                patch.object(self.service, "_start_operation", return_value={"state": "running"}) as start:
            self.service.start_basic_install(workflow_key="fixture", civitai_key="fixture", replace_existing=True)
        self.assertTrue(start.call_args.kwargs["kwargs"]["replace_existing"])

    def test_deletes_all_old_files_without_backup_and_preserves_siblings(self):
        self.old_file().chmod(stat.S_IREAD)
        sibling = self.root.parent / "keep.txt"
        sibling.write_text("outside comfy")
        self.service._clear_comfy_for_install()
        self.assertFalse(self.root.exists())
        self.assertFalse(self.service.status()["comfy_exists"])
        self.assertEqual(list(self.root.parent.iterdir()), [sibling])

    def test_cancel_before_deletion_preserves_files(self):
        old = self.old_file()
        self.service._cancel.set()
        with self.assertRaises(DownloadCancelled):
            self.service._clear_comfy_for_install()
        self.assertTrue(old.exists())

    def test_external_junction_is_never_deleted(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        marker = outside / "keep.txt"
        marker.write_text("keep")
        self.root.parent.mkdir(parents=True)
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(outside), str(self.root))
        else:
            self.root.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(InstallerServiceError, "프로젝트 내부"):
            self.service._clear_comfy_for_install()
        self.assertEqual(marker.read_text(), "keep")

    def test_confirmed_worker_clears_folder_before_installing_source(self):
        self.old_file()
        def source_install(**kwargs):
            self.assertEqual(kwargs["destination"], self.root)
            self.assertFalse(self.root.exists())
            raise DownloadCancelled("Stop before real installation")
        manifest = SimpleNamespace(models=[], comfy={"repository": "fixture", "ref": "a" * 40})
        scope = SimpleNamespace(keep=[], filtered=[], keep_bytes=0)
        with patch("comfy_installer.service.release_install_manifest", return_value=manifest), \
                patch("comfy_installer.service.selection_requirements", return_value={"model_ids": []}), \
                patch.object(self.service, "_preflight_models", return_value=({"gpu_profile": "cpu", "disk": {"free": 1, "required": 1}}, scope)), \
                patch("comfy_installer.service.install_comfy_source", side_effect=source_install) as install:
            self.service._run_install(release_version="v1", selected_item_ids=["fixture"], install_mode="standard", replace_existing=True)
        install.assert_called_once()
        self.assertEqual(self.service.status()["state"], "cancelled")


class CleanInstallApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        dist = self.source / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("fixture")
        value = document()
        value["apiKeys"].update({"workflow-key": "fixture", "other-civitai": "fixture"})
        revision = SettingsStore(self.source).save(value, '"missing"')
        self.installer = Mock()
        self.installer.status.return_value = {"state": "idle", "comfy_exists": True}
        self.installer.start_basic_install.return_value = {"state": "running"}
        self.client = TestClient(TestServer(create_app(dist, self.source, installer=self.installer)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "If-Match": revision}

    async def test_confirmation_is_forwarded_and_missing_confirmation_stays_false(self):
        for confirmed in (False, True):
            response = await self.client.post("/api/setting/install", headers=self.headers,
                json={"install_mode": "standard", **({"replace_existing": True} if confirmed else {})})
            self.assertEqual(response.status, 200)
            self.assertIs(self.installer.start_basic_install.call_args.kwargs["replace_existing"], confirmed)

    async def test_invalid_confirmation_and_stale_settings_never_start(self):
        for invalid in ("true", 1, None):
            response = await self.client.post("/api/setting/install", headers=self.headers,
                json={"install_mode": "standard", "replace_existing": invalid})
            self.assertEqual(response.status, 400)
        stale = await self.client.post("/api/setting/install", headers={**self.headers, "If-Match": '"stale"'},
            json={"install_mode": "standard", "replace_existing": True})
        self.assertEqual(stale.status, 409)
        self.installer.start_basic_install.assert_not_called()
