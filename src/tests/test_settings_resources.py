"""Local resource inventory must reflect files, never saved/sample entries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from app_paths import data_root
from comfy_installer.manifest import ManifestError
from settings_resources import ResourceDeleteBlockedError, ResourceMoveBlockedError, ResourceLibrary
from settings_server import create_app


class ResourceLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.install = Path(self.temporary.name) / "installation"
        self.source = self.install / "src"
        self.source.mkdir(parents=True)
        self.library = ResourceLibrary(self.source)
        self.models = data_root(self.source) / "comfy" / "models"

    def file(self, name, content=b"model fixture"):
        path = self.models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def junction(self, link, target):
        link.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_missing_installation_is_empty_without_creating_directories(self):
        self.assertEqual(self.library.scan(), {"models": [], "loras": []})
        self.assertFalse(self.models.exists())

    def test_nested_files_metadata_previews_and_stable_ids(self):
        self.file("checkpoints/nested/model.SAFETENSORS")
        self.file("text_encoders/encoder.gguf")
        self.file("controlnet/control.pth")
        self.file("loras/style/actual.safetensors")
        self.file("loras/style/actual.metadata.json", json.dumps({"baseModel": "SDXL", "trainedWords": ["style", "soft"]}).encode())
        preview = self.file("loras/style/actual.preview.png", b"preview fixture")
        self.file("loras/unfinished.safetensors.part")
        self.file("loras/empty.safetensors", b"")
        self.file("loras/readme.txt")
        inventory = self.library.scan()
        self.assertEqual(len(inventory["models"]), 3)
        self.assertEqual({item["kind"] for item in inventory["models"]}, {"Checkpoint", "Text encoder", "ControlNet"})
        self.assertEqual(len(inventory["loras"]), 1)
        lora = inventory["loras"][0]
        self.assertEqual(lora["size"], str(len(b"model fixture")))
        self.assertTrue(all(item["size"] == lora["size"] for item in inventory["models"]))
        self.assertEqual(lora["path"], "loras/style/actual.safetensors")
        self.assertEqual((lora["family"], lora["trigger"]), ("SDXL", "style, soft"))
        self.assertEqual(self.library.thumbnail(lora["id"]), preview)
        self.assertEqual(self.library.scan(), inventory)
        self.assertIsNone(self.library.thumbnail("../../config/settings.json"))

    def test_refresh_adds_and_removes_files_and_previews(self):
        model = self.file("diffusion_models/real.safetensors")
        self.file("diffusion_models/real.png", b"preview fixture")
        identity = self.library.scan()["models"][0]["id"]
        model.unlink()
        self.assertIsNone(self.library.thumbnail(identity))
        self.assertEqual(self.library.scan(), {"models": [], "loras": []})
        self.file("loras/new.pt")
        self.assertEqual(len(self.library.scan()["loras"]), 1)

    def test_installed_lora_opens_its_model_version_page_without_changing_download(self):
        self.file("loras/managed.safetensors")
        metadata = self.file("loras/managed.metadata.json", json.dumps({
            "source": "civitai", "version_id": 102,
            "civitai": {"id": 102, "modelId": 11, "model": {"name": "Managed LoRA"}},
        }).encode())
        original = metadata.read_bytes()
        manifest = Mock(models=[{
            "relative_path": "models/loras/managed.safetensors",
            "url": "https://civitai.com/api/download/models/102?fileId=7&token=private",
        }])
        with patch("settings_resources.load_install_manifest", return_value=manifest):
            entry = self.library.scan()["loras"][0]
            self.assertEqual(entry["downloadUrl"], "https://civitai.com/api/download/models/102?fileId=7")
            self.assertEqual(entry["sourceUrl"], "https://civitai.com/models/11?modelVersionId=102")
            self.assertEqual(self.library.details(entry["id"])["sourceUrl"], entry["sourceUrl"])
        self.assertEqual(metadata.read_bytes(), original)

    def test_source_pages_support_existing_sidecars_and_huggingface_manifest(self):
        cases = [
            ({"modelVersion": {"id": 102, "modelId": 11}}, None, "https://civitai.com/models/11?modelVersionId=102"),
            ({"civitai": {"model": {"id": 11}, "images": [{"url": "https://images.example/preview.png"}]}}, "https://civitai.com/api/download/models/102", "https://civitai.com/models/11?modelVersionId=102"),
            ({"hf_url": "https://huggingface.co/user/repo/resolve/main/style.safetensors?download=true&token=private"}, None, "https://huggingface.co/user/repo"),
            ({}, "https://huggingface.co/user/repo/resolve/revision/style.safetensors?download=true", "https://huggingface.co/user/repo"),
            ({"download_url": "https://civitai.com/models/11/style?modelVersionId=102&token=private"}, None, "https://civitai.com/models/11?modelVersionId=102"),
        ]
        self.file("loras/style.pt")
        for metadata, manifest_url, expected in cases:
            with self.subTest(expected=expected, metadata=metadata):
                self.file("loras/style.metadata.json", json.dumps(metadata).encode())
                manifest = Mock(models=[{"relative_path": "models/loras/style.pt", "url": manifest_url}])
                with patch("settings_resources.load_install_manifest", return_value=manifest):
                    entry = self.library.scan()["loras"][0]
                    self.assertEqual(entry["sourceUrl"], expected)
                    self.assertEqual(self.library.details(entry["id"])["sourceUrl"], expected)

    def test_missing_source_page_never_falls_back_to_a_file_download(self):
        self.file("loras/style.pt")
        for url in ("https://civitai.com/api/download/models/102", "https://files.example/style.safetensors", "https://user:private@huggingface.co/user/repo"):
            with self.subTest(url=url):
                self.file("loras/style.metadata.json", json.dumps({"download_url": url}).encode())
                entry = self.library.scan()["loras"][0]
                self.assertEqual(entry["sourceUrl"], "")
                details = self.library.details(entry["id"])
                self.assertEqual(details["sourceUrl"], "")
                if "civitai.com" in url:
                    self.assertEqual(details["civitaiVersionId"], "102")

    def test_external_junctions_and_extra_path_config_are_not_read(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (outside / "external.safetensors").write_bytes(b"external")
        self.file("checkpoints/internal.safetensors")
        self.junction(self.models / "external-link", outside)
        (self.models.parent / "extra_model_paths.yaml").write_text(f"external:\n  checkpoints: {outside.as_posix()}\n")
        inventory = self.library.scan()
        self.assertEqual([item["name"] for item in inventory["models"]], ["internal"])
        self.assertEqual(inventory["loras"], [])

    def test_external_models_root_is_not_read(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (outside / "external.safetensors").write_bytes(b"external")
        self.junction(self.models, outside)
        self.assertEqual(self.library.scan(), {"models": [], "loras": []})

    def test_scan_failure_is_not_reported_as_an_empty_installation(self):
        self.file("checkpoints/real.pt")
        with patch("settings_resources.os.walk", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.library.scan()

    def test_delete_removes_only_selected_model_and_its_sidecars(self):
        model = self.file("loras/nested/style.safetensors")
        sidecars = [self.file("loras/nested/style" + suffix) for suffix in (".metadata.json", ".civitai.info", ".json", ".preview.png", ".jpg")]
        neighbor = self.file("loras/nested/other.safetensors")
        unrelated = self.file("loras/nested/style.txt")
        identity = next(item["id"] for item in self.library.scan()["loras"] if item["name"] == "style")
        self.assertEqual(self.library.delete(identity), {"deleted": identity, "warning": ""})
        self.assertFalse(model.exists())
        self.assertTrue(all(not path.exists() for path in sidecars))
        self.assertTrue(neighbor.exists())
        self.assertTrue(unrelated.exists())
        self.assertIsNone(self.library.thumbnail(identity))
        with self.assertRaises(FileNotFoundError):
            self.library.delete(identity)

    def test_manifest_models_and_loras_are_protected_by_path_with_sidecars(self):
        paths = ["checkpoints/managed.safetensors", "loras/nested/managed.safetensors"]
        protected = [self.file(path) for path in paths]
        sidecars = [self.file(path.with_suffix(".metadata.json").relative_to(self.models).as_posix(), b"{}") for path in protected]
        unmanaged = self.file("loras/optional/managed.safetensors")
        manifest = Mock(models=[{"relative_path": "models/" + path} for path in paths])
        with patch("settings_resources.load_install_manifest", return_value=manifest):
            inventory = self.library.scan()
            for item in inventory["models"] + inventory["loras"]:
                with self.subTest(path=item["path"]):
                    if item["path"] in paths:
                        self.assertIn("설치 명세", item["deleteBlockedReason"])
                        with self.assertRaisesRegex(ResourceDeleteBlockedError, "설치 명세"):
                            self.library.delete(item["id"])
                    else:
                        self.assertEqual(item["deleteBlockedReason"], "")
                        self.library.delete(item["id"])
        self.assertTrue(all(path.is_file() for path in protected + sidecars))
        self.assertFalse(unmanaged.exists())

    def test_manifest_protection_resolves_internal_aliases_and_path_separators(self):
        model = self.file("loras/z-managed/required.pt")
        self.junction(self.models / "loras" / "a-alias", model.parent)
        manifest_path = "models\\loras\\z-managed\\required.pt"
        if sys.platform == "win32":
            manifest_path = manifest_path.upper()
        with patch("settings_resources.load_install_manifest", return_value=Mock(models=[{"relative_path": manifest_path}])):
            entry = self.library.scan()["loras"][0]
            self.assertEqual(entry["path"], "loras/a-alias/required.pt")
            with self.assertRaises(ResourceDeleteBlockedError):
                self.library.delete(entry["id"])
        self.assertTrue(model.is_file())

    def test_delete_reloads_manifest_after_inventory(self):
        model = self.file("loras/newly-managed.pt")
        manifest = Mock(models=[])
        with patch("settings_resources.load_install_manifest", return_value=manifest):
            entry = self.library.scan()["loras"][0]
            self.assertEqual(entry["deleteBlockedReason"], "")
            manifest.models.append({"relative_path": "models/loras/newly-managed.pt"})
            with self.assertRaises(ResourceDeleteBlockedError):
                self.library.delete(entry["id"])
        self.assertTrue(model.is_file())

    def test_unavailable_manifest_keeps_inventory_but_blocks_deletion(self):
        model = self.file("checkpoints/local.pt")
        preview = self.file("checkpoints/local.png")
        for error in (ManifestError("missing"), ManifestError("invalid")):
            with self.subTest(error=str(error)), patch("settings_resources.load_install_manifest", side_effect=error):
                entry = self.library.scan()["models"][0]
                self.assertIn("명세를 확인할 수 없어", entry["deleteBlockedReason"])
                with self.assertRaises(ResourceDeleteBlockedError):
                    self.library.delete(entry["id"])
        self.assertTrue(model.is_file())
        self.assertTrue(preview.is_file())

    def test_delete_keeps_sidecars_shared_by_another_model_format(self):
        self.file("checkpoints/shared.safetensors")
        self.file("checkpoints/shared.ckpt")
        preview = self.file("checkpoints/shared.preview.webp")
        metadata = self.file("checkpoints/shared.metadata.json")
        first, second = self.library.scan()["models"]
        self.library.delete(first["id"])
        self.assertTrue(preview.exists())
        self.assertTrue(metadata.exists())
        self.library.delete(second["id"])
        self.assertFalse(preview.exists())
        self.assertFalse(metadata.exists())

    def test_delete_rejects_paths_and_stale_ids_after_external_junction_replacement(self):
        model = self.file("loras/private.safetensors")
        identity = self.library.scan()["loras"][0]["id"]
        with self.assertRaises(FileNotFoundError):
            self.library.delete("../../config/settings.json")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        external = outside / model.name
        model.replace(external)
        model.parent.rmdir()
        self.junction(model.parent, outside)
        with self.assertRaises(FileNotFoundError):
            self.library.delete(identity)
        self.assertEqual(external.read_bytes(), b"model fixture")

    def test_delete_reports_sidecar_failure_after_successful_model_deletion(self):
        model = self.file("loras/style.pt")
        preview = self.file("loras/style.png")
        identity = self.library.scan()["loras"][0]["id"]
        unlink = Path.unlink
        def fail_preview(path, *args, **kwargs):
            if path == preview:
                raise PermissionError("busy")
            return unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_preview):
            result = self.library.delete(identity)
        self.assertEqual(result["deleted"], identity)
        self.assertIn("style.png", result["warning"])
        self.assertFalse(model.exists())
        self.assertTrue(preview.exists())


class ResourceMoveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        self.source.mkdir()
        self.library = ResourceLibrary(self.source)
        self.models = self.library.root
        self.manifest = Mock(models=[])
        patcher = patch("settings_resources.load_install_manifest", return_value=self.manifest)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.model = self.file("checkpoints/nested/anima.safetensors")
        self.metadata = self.file("checkpoints/nested/anima.metadata.json", b'{"baseModel":"Anima","trainedWords":["style"]}')
        self.preview = self.file("checkpoints/nested/anima.preview.png", b"preview fixture")
        self.identity = self.library.scan()["models"][0]["id"]

    file = ResourceLibraryTests.file
    junction = ResourceLibraryTests.junction

    def test_move_updates_inventory_and_preserves_all_sidecar_formats(self):
        extra = [self.file("checkpoints/nested/anima.safetensors.civitai.info", b'{"modelId":42}'),
                 self.file("checkpoints/nested/anima.jpg", b"alternate preview")]
        originals = {path.name: path.read_bytes() for path in [self.model, self.metadata, self.preview, *extra]}
        result = self.library.move(self.identity, "diffusion_models", "illustration")
        moved = result["item"]
        self.assertEqual(moved["path"], "diffusion_models/illustration/anima.safetensors")
        self.assertEqual(moved["kind"], "Diffusion")
        self.assertEqual(result["moved"], self.identity)
        self.assertEqual(result["warning"], "")
        self.assertNotEqual(moved["id"], self.identity)
        self.assertFalse(self.model.exists())
        for name, payload in originals.items():
            self.assertEqual((self.models / "diffusion_models/illustration" / name).read_bytes(), payload)
            self.assertFalse((self.model.parent / name).exists())
        self.assertEqual(self.library.thumbnail(moved["id"]).read_bytes(), b"preview fixture")
        self.assertIsNone(self.library.thumbnail(self.identity))
        self.assertEqual(self.library.scan()["models"], [moved])
        self.assertEqual(moved["family"], "Anima")
        self.assertEqual(moved["trigger"], "style")
        with self.assertRaises(FileNotFoundError):
            self.library.move(self.identity, "vae")

    def test_shared_sidecars_are_independent_copies_and_specific_metadata_moves(self):
        sibling = self.file("checkpoints/nested/anima.ckpt")
        specific = self.file("checkpoints/nested/anima.safetensors.json", b'{"specific":true}')
        result = self.library.move(self.identity, "diffusion_models")
        self.assertTrue(sibling.exists())
        self.assertTrue(self.metadata.exists())
        self.assertTrue(self.preview.exists())
        self.assertFalse(specific.exists())
        moved = self.models / result["item"]["path"]
        moved.with_suffix(".metadata.json").write_bytes(b"changed")
        moved.with_suffix(".preview.png").write_bytes(b"changed")
        self.assertIn(b"Anima", self.metadata.read_bytes())
        self.assertEqual(self.preview.read_bytes(), b"preview fixture")

    def test_conflicts_never_overwrite_or_remove_originals(self):
        for filename in ("anima.safetensors", "anima.SAFETENSORS", "anima.ckpt", "anima.safetensors.part", "anima.metadata.json", "anima.safetensors.json", "anima.preview.webp", "anima.jpg"):
            with self.subTest(filename=filename):
                existing = self.file("diffusion_models/" + filename, b"existing")
                with self.assertRaises(FileExistsError):
                    self.library.move(self.identity, "diffusion_models")
                self.assertEqual(existing.read_bytes(), b"existing")
                self.assertTrue(self.model.exists())
                self.assertTrue(self.metadata.exists())
                existing.unlink()

    def test_invalid_destinations_and_same_location_do_not_move_files(self):
        for folder, subfolder in (("unknown", ""), ("loras", ""), ("checkpoints", "nested"),
                                  ("diffusion_models", "../escape"), ("diffusion_models", "/absolute"),
                                  ("diffusion_models", "C:/outside"), ("diffusion_models", "hidden/.private"),
                                  ("diffusion_models", "CON"), ("diffusion_models", "bad\\path"),
                                  ("diffusion_models", []), (None, "")):
            with self.subTest(folder=folder, subfolder=subfolder), self.assertRaises(ValueError):
                self.library.move(self.identity, folder, subfolder)
        self.assertTrue(self.model.exists())
        self.junction(self.models / "diffusion_models/alias", self.model.parent)
        with self.assertRaises(ValueError):
            self.library.move(self.identity, "diffusion_models", "alias")

    def test_external_junction_destination_is_rejected(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.junction(self.models / "diffusion_models", outside)
        with self.assertRaises(ValueError):
            self.library.move(self.identity, "diffusion_models")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertTrue(self.model.exists())

    def test_current_manifest_protects_source_and_destination(self):
        for relative in ("models/checkpoints/nested/anima.safetensors", "models/diffusion_models/anima.safetensors"):
            self.manifest.models = [{"relative_path": relative}]
            with self.subTest(relative=relative), self.assertRaises(ResourceMoveBlockedError):
                self.library.move(self.identity, "diffusion_models")
        with patch("settings_resources.load_install_manifest", side_effect=ManifestError("unavailable")):
            with self.assertRaises(ResourceMoveBlockedError):
                self.library.move(self.identity, "diffusion_models")
        self.assertTrue(self.model.exists())

    def test_publish_and_source_unlink_failures_roll_back_destination(self):
        import os
        link, unlink = os.link, Path.unlink
        target = self.models / "diffusion_models"
        for failure in (self.preview, self.model, "unlink"):
            def fail_link(source, destination):
                if source == failure:
                    raise PermissionError("busy")
                return link(source, destination)
            def fail_unlink(path, *args, **kwargs):
                if failure == "unlink" and path == self.model:
                    raise PermissionError("busy")
                return unlink(path, *args, **kwargs)
            with self.subTest(failure=failure), patch("settings_resources.os.link", side_effect=fail_link), patch.object(Path, "unlink", fail_unlink):
                with self.assertRaises(PermissionError):
                    self.library.move(self.identity, "diffusion_models")
            self.assertEqual(list(target.iterdir()), [])
            self.assertTrue(self.model.exists())
            self.assertTrue(self.preview.exists())
            self.assertTrue(self.metadata.exists())

    def test_sidecar_cleanup_failure_reports_success_with_warning(self):
        unlink = Path.unlink
        def fail_preview(path, *args, **kwargs):
            if path == self.preview:
                raise PermissionError("busy")
            return unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_preview):
            result = self.library.move(self.identity, "diffusion_models")
        self.assertIn(self.preview.name, result["warning"])
        self.assertTrue((self.models / result["item"]["path"]).is_file())
        self.assertFalse(self.model.exists())

    def test_loras_can_change_subfolder_but_not_model_category(self):
        lora = self.file("loras/style.pt")
        identity = self.library.scan()["loras"][0]["id"]
        with self.assertRaises(ValueError):
            self.library.move(identity, "diffusion_models")
        result = self.library.move(identity, "loras", "anima")
        self.assertEqual(result["item"]["path"], "loras/anima/style.pt")
        self.assertFalse(lora.exists())
        self.assertEqual(self.library.scan()["loras"], [result["item"]])


class ResourceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        self.dist = self.source / "settings_ui" / "dist"
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text("settings fixture")
        self.client = TestClient(TestServer(create_app(self.dist, self.source)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1"}

    async def test_resources_require_same_origin_header_and_return_real_files(self):
        denied = await self.client.get("/api/setting/resources", headers={"Host": "127.0.0.1:8201"})
        self.assertEqual(denied.status, 403)
        first = await self.client.get("/api/setting/resources", headers=self.headers)
        self.assertEqual(await first.json(), {"models": [], "loras": []})
        self.assertEqual(first.headers["Cache-Control"], "no-store")
        model = data_root(self.source) / "comfy" / "models" / "loras" / "local.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model fixture")
        model.with_suffix(".png").write_bytes(b"preview fixture")
        inventory = await (await self.client.get("/api/setting/resources", headers=self.headers)).json()
        self.assertEqual(len(inventory["loras"]), 1)
        preview = await self.client.get(inventory["loras"][0]["thumbnail"], headers=self.headers)
        self.assertEqual(await preview.read(), b"preview fixture")
        missing = await self.client.get("/api/setting/resources/thumbnail/unknown", headers=self.headers)
        self.assertEqual(missing.status, 404)
        self.assertFalse((Path(self.temporary.name) / "config").exists())

    async def test_read_failure_returns_an_error(self):
        with patch("settings_resources.ResourceLibrary.scan", side_effect=PermissionError("denied")):
            response = await self.client.get("/api/setting/resources", headers=self.headers)
        self.assertEqual(response.status, 500)
        self.assertIn("error", await response.json())

    async def test_move_requires_same_origin_and_updates_inventory(self):
        library = ResourceLibrary(self.source)
        model = library.root / "checkpoints/anima.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model fixture")
        identity = library.scan()["models"][0]["id"]
        url = f"/api/setting/resources/{identity}/move"
        body = {"folder": "diffusion_models", "subfolder": "anima"}
        for headers in ({"Host": self.headers["Host"]}, {**self.headers, "Origin": "https://untrusted.example"}):
            self.assertEqual((await self.client.post(url, headers=headers, json=body)).status, 403)
            self.assertTrue(model.exists())
        moved = await self.client.post(url, headers=self.headers, json=body)
        self.assertEqual(moved.status, 200)
        self.assertEqual(moved.headers["Cache-Control"], "no-store")
        item = (await moved.json())["item"]
        self.assertEqual(item["path"], "diffusion_models/anima/anima.safetensors")
        self.assertEqual(item["kind"], "Diffusion")
        inventory = await (await self.client.get("/api/setting/resources", headers=self.headers)).json()
        self.assertEqual(inventory["models"], [item])
        self.assertFalse(model.exists())
        self.assertEqual((await self.client.post(url, headers=self.headers, json=body)).status, 404)

    async def test_move_validates_input_and_reports_conflicts_and_protection(self):
        url = "/api/setting/resources/" + "a" * 64 + "/move"
        for body in ([], None, {}, {"folder": []}, {"folder": "diffusion_models", "subfolder": 1}):
            response = await self.client.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(body))
            self.assertEqual(response.status, 400)
        malformed = await self.client.post(url, headers={**self.headers, "Content-Type": "application/json"}, data="{")
        self.assertEqual(malformed.status, 400)
        unsupported = await self.client.post(url, headers=self.headers, data="folder=diffusion_models")
        self.assertEqual(unsupported.status, 415)
        for error, status in ((FileExistsError(), 409), (ResourceMoveBlockedError("protected"), 403), (PermissionError(), 500)):
            with patch("settings_resources.ResourceLibrary.move", side_effect=error):
                response = await self.client.post(url, headers=self.headers, json={"folder": "diffusion_models"})
            self.assertEqual(response.status, status)
            self.assertIn("error", await response.json())

    async def test_move_rejects_active_download_or_installation(self):
        downloads, installer = Mock(), Mock()
        downloads.status.return_value = {"state": "running"}
        installer.status.return_value = {"state": "idle"}
        client = TestClient(TestServer(create_app(self.dist, self.source, downloads=downloads, installer=installer)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        with patch("settings_resources.ResourceLibrary.move") as move:
            url = "/api/setting/resources/" + "a" * 64 + "/move"
            response = await client.post(url, headers=self.headers, json={"folder": "diffusion_models"})
            self.assertEqual(response.status, 409)
            downloads.status.return_value = {"state": "idle"}
            installer.status.return_value = {"state": "running"}
            response = await client.post(url, headers=self.headers, json={"folder": "diffusion_models"})
            self.assertEqual(response.status, 409)
            move.assert_not_called()

    async def test_delete_requires_header_and_same_origin_then_updates_inventory(self):
        model = data_root(self.source) / "comfy" / "models" / "loras" / "local.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fixture")
        inventory = await (await self.client.get("/api/setting/resources", headers=self.headers)).json()
        url = "/api/setting/resources/" + inventory["loras"][0]["id"]
        for headers in ({"Host": self.headers["Host"]}, {**self.headers, "Origin": "https://untrusted.example"}):
            denied = await self.client.delete(url, headers=headers)
            self.assertEqual(denied.status, 403)
            self.assertTrue(model.exists())
        deleted = await self.client.delete(url, headers=self.headers)
        self.assertEqual(deleted.status, 200)
        self.assertFalse(model.exists())
        self.assertEqual(await (await self.client.get("/api/setting/resources", headers=self.headers)).json(), {"models": [], "loras": []})
        self.assertEqual((await self.client.delete(url, headers=self.headers)).status, 404)

    async def test_delete_rejects_active_installation_and_reports_file_errors(self):
        downloads, installer = Mock(), Mock()
        downloads.status.return_value = {"state": "running"}
        installer.status.return_value = {"state": "idle"}
        client = TestClient(TestServer(create_app(self.dist, self.source, downloads=downloads, installer=installer)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        url = "/api/setting/resources/" + "a" * 64
        with patch("settings_resources.ResourceLibrary.delete") as delete:
            blocked = await client.delete(url, headers=self.headers)
            self.assertEqual(blocked.status, 409)
            self.assertIn("다운로드", (await blocked.json())["error"])
            downloads.status.return_value = {"state": "idle"}
            installer.status.return_value = {"state": "running"}
            blocked = await client.delete(url, headers=self.headers)
            self.assertEqual(blocked.status, 409)
            self.assertIn("초기 설치", (await blocked.json())["error"])
            delete.assert_not_called()
            installer.status.return_value = {"state": "idle"}
            delete.side_effect = PermissionError("busy")
            failed = await client.delete(url, headers=self.headers)
            self.assertEqual(failed.status, 500)
            self.assertIn("파일 사용 여부", (await failed.json())["error"])

    async def test_delete_api_rejects_manifest_models_and_loras_even_with_forged_annotation(self):
        models = data_root(self.source) / "comfy" / "models"
        paths = ["checkpoints/managed.pt", "loras/managed.pt"]
        for name in paths:
            path = models / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"protected fixture")
            path.with_suffix(".png").write_bytes(b"preview fixture")
        manifest = Mock(models=[{"relative_path": "models/" + path} for path in paths])
        with patch("settings_resources.load_install_manifest", return_value=manifest):
            inventory = await (await self.client.get("/api/setting/resources", headers=self.headers)).json()
            for item in inventory["models"] + inventory["loras"]:
                blocked = await self.client.delete(
                    "/api/setting/resources/" + item["id"], headers=self.headers,
                    json={"deleteBlockedReason": "", "name": "renamed", "path": "loras/optional.pt"},
                )
                self.assertEqual(blocked.status, 403)
                self.assertEqual((await blocked.json())["error"], item["deleteBlockedReason"])
                self.assertEqual((models / item["path"]).read_bytes(), b"protected fixture")
                self.assertEqual((models / item["path"]).with_suffix(".png").read_bytes(), b"preview fixture")
        with patch("settings_resources.load_install_manifest", side_effect=ManifestError("unavailable")):
            blocked = await self.client.delete("/api/setting/resources/" + inventory["loras"][0]["id"], headers=self.headers)
            self.assertEqual(blocked.status, 403)
            self.assertIn("명세를 확인할 수 없어", (await blocked.json())["error"])

    async def test_download_api_only_resolves_or_starts_on_explicit_post(self):
        downloads = Mock()
        downloads.status.return_value = {"state": "idle", "items": [], "completed": 0, "total": 0}
        downloads.resolve.return_value = {"groups": []}
        client = TestClient(TestServer(create_app(self.dist, self.source, downloads=downloads)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        denied = await client.get("/api/setting/resources/download", headers={"Host": "127.0.0.1:8201"})
        self.assertEqual(denied.status, 403)
        status = await client.get("/api/setting/resources/download", headers=self.headers)
        self.assertEqual((await status.json())["state"], "idle")
        downloads.resolve.assert_not_called()
        downloads.start.assert_not_called()
        resolved = await client.post("/api/setting/resources/download/resolve", headers=self.headers, json={"urls": ["https://huggingface.co/user/repo"], "kind": "models"})
        self.assertEqual(resolved.status, 200)
        downloads.resolve.assert_called_once_with(["https://huggingface.co/user/repo"], "models", {}, None)
        downloads.start.side_effect = ValueError("선택할 파일이 없습니다.")
        failed = await client.post("/api/setting/resources/download", headers=self.headers, json={"files": ["unknown"], "kind": "models"})
        self.assertEqual(failed.status, 400)

    async def test_download_api_reports_missing_comfyui_and_rejects_start_and_resume(self):
        response = await self.client.get("/api/setting/resources/download", headers=self.headers)
        self.assertIn("초기 설치", (await response.json())["installBlockers"][0])
        for action, body in (("", {"files": ["unknown"], "kind": "models"}), ("/resume", {})):
            failed = await self.client.post("/api/setting/resources/download" + action, headers=self.headers, json=body)
            self.assertEqual(failed.status, 400)
            self.assertIn("초기 설치", (await failed.json())["error"])
        self.assertFalse((data_root(self.source) / "comfy").exists())
