"""Download sources and resume behavior, using only isolated mock responses."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import sys
from threading import Event
import unittest

import httpx
from PIL import Image

from app_paths import data_root
from comfy_installer.downloader import DownloadCancelled, DownloadError, ResumableDownloader
from comfy_installer.operations import uv_python_path
from settings_downloads import ResourceDownloadService, public_url
from settings_resources import ResourceLibrary


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "installation" / "src"
        self.source.mkdir(parents=True)
        self.comfy_root = data_root(self.source) / "comfy"
        (self.comfy_root / "comfy").mkdir(parents=True)
        (self.comfy_root / "main.py").write_text("# isolated ComfyUI fixture")
        (self.comfy_root / "folder_paths.py").write_text("# isolated ComfyUI fixture")
        python = uv_python_path(self.comfy_root / ".venv")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"isolated Python fixture")
        receipt = self.comfy_root / ".installer-state/runtime-receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"schema_version": 1}')
        self.payload = b"isolated model download fixture"
        image = BytesIO()
        Image.new("RGB", (2, 2), "blue").save(image, format="PNG")
        self.preview = image.getvalue()
        self.civitai_images = []
        self.hf_preview = False
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.requests = []
        self.transport = httpx.MockTransport(self.respond)
        self.factory = lambda: httpx.Client(transport=self.transport, follow_redirects=True)
        self.downloader = ResumableDownloader(client_factory=self.factory, max_retries=1)
        self.service = ResourceDownloadService(self.source, client_factory=self.factory, downloader=self.downloader)
        self.keys = {"other-civitai": "test-civitai-key", "other-huggingface": "test-hf-key"}

    def version(self, identity):
        return {"id": identity, "name": f"version-{identity}", "baseModel": "SDXL", "trainedWords": ["trigger"], "model": {"name": "Fixture", "type": "Checkpoint"}, "images": self.civitai_images,
                "files": [{"id": identity, "name": f"civitai-{identity}.safetensors", "type": "Model", "sizeKB": 0.03, "hashes": {"SHA256": self.digest}, "downloadUrl": f"https://civitai.com/api/download/models/{identity}"}]}

    def respond(self, request):
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "media.example":
            return httpx.Response(200, content=b"not an image" if path == "/bad.png" else self.preview)
        if request.url.host == "huggingface.co" and path.endswith(".preview.png"):
            return httpx.Response(200, content=self.preview)
        if request.url.host == "civitai.com" and path == "/api/v1/models/11":
            return httpx.Response(200, json={"name": "Fixture", "type": "Checkpoint", "modelVersions": [self.version(101), self.version(102)]})
        if request.url.host == "civitai.com" and path.startswith("/api/v1/model-versions/"):
            return httpx.Response(200, json=self.version(int(path.rsplit("/", 1)[1])))
        if request.url.host == "huggingface.co" and path.startswith("/api/models/"):
            return httpx.Response(200, json=([{"type": "file", "path": "diffusion_models/hf.preview.png", "size": len(self.preview)}] if self.hf_preview else []) + [{"type": "file", "path": "diffusion_models/hf.safetensors", "size": len(self.payload), "lfs": {"oid": self.digest, "size": len(self.payload)}},
                                             {"type": "file", "path": "small.pt", "size": len(self.payload)}, {"type": "file", "path": "README.md", "size": 10}])
        if request.url.host == "civarchive.com" and path == "/api/models/33":
            return httpx.Response(200, json={"data": {"name": "Archive fixture", "type": "LORA", "version": {"id": 303, "name": "v1", "base_model": "SDXL", "trigger": ["archive trigger"], "files": [
                {"id": 3, "name": "archive.safetensors", "sha256": self.digest, "sizeKB": 0.03, "mirrors": [{"url": "https://mirror.example/archive.safetensors", "filename": "archive.safetensors"}]}]}}})
        if "/api/download/models/" in path or "/resolve/" in path or request.url.host == "mirror.example":
            range_header = request.headers.get("Range")
            if range_header:
                start, end = range_header.removeprefix("bytes=").split("-")
                start = int(start)
                end = int(end) if end else len(self.payload) - 1
                return httpx.Response(206, content=self.payload[start:end + 1], headers={"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"})
            return httpx.Response(200, content=self.payload)
        raise AssertionError(f"Unexpected mock URL: {request.url}")

    def resolve(self, value, kind="models", **kwargs):
        result = self.service.resolve([value], kind, self.keys, **kwargs)["groups"][0]
        self.assertNotIn("error", result)
        return result

    def install(self, files, kind="models", folder="auto", subfolder=""):
        self.service.start([file["id"] for file in files], kind, folder, subfolder, self.keys)
        self.service._worker.join(3)
        self.assertFalse(self.service._worker.is_alive())
        return self.service.status()

    def test_idle_and_scan_do_not_access_external_services(self):
        self.assertEqual(self.service.status()["state"], "idle")
        self.assertEqual(ResourceLibrary(self.source).scan(), {"models": [], "loras": []})
        self.assertEqual(self.requests, [])

    def test_civitai_versions_explicit_file_and_shared_key(self):
        group = self.resolve("https://civitai.com/models/11/name")
        self.assertEqual([version["id"] for version in group["versions"]], ["101", "102"])
        selected = self.resolve("https://civitai.com/models/11/name", version_id="102")
        status = self.install(selected["files"], subfolder="illustration")
        self.assertEqual(status["state"], "completed")
        target = data_root(self.source) / "comfy/models/checkpoints/illustration/civitai-102.safetensors"
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(json.loads(target.with_suffix(".metadata.json").read_text(encoding="utf-8"))["sha256"], self.digest)
        self.assertTrue(all(request.headers.get("Authorization") == "Bearer test-civitai-key" for request in self.requests))
        self.assertNotIn("test-civitai-key", json.dumps(status))

    def test_huggingface_repo_and_direct_file(self):
        group = self.resolve("https://huggingface.co/user/repo")
        self.assertEqual(len(group["files"]), 2)
        direct = self.resolve("https://huggingface.co/user/repo/blob/main/diffusion_models/hf.safetensors")
        self.assertEqual(len(direct["files"]), 1)
        self.assertEqual(self.install(direct["files"])["state"], "completed")
        self.assertTrue(all(request.headers.get("Authorization") == "Bearer test-hf-key" for request in self.requests))

    def test_huggingface_without_published_sha256_uses_size_and_computes_hash(self):
        group = self.resolve("user/repo")
        file = next(item for item in group["files"] if item["name"] == "small.pt")
        self.assertEqual(self.install([file], folder="checkpoints")["state"], "completed")
        metadata = data_root(self.source) / "comfy/models/checkpoints/small.metadata.json"
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["sha256"], self.digest)

    def test_civarchive_mirror_never_receives_provider_keys(self):
        group = self.resolve("https://civarchive.com/models/33", "loras")
        self.assertEqual(self.install(group["files"], kind="loras")["state"], "completed")
        self.assertTrue(all("Authorization" not in request.headers for request in self.requests))
        self.assertEqual(ResourceLibrary(self.source).scan()["loras"][0]["trigger"], "archive trigger")

    def test_multiple_sources_can_be_selected_in_one_queue(self):
        groups = self.service.resolve(["https://civitai.com/models/11", "https://huggingface.co/user/repo/resolve/main/diffusion_models/hf.safetensors"], "models", self.keys)["groups"]
        files = [group["files"][0] for group in groups]
        status = self.install(files)
        self.assertEqual((status["state"], status["completed"], status["total"]), ("completed", 2, 2))
        self.assertEqual(len(ResourceLibrary(self.source).scan()["models"]), 2)

    def test_existing_different_file_is_preserved(self):
        group = self.resolve("101")
        target = data_root(self.source) / "comfy/models/checkpoints/civitai-101.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing user content")
        status = self.install(group["files"])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(target.read_bytes(), b"existing user content")

    def test_pause_and_resume_keep_partial_file_and_use_range(self):
        group = self.resolve("101")
        started = Event()
        real_downloader = self.downloader
        prefix = self.payload[:8]

        class PausingDownloader:
            first = True
            def download(inner, **kwargs):
                if inner.first:
                    inner.first = False
                    target = kwargs["target"]
                    target.parent.mkdir(parents=True)
                    target.with_name(target.name + ".part").write_bytes(prefix)
                    started.set()
                    self.assertTrue(kwargs["cancel_event"].wait(2))
                    raise DownloadCancelled()
                return real_downloader.download(**kwargs)

        self.service.downloader = PausingDownloader()
        self.service.start([group["files"][0]["id"]], "models", "auto", "", self.keys)
        self.assertTrue(started.wait(2))
        self.service.stop(pause=True)
        self.service._worker.join(3)
        self.assertEqual(self.service.status()["state"], "paused")
        self.service.resume(self.keys)
        self.service._worker.join(3)
        self.assertEqual(self.service.status()["state"], "completed")
        self.assertTrue(any(request.headers.get("Range") == "bytes=8-" for request in self.requests))

    def test_invalid_urls_paths_and_wrong_library_are_rejected(self):
        self.assertIn("error", self.service.resolve(["https://unknown.example/model.pt"], "models", self.keys)["groups"][0])
        self.assertIn("error", self.service.resolve(["https://civarchive.com/models/33"], "models", self.keys)["groups"][0])
        group = self.resolve("101")
        for subfolder in ("../outside", "E:/outside", "folder/../../outside", "folder\\outside"):
            with self.assertRaises(ValueError):
                self.service.start([group["files"][0]["id"]], "models", "auto", subfolder, self.keys)
        with self.assertRaises(ValueError):
            public_url("https://127.0.0.1/model.pt")
        self.assertFalse(self.service.root.exists())

    def test_pinned_downloads_still_require_a_valid_expected_hash(self):
        with self.assertRaises(DownloadError):
            self.downloader.download(url="https://mirror.example/model.safetensors", target=self.source / "unused", expected_size=10, expected_sha256="")

    def test_download_target_cannot_follow_an_external_junction(self):
        group = self.resolve("101")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.service.root.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(outside), str(self.service.root))
        else:
            self.service.root.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.service.start([group["files"][0]["id"]], "models", "auto", "", self.keys)
        self.assertEqual(list(outside.iterdir()), [])

    def test_models_directory_alone_does_not_allow_install_or_resume(self):
        group = self.resolve("101")
        (self.comfy_root / "main.py").unlink()
        self.service.root.mkdir()
        self.assertIn("ComfyUI", self.service.status()["installBlockers"][0])
        with self.assertRaisesRegex(ValueError, "초기 설치"):
            self.service.start([group["files"][0]["id"]], "models", "auto", "", self.keys)
        with self.assertRaisesRegex(ValueError, "초기 설치"):
            self.service.resume(self.keys)
        self.assertEqual(list(self.service.root.iterdir()), [])

    def test_civitai_key_is_required_before_lookup_start_and_resume(self):
        result = self.service.resolve(["101"], "models", {})["groups"][0]
        self.assertIn("Civitai API 키", result["error"])
        self.assertEqual(self.requests, [])
        group = self.resolve("101")
        with self.assertRaisesRegex(ValueError, "Civitai API 키"):
            self.service.start([group["files"][0]["id"]], "models", "auto", "", {})
        self.assertFalse(self.service.root.exists())
        self.service._state = "paused"
        self.service._jobs = [{"source": "civitai", "state": "paused"}]
        with self.assertRaisesRegex(ValueError, "Civitai API 키"):
            self.service.resume({})

    def test_incomplete_comfyui_installation_without_a_receipt_is_blocked(self):
        (self.comfy_root / ".installer-state/runtime-receipt.json").unlink()
        self.assertIn("초기 설치", self.service.status()["installBlockers"][0])

    def test_public_huggingface_does_not_require_a_civitai_key(self):
        group = self.service.resolve(["user/repo"], "models", {})["groups"][0]
        self.service.start([group["files"][0]["id"]], "models", "auto", "", {})
        self.service._worker.join(3)
        self.assertEqual(self.service.status()["state"], "completed")
        self.assertTrue(all("Authorization" not in request.headers for request in self.requests))

    def test_civitai_thumbnail_is_saved_and_exposed_without_leaking_keys(self):
        self.civitai_images = [{"type": "video", "url": "https://media.example/video.mp4"},
                              {"url": "https://media.example/bad.png"}, {"url": "https://media.example/preview.png"}]
        group = self.resolve("101")
        self.assertEqual(self.install(group["files"])["state"], "completed")
        preview = self.service.root / "checkpoints/civitai-101.preview.png"
        self.assertEqual(preview.read_bytes(), self.preview)
        library = ResourceLibrary(self.source)
        model = library.scan()["models"][0]
        self.assertEqual(library.thumbnail(model["id"]), preview)
        media_requests = [request for request in self.requests if request.url.host == "media.example"]
        self.assertEqual(len(media_requests), 2)
        self.assertTrue(all("Authorization" not in request.headers for request in media_requests))

    def test_huggingface_direct_file_finds_its_sibling_preview(self):
        self.hf_preview = True
        group = self.resolve("https://huggingface.co/user/repo/blob/main/diffusion_models/hf.safetensors")
        self.assertEqual(len(group["files"]), 1)
        self.assertEqual(self.install(group["files"])["state"], "completed")
        self.assertEqual((self.service.root / "diffusion_models/hf.preview.png").read_bytes(), self.preview)

    def test_thumbnail_failure_does_not_turn_a_saved_model_into_a_failed_download(self):
        self.civitai_images = [{"url": "https://media.example/bad.png"}]
        group = self.resolve("101")
        status = self.install(group["files"])
        self.assertEqual(status["state"], "completed")
        self.assertIn("썸네일", status["items"][0]["warning"])
        self.assertEqual((self.service.root / "checkpoints/civitai-101.safetensors").read_bytes(), self.payload)

    def test_existing_thumbnail_is_preserved(self):
        self.civitai_images = [{"url": "https://media.example/preview.png"}]
        group = self.resolve("101")
        preview = self.service.root / "checkpoints/civitai-101.jpg"
        preview.parent.mkdir(parents=True)
        preview.write_bytes(b"user preview")
        self.assertEqual(self.install(group["files"])["state"], "completed")
        self.assertEqual(preview.read_bytes(), b"user preview")
        self.assertFalse(any(request.url.host == "media.example" for request in self.requests))
