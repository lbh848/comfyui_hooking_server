"""Style presets preserve both columns and reject stale or invalid writes."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from asset_style_presets import AssetStylePresets, StylePresetConflict, register_style_preset_runtime
from settings_assets import allowed_asset_path, register_assets, register_asset_runtime


def settings_fixture():
    return {
        "defaults": {"asset_model_preset_id": "model-a", "asset_workflow_type": "anima_only",
                     "background_preset": "garden", "raw_background_tags": ["garden"],
                     "anima_artist_preset": "水彩", "raw_anima_artist_tags": ["watercolor"], "img_w": 704,
                     "style_lora_enabled": False, "style_lora_list": []},
        "values": {"asset_model_preset_id": "model-b", "asset_workflow_type": "anima_ilxl",
                   "background_preset": "garden", "raw_background_tags": ["night garden", "moonlight"],
                   "anima_artist_preset": "水彩", "raw_anima_artist_tags": ["watercolor", "soft light"],
                   "img_w": 832, "img_h": 1216, "seed": -1, "hrf_anima": True, "hrf_size": 1.5,
                   "style_lora_enabled": True, "style_lora_list": [{"name": "style.safetensors", "strength": 0.65}]},
    }


class AssetStylePresetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Path(directory.name) / "settings.json"
        self.config.write_text('{"unchanged": true}', encoding="utf-8")
        # Store unit tests start empty, independently of shipped style presets.
        self.builtin = self.config.with_name("builtin_styles.json")
        self.store = AssetStylePresets(self.config, self.builtin)

    def change(self, action, **fields):
        return self.store.change({"action": action, "revision": self.store.catalog()["revision"], **fields})

    def create(self, name="수채화"):
        return self.change("create", name=name, settings=settings_fixture())

    def test_roundtrip_preserves_default_and_applied_values_and_config(self):
        self.assertEqual(self.store.catalog()["presets"], [])
        self.assertFalse(self.store.path.exists())
        result = self.create(" 수채화 ")
        saved = AssetStylePresets(self.config, self.builtin).catalog()["presets"][0]
        self.assertEqual(saved["name"], "수채화")
        self.assertEqual(saved["id"], result["saved_id"])
        self.assertEqual(saved["settings"], settings_fixture())
        self.assertEqual(self.config.read_text(), '{"unchanged": true}')

    def test_copy_update_rename_delete_are_independent(self):
        first = self.create()["saved_id"]
        second = self.create("수채화 복사")["saved_id"]
        changed = settings_fixture()
        changed["values"]["img_w"] = 1024
        self.change("save", id=first, settings=changed)
        self.change("rename", id=first, name="밝은 수채화")
        presets = self.store.catalog()["presets"]
        self.assertEqual([item["name"] for item in presets], ["밝은 수채화", "수채화 복사"])
        self.assertEqual(presets[1]["settings"], settings_fixture())
        self.change("delete", id=first, confirmed=True)
        self.assertEqual(self.store.catalog()["presets"][0]["id"], second)
        self.change("delete", id=second, confirmed=True)
        self.assertEqual(self.store.catalog()["presets"], [])

    def test_stale_client_cannot_save_rename_or_delete(self):
        initial = self.create()
        self.create("다른 그림체")
        before = self.store.path.read_bytes()
        for action in ("save", "rename", "delete", "create"):
            with self.subTest(action=action), self.assertRaises(StylePresetConflict):
                self.store.change({"action": action, "revision": initial["revision"], "id": initial["saved_id"],
                                   "name": "stale", "settings": settings_fixture(), "confirmed": True})
        self.assertEqual(before, self.store.path.read_bytes())

    def test_invalid_names_missing_ids_and_unconfirmed_deletion_do_not_write(self):
        identity = self.create("Watercolor")["saved_id"]
        before = self.store.path.read_bytes()
        for body in (
            {"action": "create", "name": " watercolor ", "settings": settings_fixture()},
            {"action": "rename", "id": identity, "name": " "},
            {"action": "rename", "id": identity, "name": "x" * 81},
            {"action": "delete", "id": identity},
            {"action": "save", "id": "missing", "settings": settings_fixture()},
            {"action": "unknown"},
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.change(**body)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_invalid_style_fields_do_not_overwrite_saved_settings(self):
        identity = self.create()["saved_id"]
        before = self.store.path.read_bytes()
        for key, value in (("character", "must not be stored"), ("snapref_images", []), ("raw_anima_artist_tags", "bad"),
                           ("background_preset", []), ("raw_background_tags", "bad"),
                           ("style_lora_list", ["bad"]), ("img_w", 0), ("seed", 1.5), ("hrf_size", float("nan")),
                           ("hrf_anima", "true"), ("asset_workflow_type", "unsupported")):
            settings = settings_fixture()
            settings["values"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.change("save", id=identity, settings=settings)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_corrupt_file_and_failed_writes_preserve_data(self):
        self.create()
        before = self.store.path.read_bytes()
        with patch("asset_style_presets.write_json_atomic", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.create("new")
        self.assertEqual(before, self.store.path.read_bytes())
        self.store.path.write_text("broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.change({"action": "create", "name": "new", "settings": settings_fixture(), "revision": "stale"})
        self.assertEqual(self.store.path.read_text(), "broken")


class AssetStylePresetApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.app = web.Application()
        register_style_preset_runtime(SimpleNamespace(app=self.app, CONFIG_FILE=Path(directory.name) / "settings.json"))
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_http_roundtrip_validation_and_conflict(self):
        response = await self.client.get("/api/asset/style-presets")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        initial = await response.json()
        body = {"action": "create", "name": "테스트", "settings": settings_fixture(), "revision": initial["revision"]}
        created = await self.client.post("/api/asset/style-presets", json=body)
        self.assertEqual(created.status, 200)
        saved = await created.json()
        reread = await (await self.client.get("/api/asset/style-presets")).json()
        self.assertEqual(reread["presets"], saved["presets"])
        stale = await self.client.post("/api/asset/style-presets", json=body)
        self.assertEqual(stale.status, 409)
        self.assertIn("error", await stale.json())
        invalid = await self.client.post("/api/asset/style-presets", json=[])
        self.assertEqual(invalid.status, 400)

    async def test_settings_gateway_forwards_allowed_path_and_requires_schema(self):
        upstream = self.client

        class Runtime:
            async def request(self, request, *, asset):
                self_asset = asset
                path = "/api/" + request.match_info["path"]
                if not self_asset or not allowed_asset_path(path, request.method):
                    raise web.HTTPForbidden()
                response = await upstream.request(request.method, path, data=await request.read(), headers={"Content-Type": "application/json"})
                return web.Response(body=await response.read(), status=response.status, content_type="application/json")

        gateway = web.Application()
        register_assets(gateway, Runtime(), "2")
        client = TestClient(TestServer(gateway))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        path = "/api/asset/asset/style-presets"
        self.assertEqual((await client.get(path)).status, 403)
        self.assertEqual((await client.get(path, headers={"X-Comfy-Settings": "1"})).status, 409)
        headers = {"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
        catalog = await (await client.get(path, headers=headers)).json()
        response = await client.post(path, headers=headers, json={"action": "create", "name": "via gateway",
                                    "settings": settings_fixture(), "revision": catalog["revision"]})
        self.assertEqual(response.status, 200)
        self.assertTrue(allowed_asset_path("/api/asset/style-presets", "POST"))
        self.assertFalse(allowed_asset_path("/api/asset/style-presets", "DELETE"))


class AssetStyleGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Path(directory.name) / "settings.json"
        self.store = AssetStylePresets(self.config)
        saved = self.store.change({"action": "create", "name": "그림체", "settings": settings_fixture(),
                                   "revision": self.store.catalog()["revision"]})
        self.identity = saved["saved_id"]
        self.queue = AsyncMock(return_value=[SimpleNamespace(batch_id="batch")])
        self.app = web.Application()
        register_asset_runtime(SimpleNamespace(app=self.app, CONFIG_FILE=self.config, app_config={},
                                               queue_manager=SimpleNamespace(add_items_batch=self.queue)))
        snapshot = {"config": {"asset_workflow_type": "anima_only"},
                    "workflows": {"anima_only": {"fields": {"STYLE_LORA_DATA": {"list": []}}}}}
        mock = patch("asset_model_presets.AssetModelPresets.snapshot", return_value=snapshot)
        self.snapshot = mock.start()
        self.addCleanup(mock.stop)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def job(self, **fields):
        return {"asset_style_preset_id": self.identity, "character": "character", "appearance": "appearance",
                "outfit": "outfit", "expression": "expression", "asset_workflow_type": "anima_only",
                "prompt_json": {"schema": "soya.illustration.v2", "fields": {"ANIMA_CONTENT": "edited style tags", "SEED": 123}}, **fields}

    async def test_missing_unknown_and_deleted_styles_cannot_preview_or_generate(self):
        self.store.change({"action": "delete", "id": self.identity, "confirmed": True, "revision": self.store.catalog()["revision"]})
        for identity in (None, "", "missing", self.identity, 1):
            job = self.job(asset_style_preset_id=identity)
            if identity is None:
                job.pop("asset_style_preset_id")
            for path, body in (("preview", job), ("enqueue", {"jobs": [job]})):
                with self.subTest(identity=identity, path=path):
                    response = await self.client.post(f"/api/asset/{path}", json=body)
                    self.assertEqual(response.status, 400)
                    self.assertIn("그림체 프리셋", (await response.json())["error"])
        self.snapshot.assert_not_called()
        self.queue.assert_not_awaited()

    async def test_selected_style_allows_preview_and_keeps_edited_applied_values(self):
        response = await self.client.post("/api/asset/preview", json=self.job())
        self.assertEqual(response.status, 200)
        preview = await response.json()
        self.assertEqual(preview["asset_style_preset_id"], self.identity)
        self.assertEqual(preview["prompt_json"]["fields"]["ANIMA_CONTENT"], "edited style tags")
        self.queue.assert_not_awaited()

    async def test_batch_rejects_all_jobs_if_one_slot_has_no_style(self):
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job(), self.job(asset_style_preset_id="")]})
        self.assertEqual(response.status, 400)
        self.assertIn("슬롯 2", (await response.json())["error"])
        self.queue.assert_not_awaited()

    async def test_selected_style_reaches_queue(self):
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job()]})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["count"], 1)
        self.assertEqual(self.queue.await_args.args[0][0]["params"]["body"]["asset_style_preset_id"], self.identity)

    async def test_character_applied_values_need_no_individual_preset_names(self):
        for names in ({}, {"appearance": "", "outfit": "", "expression": ""},
                      {"appearance": None, "outfit": "  ", "expression": None}):
            job = self.job()
            for key in ("appearance", "outfit", "expression"):
                job.pop(key)
            job.update(names)
            for path, body in (("preview", job), ("enqueue", {"jobs": [job]})):
                with self.subTest(names=names, path=path):
                    response = await self.client.post(f"/api/asset/{path}", json=body)
                    self.assertEqual(response.status, 200, await response.text())
                    result = await response.json() if path == "preview" else self.queue.await_args.args[0][0]["params"]["body"]
                    self.assertEqual([result[key] for key in ("appearance", "outfit", "expression")], ["기본"] * 3)
                    self.assertEqual(result["prompt_json"]["fields"]["ANIMA_CONTENT"], "edited style tags")

    async def test_character_is_still_required_and_batch_validation_is_atomic(self):
        for character in (None, "", " ", 1):
            with self.subTest(character=character):
                response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job(), self.job(character=character)]})
                self.assertEqual(response.status, 400)
                self.assertIn("슬롯 2: 캐릭터를 선택하세요.", (await response.json())["error"])
        self.queue.assert_not_awaited()

    async def test_existing_named_presets_keep_their_output_location(self):
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job()]})
        self.assertEqual(response.status, 200)
        job = self.queue.await_args.args[0][0]["params"]["body"]
        self.assertEqual([job[key] for key in ("appearance", "outfit", "expression")], ["appearance", "outfit", "expression"])


if __name__ == "__main__":
    unittest.main()
