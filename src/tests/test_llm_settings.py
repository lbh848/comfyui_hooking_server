"""Regression coverage for migration, per-slot options and shared request limits."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from app_paths import SOURCE_DIR
from json_store import write_json_atomic
from llm_settings import import_llm_settings, RuntimeSettingsReader, migrate_document, runtime_config, UNCONFIGURED_SERVICE, SLOT_IDS, LLM_SP_SLOTS
from settings_store import SettingsStore, SettingsDocument, SettingsConflict, ValidationError


def document():
    return {"version": 1, "apiKeys": {}, "draft": {
        "version": 1, "install": {"target": "both", "mode": "lazy", "path": "", "saved": False},
        "models": [], "loras": [],
        "providers": [{"id": "openai", "name": "OpenAI", "endpoint": "", "model": "", "enabled": True}],
        "otherProviders": [{"id": "other-civitai", "name": "Civitai", "endpoint": "", "model": "", "enabled": True}],
        "runtime": {"sage": True, "loading": "auto"},
        "updates": {"channel": "stable", "backup": True, "workflow": True, "saved": False},
    }}


def load_engine():
    # Load the actual engine without constructing unrelated mode singletons.
    package = types.ModuleType("modes")
    package.__path__ = [str(SOURCE_DIR / "modes")]
    spec = importlib.util.spec_from_file_location("llm_settings_test_engine", SOURCE_DIR / "modes/llm_service.py")
    engine = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modes": package, spec.name: engine}):
        spec.loader.exec_module(engine)
    return engine


class LlmSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config"
        self.config.mkdir()

    def test_import_preserves_ten_slots_and_distinct_keys(self):
        original = {"llm_model": "model-one", "llm_service": "gemini", "llm_max_concurrency": 5,
                    "llm_model2": "model-two", "llm_service2": "gemini", "llm_max_concurrency2": 3,
                    "llm_model10": "model-ten", "llm_alias10": "Ten", "llm_service10": "gemini",
                    "llm_temperature10": 0.4, "llm_custom_body10": '{"top_p":0.9}'}
        write_json_atomic(self.config / "config.json", original)
        write_json_atomic(self.config / "key/llm_keys.json", {"llm_api_key": "test-shared", "llm_api_key2": "test-shared", "llm_api_key10": "test-distinct"})
        value = migrate_document(document(), self.config)
        SettingsDocument.model_validate(value)
        slots = value["draft"]["llms"]
        self.assertEqual({slot["id"] for slot in slots}, set(SLOT_IDS))
        self.assertEqual(slots[0]["providerId"], slots[1]["providerId"])
        self.assertNotEqual(slots[0]["providerId"], slots[9]["providerId"])
        projected = runtime_config(value)
        self.assertEqual(projected["llm_max_concurrency"], 3)
        self.assertEqual(projected["llm_alias10"], "Ten")
        self.assertEqual(projected["llm_temperature10"], 0.4)
        self.assertEqual(json.loads((self.config / "config.json").read_text()), original)

    def test_provider_key_url_and_limit_shared_but_slot_options_independent(self):
        value = migrate_document(document(), self.config)
        value["draft"]["llms"][9].update(model="ten", temperature=0.2)
        value["draft"]["llmProviders"][0].update(maxConcurrency=4, url="https://shared.example/v1/{model}")
        value["apiKeys"]["openai"] = "test-provider-key"
        projected = runtime_config(value)
        self.assertEqual(projected["llm_api_key"], projected["llm_api_key10"])
        self.assertEqual(projected["llm_max_concurrency10"], 4)
        self.assertEqual(projected["llm_temperature"], 1)
        self.assertEqual(projected["llm_temperature10"], 0.2)
        self.assertEqual(projected["llm_url"], "https://shared.example/v1/{model}")
        self.assertEqual(projected["llm_url10"], projected["llm_url"])
        value["draft"]["llmProviders"][0]["url"] = ""
        self.assertEqual(runtime_config(value)["llm_url10"], "")

    def test_save_reload_clear_key_and_conflict(self):
        store = SettingsStore(self.root)
        value = migrate_document(document(), self.config)
        revision = store.save(value, '"missing"')
        reader = RuntimeSettingsReader(self.config)
        self.assertEqual(reader.read()["llm_api_key"], "")
        value["apiKeys"]["openai"] = "test-key"
        value["draft"]["llmProviders"][0]["url"] = "https://updated.example/v1"
        revision2 = store.save(value, revision)
        self.assertEqual(reader.read()["llm_api_key10"], "test-key")
        self.assertEqual(reader.read()["llm_url10"], "https://updated.example/v1")
        value["apiKeys"]["openai"] = ""
        value["draft"]["llmProviders"][0]["url"] = ""
        with self.assertRaises(SettingsConflict):
            store.save(value, revision)
        store.save(value, revision2)
        self.assertEqual(reader.read()["llm_api_key10"], "")
        self.assertEqual(reader.read()["llm_url10"], "")
        self.assertEqual(store.read()[0], value)

    def test_invalid_slot_references_and_limits_rejected(self):
        value = migrate_document(document(), self.config)
        mutations = [lambda d: d["llms"][0].update(providerId="missing"),
                     lambda d: d["llms"][9].update(id="llm1"),
                     lambda d: d["llmProviders"][0].update(maxConcurrency=0),
                     lambda d: d["llmProviders"][0].update(maxConcurrency=21),
                     lambda d: d["llms"][0].update(stream_idle_timeout_seconds=5)]
        for mutate in mutations:
            invalid = copy.deepcopy(value)
            mutate(invalid["draft"])
            with self.assertRaises(ValidationError):
                SettingsDocument.model_validate(invalid)

    def test_empty_slots_do_not_lower_configured_provider_limit(self):
        write_json_atomic(self.config / "config.json", {"llm_model10": "local-model", "llm_max_concurrency10": 7})
        result = import_llm_settings(self.config)
        self.assertEqual(result["llmProviders"][0]["maxConcurrency"], 7)

    def test_vertex_credentials_are_imported_per_provider(self):
        write_json_atomic(self.config / "config.json", {"llm_service": "vertex", "llm_service10": "vertex"})
        account = {"project_id": "test-project", "client_email": "test@example.invalid", "private_key": "test-only"}
        write_json_atomic(self.config / "key/vertex.json", account)
        write_json_atomic(self.config / "key/vertex10.json", account)
        value = import_llm_settings(self.config)
        self.assertEqual(value["llms"][0]["providerId"], value["llms"][9]["providerId"])
        self.assertEqual(json.loads(value["apiKeys"]["vertex"]), account)

    def test_engine_gate_shares_capacity_across_slots_and_uses_provider_url(self):
        engine = load_engine()
        value = migrate_document(document(), self.config)
        value["draft"]["llms"][0].update(model="one")
        value["draft"]["llms"][9].update(model="ten")
        value["draft"]["llmProviders"][0]["url"] = "https://shared.example/v1/{model}"
        store = SettingsStore(self.root)
        revision = store.save(value, '"missing"')
        async def exercise():
            first = engine._request_gate("llm1")
            tenth = engine._request_gate("llm10")
            identity = await first.acquire()
            self.assertEqual(engine._llm_provider_identity("llm1"), engine._llm_provider_identity("llm10"))
            self.assertFalse(tenth.has_capacity_now())
            waiting = asyncio.create_task(tenth.acquire())
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
            value["draft"]["llmProviders"][0]["maxConcurrency"] = 2
            store.save(value, revision)
            second_identity = await asyncio.wait_for(waiting, 2.5)
            self.assertEqual(identity, second_identity)
            await first.release(identity)
            await tenth.release(second_identity)
            self.assertEqual(engine._slot_config_overrides("llm10")["llm_model"], "ten")
            self.assertEqual(engine._slot_config_overrides("llm10")["llm_url"], "https://shared.example/v1/{model}")
        with patch.object(engine, "_settings_reader", RuntimeSettingsReader(self.config)):
            asyncio.run(exercise())

    def test_delete_all_providers_preserves_slot_settings_and_does_not_reimport(self):
        store = SettingsStore(self.root)
        value = migrate_document(document(), self.config)
        value["draft"]["llms"][0].update(model="retained-model", temperature=0.4)
        value["draft"]["llmProviders"] = []
        value["apiKeys"] = {}
        for slot in value["draft"]["llms"]:
            slot["providerId"] = ""
        store.save(value, '"missing"')
        reloaded = store.read()[0]
        self.assertEqual(reloaded, value)
        self.assertEqual(reloaded["draft"]["llms"][0]["model"], "retained-model")
        projected = runtime_config(reloaded)
        self.assertEqual(projected["llm_service"], UNCONFIGURED_SERVICE)
        self.assertEqual(projected["llm_api_key"], "")

    def test_unassigned_provider_never_dispatches_an_api_request(self):
        engine = load_engine()
        self.assertEqual({s["id"] for s in engine.get_service_catalog()}, {"vertex", "gemini", "openai", "claude"})
        async def exercise():
            with patch.object(engine.httpx, "AsyncClient", side_effect=AssertionError("Unexpected API request")), patch.object(engine, "_llm_log"):
                result = await engine._dispatch_unlimited([], UNCONFIGURED_SERVICE, "retained-model")
                self.assertIn("프로바이더를 선택", result)
                events = [event async for event in engine._dispatch_stream_unlimited([], UNCONFIGURED_SERVICE, "retained-model")]
                self.assertEqual(events[0]["type"], "error")
                self.assertIn("프로바이더를 선택", events[0]["error"])
        asyncio.run(exercise())

    def test_service_owned_by_provider_and_changed_for_all_its_slots(self):
        value = migrate_document(document(), self.config)
        value["draft"]["llmProviders"][0]["service"] = "gemini"
        projected = runtime_config(value)
        self.assertEqual(projected["llm_service"], "gemini")
        self.assertEqual(projected["llm_service10"], "gemini")
        self.assertTrue(all("service" not in slot for slot in value["draft"]["llms"]))
        value["draft"]["llms"][9]["providerId"] = "claude"
        self.assertEqual(runtime_config(value)["llm_service10"], "claude")

    def test_existing_ollama_profile_keeps_key_and_slots_with_service_name(self):
        value = migrate_document(document(), self.config)
        for provider in value["draft"]["llmProviders"]:
            provider.pop("service")
            provider.pop("url")
        value["draft"]["llmProviders"][0]["name"] = "ollama.com"
        value["apiKeys"]["openai"] = "test-ollama-key"
        for slot in value["draft"]["llms"]:
            slot.update(service="openai", url="https://ollama.com/v1")
        migrated = migrate_document(value, self.config)
        SettingsDocument.model_validate(migrated)
        self.assertEqual(migrated["draft"]["llmProviders"][0]["name"], "Ollama")
        self.assertEqual(migrated["draft"]["llmProviders"][0]["service"], "openai")
        self.assertEqual(runtime_config(migrated)["llm_api_key10"], "test-ollama-key")
        self.assertEqual(runtime_config(migrated)["llm_url10"], "https://ollama.com/v1")
        self.assertEqual(migrate_document(migrated, self.config), migrated)

    def test_added_compat_services_return_to_openai_without_losing_urls_or_keys(self):
        defaults = {"ollama-cloud": "https://ollama.com", "ollama": "http://localhost:11434",
                    "openrouter": "https://openrouter.ai/api", "lmstudio": "http://localhost:1234"}
        for service, url in defaults.items():
            value = migrate_document(document(), self.config)
            value["draft"]["llmProviders"][0]["service"] = service
            value["draft"]["llmProviders"][0].pop("url")
            value["apiKeys"]["openai"] = "test-existing-key"
            value["draft"]["llms"][0]["url"] = "https://custom.example/v1"
            migrated = migrate_document(value, self.config)
            self.assertEqual(migrated["draft"]["llmProviders"][0]["service"], "openai")
            self.assertEqual(runtime_config(migrated)["llm_url"], "https://custom.example/v1")
            self.assertEqual(runtime_config(migrated)["llm_url10"], url)
            self.assertEqual(runtime_config(migrated)["llm_api_key10"], "test-existing-key")
            self.assertEqual(migrate_document(migrated, self.config), migrated)

    def test_legacy_slot_urls_split_profiles_losslessly_and_only_once(self):
        value = migrate_document(document(), self.config)
        for provider in value["draft"]["llmProviders"]:
            provider.pop("url")
        value["apiKeys"]["openai"] = "test-shared-key"
        urls = ["https://same.example/team-a", "https://same.example/team-b", ""]
        for index, slot in enumerate(value["draft"]["llms"]):
            slot["url"] = urls[index % 3]
        original = copy.deepcopy(value)
        migrated = migrate_document(value, self.config)
        SettingsDocument.model_validate(migrated)
        self.assertEqual(value, original)
        projected = runtime_config(migrated)
        for index, slot in enumerate(migrated["draft"]["llms"]):
            if slot["id"] in LLM_SP_SLOTS:
                self.assertEqual(slot["providerId"], "")
                continue
            suffix = "" if index == 0 else str(index + 1)
            self.assertEqual(projected[f"llm_url{suffix}"], urls[index % 3])
            self.assertEqual(projected[f"llm_api_key{suffix}"], "test-shared-key")
            self.assertNotIn("url", slot)
        self.assertEqual(len({slot["providerId"] for slot in migrated["draft"]["llms"] if slot["id"] not in LLM_SP_SLOTS}), 3)
        self.assertEqual(migrate_document(migrated, self.config), migrated)
        store = SettingsStore(self.root)
        store.save(original, '"missing"')
        self.assertEqual(store.read()[0], migrated)

    def test_import_keeps_full_url_paths_in_distinct_profiles(self):
        original = {"llm_url": "https://same.example/team-a", "llm_url2": "https://same.example/team-b"}
        write_json_atomic(self.config / "config.json", original)
        value = migrate_document(document(), self.config)
        self.assertNotEqual(value["draft"]["llms"][0]["providerId"], value["draft"]["llms"][1]["providerId"])
        projected = runtime_config(value)
        self.assertEqual(projected["llm_url"], original["llm_url"])
        self.assertEqual(projected["llm_url2"], original["llm_url2"])

    def test_mixed_legacy_services_are_split_without_losing_credentials(self):
        value = migrate_document(document(), self.config)
        for provider in value["draft"]["llmProviders"]:
            provider.pop("service")
        value["apiKeys"]["openai"] = "test-shared-key"
        for slot in value["draft"]["llms"]:
            slot["service"] = "openai"
        value["draft"]["llms"][9]["service"] = "gemini"
        migrated = migrate_document(value, self.config)
        SettingsDocument.model_validate(migrated)
        projected = runtime_config(migrated)
        self.assertEqual(projected["llm_service"], "openai")
        self.assertEqual(projected["llm_service10"], "gemini")
        self.assertEqual(projected["llm_api_key10"], "test-shared-key")


class SettingsUiCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from settings_server import create_app, SETTINGS_UI_SCHEMA
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dist = self.root / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
        self.store = SettingsStore(self.root)
        self.revision = self.store.save(document(), '"missing"')
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": SETTINGS_UI_SCHEMA}
        self.client = TestClient(TestServer(create_app(dist, self.root)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_stale_ui_gets_reload_guidance_and_cannot_overwrite_settings(self):
        original = self.store.path.read_bytes()
        for version in (None, "1"):
            headers = {key: value for key, value in self.headers.items() if key != "X-Comfy-Settings-Schema"}
            if version is not None:
                headers["X-Comfy-Settings-Schema"] = version
            response = await self.client.get("/api/setting/config", headers=headers)
            self.assertEqual(response.status, 409)
            self.assertIn("새로고침", (await response.json())["error"])
            response = await self.client.put("/api/setting/config", headers={**headers, "If-Match": self.revision}, json=document())
            self.assertEqual(response.status, 409)
            self.assertEqual(self.store.path.read_bytes(), original)

    async def test_current_ui_can_read_edit_and_reload_provider_url(self):
        response = await self.client.get("/api/setting/config", headers=self.headers)
        self.assertEqual(response.status, 200)
        value = (await response.json())["settings"]
        self.assertTrue(all("url" not in slot for slot in value["draft"]["llms"]))
        value["draft"]["llmProviders"][0]["url"] = "https://shared.example/v1"
        saved = await self.client.put("/api/setting/config", headers={**self.headers, "If-Match": response.headers["ETag"]}, json=value)
        self.assertEqual(saved.status, 200)
        reloaded = await self.client.get("/api/setting/config", headers=self.headers)
        self.assertEqual((await reloaded.json())["settings"], value)


if __name__ == "__main__":
    unittest.main()
