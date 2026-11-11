"""Workflow management uses selected distribution items and isolated group storage."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from comfy_installer.service import InstallerServiceError
from comfy_installer.workflow_library import WorkflowLibraryError, configured_workflow_integrity, library_status
from settings_server import create_app
from settings_store import SettingsConflict
from settings_workflows import WorkflowGroups, default_workflow_groups, delete_local_workflow, workflow_catalog


def installer_fixture():
    service = Mock()
    service.status.return_value = {"state": "idle", "operation": None, "logs": []}
    service.workflow_library_status.return_value = {"releases": [{"release_version": "v6", "items": [
        {"id": "anima", "name": "삽화 생성", "filename": "anima.json", "bindings": ["illustration_workflow_source_paths.v3_anima"], "model_count": 2},
        {"id": "utility", "name": "유틸리티", "filename": "utility.json", "bindings": ["utility_workflow_source_path"]},
    ]}]}
    service.e2e_workflow_catalog.return_value = {"release_version": "v6", "items": [{"id": "anima", "e2e_profiles": ["illustration"]}], "skipped": [{"id": "utility", "reason": "검사 방법 미등록"}]}
    service.workflow_integrity_report.return_value = {"release_version": "v6", "items": [], "counts": {}}
    service.start_e2e.return_value = {"state": "running", "operation": "e2e", "selected_item_ids": ["anima"]}
    service.cancel.return_value = {"state": "running", "operation": "e2e", "cancel_requested": True}
    return service


class WorkflowGroupsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = WorkflowGroups(Path(self.temporary.name) / "src")

    def test_groups_survive_reload_and_stale_writes_cannot_overwrite(self):
        value, revision = self.store.read()
        self.assertEqual(value, {"groups": []})
        self.assertFalse(self.store.path.exists())
        payload = {"groups": [{"id": "g1", "name": " 삽화 ", "itemIds": ["anima", "anima"]}]}
        saved, updated = self.store.save(payload, revision)
        self.assertEqual(saved["groups"][0], {"id": "g1", "category": "사용자 그룹", "name": "삽화", "itemIds": ["anima"]})
        self.assertEqual(WorkflowGroups(Path(self.temporary.name) / "src").read(), (saved, updated))
        with self.assertRaises(SettingsConflict):
            self.store.save({"groups": []}, revision)
        self.assertEqual(self.store.read(), (saved, updated))
        self.assertEqual(self.store.save({"groups": []}, updated)[0], {"groups": []})

    def test_invalid_or_corrupt_groups_are_not_overwritten(self):
        self.store.path.parent.mkdir()
        self.store.path.write_bytes(b"broken")
        with self.assertRaises(ValueError):
            self.store.save({"groups": []}, '"empty"')
        self.assertEqual(self.store.path.read_bytes(), b"broken")
        for groups in ([{"id": "a", "name": "", "itemIds": ["x"]}], [{"id": "a", "name": "A", "itemIds": []}], [
            {"id": "a", "name": "same", "itemIds": ["x"]}, {"id": "b", "name": "SAME", "itemIds": ["y"]},
        ]):
            with self.assertRaises(ValueError):
                WorkflowGroups.validate({"groups": groups})

    def test_catalog_preserves_non_e2e_items_and_reports_blockers(self):
        service = installer_fixture()
        catalog = workflow_catalog(service)
        self.assertEqual([item["id"] for item in catalog["items"]], ["anima", "utility"])
        self.assertEqual(catalog["items"][0]["e2eBlockedReason"], "")
        self.assertEqual(catalog["items"][1]["e2eBlockedReason"], "검사 방법 미등록")
        service.e2e_workflow_catalog.side_effect = InstallerServiceError("설치 필요")
        self.assertTrue(all(item["e2eBlockedReason"] == "설치 필요" for item in workflow_catalog(service)["items"]))
        service.workflow_library_status.return_value = {"releases": []}
        self.assertEqual(workflow_catalog(service)["items"], [])

    def test_two_level_defaults_use_bindings_and_group_sibling_variants(self):
        items = [
            {"id": "v1", "name": "renamed", "bindings": ["illustration_workflow_source_paths.v1"]},
            {"id": "v3", "bindings": ["illustration_workflow_source_paths.v3"]},
            {"id": "anima", "bindings": ["comfy_workflow_source_path", "illustration_workflow_source_paths.v3_anima"]},
            {"id": "fast", "bindings": ["workflow_pack_extra_paths.illustration_anima_fast"]},
            {"id": "asset", "bindings": ["anima_only_asset_workflow_source_path"]},
            {"id": "new", "name": "V3 illustration", "bindings": ["new_binding"]},
        ]
        groups = default_workflow_groups(items)
        pairs = {(group["category"], group["name"]): group["itemIds"] for group in groups}
        self.assertEqual(pairs[("삽화", "V1 계열")], ["v1"])
        self.assertEqual(pairs[("삽화", "ANIMA+ILXL")], ["v3"])
        self.assertEqual(pairs[("삽화", "ANIMA")], ["anima", "fast"])
        self.assertEqual(pairs[("에셋", "ANIMA")], ["asset"])
        self.assertEqual(pairs[("기타", "미분류")], ["new"])
        self.assertEqual(default_workflow_groups(items), groups)
        self.assertEqual(WorkflowGroups.validate({"groups": groups})["groups"], groups)

    def test_training_groups_combine_character_and_style_by_model(self):
        items = [
            {"id": "character-anima", "bindings": ["lora_training_workflow_source_paths.anima"]},
            {"id": "character-ilxl", "bindings": ["lora_training_workflow_source_paths.sdxl"]},
            {"id": "style-anima", "bindings": ["style_lora_training_workflow_source_paths.anima"]},
            {"id": "style-ilxl", "bindings": ["style_lora_training_workflow_source_paths.sdxl"]},
        ]
        groups = default_workflow_groups(items)
        self.assertEqual({(group["category"], group["name"]): group["itemIds"] for group in groups}, {
            ("LoRA 학습", "ANIMA"): ["character-anima", "style-anima"],
            ("LoRA 학습", "ILXL"): ["character-ilxl", "style-ilxl"],
        })

    def test_face_extraction_belongs_to_common_not_lora_training(self):
        groups = default_workflow_groups([
            {"id": "face", "bindings": ["face_extract_workflow_source_path"]},
            {"id": "character", "bindings": ["lora_training_workflow_source_paths.anima"]},
        ])
        by_category = {group["category"]: group["itemIds"] for group in groups}
        self.assertEqual(by_category, {"공통": ["face"], "LoRA 학습": ["character"]})


class SelectedIntegrityTests(unittest.TestCase):
    def test_only_selected_files_are_checked_without_modifying_any_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "SOYA_DISTRIBUTION" / "v1"
            release.mkdir(parents=True)
            original = b'{"nodes":[],"links":[]}'
            (release / "one.json").write_bytes(original)
            user = root / "user.json"
            user.write_bytes(original)
            items = [
                {"id": "one", "filename": "one.json", "sha256": hashlib.sha256(original).hexdigest(), "bindings": ["one"]},
                {"id": "two", "filename": "missing.json", "sha256": "0" * 64, "bindings": ["two"]},
            ]
            (release / ".soya-pack.json").write_text(json.dumps({"release_version": "v1", "items": items}))
            config = root / "config.json"
            config.write_text(json.dumps({"one": str(user)}))
            kwargs = {"config_path": config, "library_root": root, "release_version": "v1"}
            report = configured_workflow_integrity(**kwargs, selected_item_ids=["one"])
            self.assertEqual([item["id"] for item in report["items"]], ["one"])
            self.assertEqual(report["counts"]["clean"], 1)
            self.assertEqual(user.read_bytes(), original)
            user.write_bytes(b"changed")
            report = configured_workflow_integrity(**kwargs, selected_item_ids=["one"])
            self.assertEqual(report["counts"]["needs_repair"], 1)
            self.assertEqual(user.read_bytes(), b"changed")
            self.assertEqual(configured_workflow_integrity(**kwargs)["counts"]["distribution_error"], 1)
            for selected in ([], ["unknown"]):
                with self.assertRaises(WorkflowLibraryError):
                    configured_workflow_integrity(**kwargs, selected_item_ids=selected)


class CombinedWorkflowCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.comfy = self.root / "data/comfy"
        self.folder = self.comfy / "user/default/workflows"
        self.folder.mkdir(parents=True)
        self.library = self.root / "library"
        self.service = installer_fixture()
        self.service.workflow_library_status.side_effect = lambda: library_status(self.comfy, self.library)

    def write(self, name, value):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_pack_and_recursive_folder_union_deduplicates_content_not_names(self):
        original = self.write("old-pack-name.json", {"nodes": [], "links": []})
        self.write("new-pack-name.json", {"nodes": [{"id": 1}], "links": []})
        self.write("personal/a/same.json", {"nodes": [{"id": 2}]})
        self.write("personal/b/same.json", {"nodes": [{"id": 3}]})
        self.write("personal/api.JSON", {"1": {"class_type": "SaveImage", "inputs": {}}})
        self.write("not-workflow.json", {"settings": True})
        self.write(".hidden/workflow.json", {"nodes": []})
        self.write(".soya-pack.json", {"nodes": []})
        self.write("broken.json", {}).write_text("broken", encoding="utf-8")
        release = self.library / "SOYA_DISTRIBUTION/v6"
        release.mkdir(parents=True)
        (release / ".soya-pack.json").write_text(json.dumps({"release_version": "v6", "items": [{
            "id": "anima", "filename": "new-pack-name.json", "bindings": ["illustration_workflow_source_paths.v3_anima"],
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        }]}), encoding="utf-8")
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        catalog = workflow_catalog(self.service)
        self.assertEqual(len(catalog["items"]), 5)
        packed, *local = catalog["items"]
        self.assertEqual(packed["localPaths"], ["old-pack-name.json"])
        self.assertEqual(packed["source"], "pack")
        self.assertEqual({item["filename"] for item in local}, {"new-pack-name.json", "personal/a/same.json", "personal/b/same.json", "personal/api.JSON"})
        self.assertEqual(len({item["id"] for item in local}), 4)
        self.assertTrue(all(item["source"] == "local" and item["e2eBlockedReason"] and item["integrityBlockedReason"] for item in local))
        self.assertEqual(next(group for group in catalog["defaultGroups"] if group["category"] == "미분류")["itemIds"], [item["id"] for item in local])
        self.assertEqual(catalog["folderError"], "")
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_local_items_work_without_pack_and_keep_ids_when_edited(self):
        path = self.write("personal.json", {"nodes": []})
        catalog = workflow_catalog(self.service)
        self.assertEqual(catalog["releaseVersion"], "")
        self.service.e2e_workflow_catalog.assert_not_called()
        identity = catalog["items"][0]["id"]
        store = WorkflowGroups(self.root / "src")
        payload = {"groups": [{"id": "personal", "category": "미분류", "name": "직접 만든 작업", "itemIds": [identity]}]}
        store.save(payload, store.read()[1])
        path.write_text('{"nodes":[{"id":1}]}', encoding="utf-8")
        self.assertEqual(workflow_catalog(self.service)["items"][0]["id"], identity)
        self.assertEqual(store.read()[0], payload)
        path.unlink()
        self.assertEqual(workflow_catalog(self.service)["items"], [])

    def test_unreadable_file_does_not_hide_other_workflows(self):
        blocked = self.write("blocked.json", {"nodes": []})
        self.write("readable.json", {"nodes": []})
        read_bytes = Path.read_bytes
        def guarded(path):
            if path == blocked:
                raise PermissionError("fixture")
            return read_bytes(path)
        with patch.object(Path, "read_bytes", guarded):
            catalog = workflow_catalog(self.service)
        self.assertEqual([item["filename"] for item in catalog["items"]], ["readable.json"])
        self.assertTrue(catalog["folderError"])

    def test_resolved_external_paths_are_excluded(self):
        outside = self.root / "outside.json"
        outside.write_text('{"nodes":[]}', encoding="utf-8")
        alias = self.write("escape.json", {"nodes": []})
        self.write("inside.json", {"nodes": []})
        resolve = Path.resolve
        def resolved(path, *args, **kwargs):
            return outside if path == alias else resolve(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolved):
            self.assertEqual([item["filename"] for item in workflow_catalog(self.service)["items"]], ["inside.json"])

    def test_external_workflow_root_is_not_scanned(self):
        outside = self.root / "external-workflows"
        outside.mkdir()
        (outside / "external.json").write_text('{"nodes":[]}', encoding="utf-8")
        with patch("comfy_installer.workflow_library.embedded_workflow_base_dir", return_value=outside):
            catalog = workflow_catalog(self.service)
        self.assertEqual(catalog["items"], [])
        self.assertTrue(catalog["folderError"])


class WorkflowApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        self.dist = self.source / "settings_ui" / "dist"
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text("fixture")
        self.installer = installer_fixture()
        self.runtime = Mock()
        self.runtime.manager.is_running.return_value = False
        self.downloads = Mock()
        self.downloads.status.return_value = {"state": "idle"}
        self.client = TestClient(TestServer(create_app(self.dist, self.source, installer=self.installer, runtime=self.runtime, downloads=self.downloads)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
        self.selection = {"releaseVersion": "v6", "selectedItemIds": ["anima", "anima"]}

    async def test_local_and_schema_guards_and_no_implicit_execution(self):
        for headers in ({"Host": "127.0.0.1:8201"}, {**self.headers, "Origin": "https://example.com"}):
            response = await self.client.get("/api/setting/workflows", headers=headers)
            self.assertEqual(response.status, 403)
        response = await self.client.get("/api/setting/workflows", headers={**self.headers, "X-Comfy-Settings-Schema": "old"})
        self.assertEqual(response.status, 409)
        response = await self.client.get("/api/setting/workflows", headers=self.headers)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(len((await response.json())["items"]), 2)
        self.installer.start_e2e.assert_not_called()
        self.installer.workflow_integrity_report.assert_not_called()

    async def test_selected_integrity_and_e2e_are_forwarded_to_existing_engine(self):
        for action in ("integrity", "e2e"):
            response = await self.client.post("/api/setting/workflows/" + action, headers=self.headers, json=self.selection)
            self.assertEqual(response.status, 200, await response.text())
        self.installer.workflow_integrity_report.assert_called_once_with(release_version="v6", selected_item_ids=["anima"])
        self.installer.start_e2e.assert_called_once_with(release_version="v6", selected_item_ids=["anima"])

    async def test_local_or_mixed_selection_cannot_reach_pack_validation(self):
        for action in ("e2e", "integrity"):
            for selected in (["local:abc"], ["anima", "local:abc"]):
                response = await self.client.post("/api/setting/workflows/" + action, headers=self.headers, json={"releaseVersion": "v6", "selectedItemIds": selected})
                self.assertEqual(response.status, 400)
                self.assertIn("사용자 워크플로우", (await response.json())["error"])
        self.installer.start_e2e.assert_not_called()
        self.installer.workflow_integrity_report.assert_not_called()

    async def test_local_install_routes_to_local_model_installer(self):
        self.installer.start_local_workflow_models.return_value = {
            "state": "running", "operation": "workflow_models", "selected_item_ids": ["local:abc"],
        }
        response = await self.client.post("/api/setting/workflows/install", headers=self.headers,
                                          json={"releaseVersion": "v6", "selectedItemIds": ["local:abc"]})
        self.assertEqual(response.status, 200, await response.text())
        self.installer.start_local_workflow_models.assert_called_once_with(identity="local:abc", civitai_key="")
        self.installer.start_workflow_models.assert_not_called()

    async def test_empty_selection_errors_and_busy_guards(self):
        for body in ({}, [], {"releaseVersion": "v6", "selectedItemIds": []}, {"releaseVersion": "v6", "selectedItemIds": [42]}):
            response = await self.client.post("/api/setting/workflows/e2e", headers=self.headers, json=body)
            self.assertEqual(response.status, 400)
        self.runtime.manager.is_running.return_value = True
        response = await self.client.post("/api/setting/workflows/e2e", headers=self.headers, json=self.selection)
        self.assertEqual(response.status, 409)
        self.runtime.manager.is_running.return_value = False
        self.downloads.status.return_value = {"state": "running"}
        response = await self.client.post("/api/setting/workflows/e2e", headers=self.headers, json=self.selection)
        self.assertEqual(response.status, 409)
        self.downloads.status.return_value = {"state": "idle"}
        self.installer.status.return_value = {"state": "running", "operation": "install"}
        for action in ("integrity", "e2e", "cancel"):
            response = await self.client.post("/api/setting/workflows/" + action, headers=self.headers, json=self.selection)
            self.assertEqual(response.status, 409)
        self.installer.start_e2e.assert_not_called()
        self.installer.cancel.assert_not_called()

    async def test_cancel_and_status_use_shared_installer(self):
        self.installer.status.return_value = {"state": "running", "operation": "e2e", "logs": [{"seq": 1, "message": "검증"}]}
        status = await self.client.get("/api/setting/workflows/status", headers=self.headers)
        self.assertEqual((await status.json())["logs"][0]["message"], "검증")
        response = await self.client.post("/api/setting/workflows/cancel", headers=self.headers, json={})
        self.assertEqual(response.status, 200)
        self.installer.cancel.assert_called_once()
        self.installer.status.return_value = {"state": "idle", "operation": None}

    async def test_groups_roundtrip_conflict_and_validation(self):
        self.installer.workflow_library_status.return_value["workflow_files"] = [{
            "relative_path": "personal.json", "name": "personal.json", "size": 12, "sha256": "user-hash",
        }]
        catalog = workflow_catalog(self.installer)
        local = next(item for item in catalog["items"] if item["source"] == "local")
        url = "/api/setting/workflows/groups"
        first = await self.client.get(url, headers=self.headers)
        self.assertEqual(await first.json(), {"groups": [], "useDefaults": True})
        payload = {"groups": [group for group in catalog["defaultGroups"] if local["id"] not in group["itemIds"]] + [
            {"id": "g1", "category": "학습", "name": "개인 모델", "itemIds": [local["id"]]},
        ]}
        no_revision = await self.client.put(url, headers=self.headers, json=payload)
        self.assertEqual(no_revision.status, 428)
        headers = {**self.headers, "If-Match": first.headers["ETag"]}
        saved = await self.client.put(url, headers=headers, json=payload)
        self.assertEqual(saved.status, 200)
        self.assertEqual(await saved.json(), {**payload, "useDefaults": False})
        stale = await self.client.put(url, headers=headers, json={"groups": []})
        self.assertEqual(stale.status, 409)
        invalid = await self.client.put(url, headers={**headers, "If-Match": saved.headers["ETag"]}, json={"groups": [{"name": "bad"}]})
        self.assertEqual(invalid.status, 400)
        final = await self.client.get(url, headers=self.headers)
        self.assertEqual(await final.json(), {**payload, "useDefaults": False})

    async def test_pack_group_moves_removal_and_renaming_are_blocked(self):
        url = "/api/setting/workflows/groups"
        first = await self.client.get(url, headers=self.headers)
        defaults = workflow_catalog(self.installer)["defaultGroups"]
        headers = {**self.headers, "If-Match": first.headers["ETag"]}
        cases = [[], [group for group in defaults if "anima" not in group["itemIds"]]]
        for field, value in (("category", "에셋"), ("name", "다른 모델")):
            cases.append([{**group, field: value} if "anima" in group["itemIds"] else group for group in defaults])
        cases.append([*defaults, {"id": "extra", "category": "미분류", "name": "개인", "itemIds": ["anima"]}])
        for groups in cases:
            response = await self.client.put(url, headers=headers, json={"groups": groups})
            self.assertEqual(response.status, 403, await response.text())
            self.assertIn("배포 워크플로우", (await response.json())["error"])
        self.assertFalse(WorkflowGroups(self.source).path.exists())

    def local_workflow(self):
        self.installer.comfy_root = self.source / "comfy"
        folder = self.installer.comfy_root / "user/default/workflows/personal"
        folder.mkdir(parents=True)
        path = folder / "작업.JSON"
        path.write_text('{"nodes":[{"widgets_values":["personal.safetensors"]}]}', encoding="utf-8")
        self.library = self.source / "library"
        self.installer.workflow_library_status.side_effect = lambda: library_status(self.installer.comfy_root, self.library)
        item = workflow_catalog(self.installer)["items"][0]
        return path, {"id": item["id"], "revision": item["fileRevision"]}

    async def test_delete_user_json_without_pack_keeps_models_and_other_workflows(self):
        path, body = self.local_workflow()
        sibling = path.with_name("other.json")
        sibling.write_bytes(path.read_bytes())
        model = self.installer.comfy_root / "models/checkpoints/personal.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model fixture")
        # Running ComfyUI and model downloads do not prevent JSON-only deletion.
        self.runtime.manager.is_running.return_value = True
        self.downloads.status.return_value = {"state": "running"}
        response = await self.client.post("/api/setting/workflows/delete", headers=self.headers, json=body)
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(await response.json(), {"deleted": body["id"]})
        self.assertFalse(path.exists())
        self.assertTrue(sibling.is_file())
        self.assertEqual(model.read_bytes(), b"model fixture")
        catalog = await self.client.get("/api/setting/workflows", headers=self.headers)
        self.assertEqual([item["filename"] for item in (await catalog.json())["items"]], ["personal/other.json"])
        self.installer.cleanup_workflow_models.assert_not_called()
        self.installer.e2e_workflow_catalog.assert_not_called()
        repeated = await self.client.post("/api/setting/workflows/delete", headers=self.headers, json=body)
        self.assertEqual(repeated.status, 409)

    async def test_delete_requires_current_revision_and_respects_request_guards(self):
        path, body = self.local_workflow()
        url = "/api/setting/workflows/delete"
        for headers, status in (({"Host": "127.0.0.1:8201"}, 403),
                                ({**self.headers, "Origin": "https://example.com"}, 403),
                                ({**self.headers, "X-Comfy-Settings-Schema": "old"}, 409)):
            response = await self.client.post(url, headers=headers, json=body)
            self.assertEqual(response.status, status)
        for invalid in ({}, {"id": body["id"]}, {"id": "anima", "revision": body["revision"]},
                        {"id": "../../outside.json", "revision": body["revision"]}):
            response = await self.client.post(url, headers=self.headers, json=invalid)
            self.assertEqual(response.status, 400)
        self.installer.status.return_value = {"state": "running", "operation": "workflow_models"}
        response = await self.client.post(url, headers=self.headers, json=body)
        self.assertEqual(response.status, 409)
        self.installer.status.return_value = {"state": "idle"}
        path.write_text('{"nodes":[{"id":2}]}', encoding="utf-8")
        changed = path.read_bytes()
        response = await self.client.post(url, headers=self.headers, json=body)
        self.assertEqual(response.status, 409)
        self.assertIn("변경", (await response.json())["error"])
        self.assertEqual(path.read_bytes(), changed)

    async def test_delete_rejects_forged_local_id_for_pack_file(self):
        path, body = self.local_workflow()
        release = self.library / "SOYA_DISTRIBUTION/v6"
        release.mkdir(parents=True)
        (release / ".soya-pack.json").write_text(json.dumps({"release_version": "v6", "items": [{
            "id": "anima", "filename": path.name, "sha256": body["revision"], "bindings": [],
        }]}), encoding="utf-8")
        response = await self.client.post("/api/setting/workflows/delete", headers=self.headers, json=body)
        self.assertEqual(response.status, 409)
        self.assertIn("배포 워크플로우", (await response.json())["error"])
        self.assertTrue(path.is_file())

    async def test_delete_rechecks_contents_and_path_after_inventory(self):
        path, body = self.local_workflow()
        snapshot = self.installer.workflow_library_status()
        self.installer.workflow_library_status.side_effect = None
        self.installer.workflow_library_status.return_value = snapshot
        original = path.read_bytes()
        path.write_text('{"nodes":[]}', encoding="utf-8")
        with self.assertRaisesRegex(WorkflowLibraryError, "변경"):
            delete_local_workflow(self.installer, identity=body["id"], revision=body["revision"])
        path.write_bytes(original)
        outside = self.source / "outside.json"
        outside.write_bytes(original)
        resolve = Path.resolve
        def resolved(candidate, *args, **kwargs):
            return outside if candidate == path else resolve(candidate, *args, **kwargs)
        with patch.object(Path, "resolve", resolved), self.assertRaisesRegex(WorkflowLibraryError, "폴더 밖"):
            delete_local_workflow(self.installer, identity=body["id"], revision=body["revision"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(outside.read_bytes(), original)

