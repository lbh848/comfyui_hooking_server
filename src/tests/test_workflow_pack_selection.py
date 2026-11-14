"""Workflow packs use fixed dependencies without owning the model catalog."""
import hashlib
import json
from pathlib import Path
import tempfile
import stat
import unittest
from unittest.mock import patch

from comfy_installer.crypto import create_workflow_pack, extract_workflow_pack
from comfy_installer.manifest import default_install_model_ids, load_install_manifest
from comfy_installer.pack_cli import build_workflow_items
from comfy_installer.service import ComfyInstallerService, InstallerServiceError
from comfy_installer.workflow_library import WorkflowLibraryError, install_managed_workflows, library_status, release_install_manifest, selection_requirements
from settings_workflows import workflow_catalog


class WorkflowPackSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "src"
        self.source.mkdir()
        self.catalog_path = self.source / "install_manifest.json"
        self.data = load_install_manifest().data
        self.data["models"] = [
            self.model("selected-model", "diffusion_models/selected.safetensors"),
            self.model("selected-style", "loras/style.safetensors"),
            self.model("fixed-vae", "vae/fixed.safetensors"),
        ]
        self.data["validation_profiles"] = {}
        self.data["workflows"] = {
            "release_version": "v6", "excluded_filenames": [],
            "required_bindings": ["fixture"], "optional_bindings": [],
            "items": [{"id": "fixture", "bindings": ["fixture"], "model_ids": ["fixed-vae"]}],
        }
        self.save_catalog()
        self.workflow = self.root / "workflow.json"
        self.workflow_data = {
            "prompt": {"class_type": "PrimitiveString", "inputs": {"value": json.dumps({
                "schema": "soya.illustration.v2", "fields": {
                    "ANIMA_DIFFUSION_MODEL": "selected.safetensors",
                    "STYLE_LORA_DATA": {"LORAS": [{"lora_path": "style.safetensors", "str": 0.7}]},
                },
            })}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "fixed.safetensors"}},
        }
        self.workflow.write_text(json.dumps(self.workflow_data), encoding="utf-8")
        self.bindings = {"fixture": self.workflow}
        self.pack = self.root / "workflows.soyawfp"
        self.library = self.root / "data" / "comfy_workflow_library"
        self.catalog_loader = patch(
            "comfy_installer.workflow_library.load_install_manifest",
            side_effect=lambda: load_install_manifest(self.catalog_path),
        )
        self.catalog_loader.start()
        self.addCleanup(self.catalog_loader.stop)

    @staticmethod
    def model(identity, path):
        return {"id": identity, "relative_path": "models/" + path,
                "url": "https://example.com/" + path, "size": 128, "sha256": "a" * 64}

    def save_catalog(self):
        self.catalog_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def create_pack(self):
        manifest = load_install_manifest(self.catalog_path)
        create_workflow_pack(self.bindings, self.pack, "fixture-key", release_version="v6",
                             workflow_items=build_workflow_items(self.bindings, manifest))

    def import_pack(self):
        service = ComfyInstallerService(project_root=self.source,
                                       manifest=load_install_manifest(self.catalog_path))
        with patch.object(service, "workflow_library_status", return_value={}):
            service.unpack_workflow_pack(workflow_pack=self.pack, workflow_key="fixture-key")

    def test_application_model_and_lora_defaults_are_not_fixed_dependencies(self):
        items = build_workflow_items(self.bindings, load_install_manifest(self.catalog_path))
        self.assertEqual(items[0]["model_ids"], ["fixed-vae"])

    def write_preset(self, catalog="illustration_model", *, fields=None, workflow=None):
        asset = catalog == "asset_model"
        prefix, profile = ("asset", "anima_only") if asset else ("illustration", "v3_anima")
        path = self.source / "presets" / (catalog + "_presets.json")
        path.parent.mkdir(exist_ok=True)
        preset = {"id": "distribution", "name": "Distribution", "config": {
            prefix + "_workflow_type": profile,
            prefix + "_workflow_source_paths": {profile: workflow or str(self.workflow)},
        }, "workflows": {profile: {"fields": fields if fields is not None else {
            "ANIMA_DIFFUSION_MODEL": "selected.safetensors",
            "STYLE_LORA_DATA": {"list": [{"lora_path": "style.safetensors", "str": .7}]},
        }}}}
        path.write_text(json.dumps({"version": 1, "presets": [preset]}), encoding="utf-8")
        return path

    def test_all_distribution_catalogs_add_live_requirements_without_changing_pack(self):
        self.create_pack()
        self.import_pack()
        before = self.pack.read_bytes()
        state_path = self.library / "SOYA_DISTRIBUTION/v6/.soya-pack.json"
        state_before = state_path.read_bytes()
        for catalog in ("illustration_model", "emotion_model", "asset_model"):
            with self.subTest(catalog=catalog):
                path = self.write_preset(catalog)
                selected = selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
                self.assertEqual(selected["model_ids"], ["fixed-vae", "selected-model", "selected-style"])
                self.assertEqual(selected["model_bytes"], 384)
                self.write_preset(catalog, fields={"STYLE_LORA_DATA": {"list": [
                    {"lora_path": "unregistered.safetensors", "enabled": False}]}})
                selected = selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
                self.assertEqual(selected["model_ids"], ["fixed-vae"])
                path.unlink()
        self.assertEqual(self.pack.read_bytes(), before)
        self.assertEqual(state_path.read_bytes(), state_before)

    def test_initial_install_workflow_install_and_readiness_share_preset_requirements(self):
        self.write_preset()
        self.create_pack()
        self.import_pack()
        service = ComfyInstallerService(project_root=self.source, manifest=load_install_manifest(self.catalog_path))
        manifest = release_install_manifest(library_root=self.library, release_version="v6")
        selected = selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
        by_id = {model["id"]: model for model in manifest.models}
        with patch.object(service, "_read_config", return_value={}):
            scope = service._scope_selected_models([by_id[i] for i in selected["model_ids"]], manifest=manifest)
        self.assertEqual({model["id"] for model in scope.keep}, {"fixed-vae", "selected-model", "selected-style"})
        # The fixed workflow model exists; only preset dependencies are missing.
        for model in manifest.models:
            path = service.comfy_root / model["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if model["id"] == "fixed-vae":
                path.write_bytes(b"x" * model["size"])
        (service.comfy_root / "main.py").write_text("", encoding="utf-8")
        with patch("comfy_installer.service.uv_python_path", return_value=self.workflow), patch.object(service, "_start_operation", return_value={}) as start:
            service.start_workflow_models(release_version="v6", selected_item_ids=["fixture"])
        self.assertEqual({model["id"] for model in start.call_args.kwargs["kwargs"]["models"]}, {"selected-model", "selected-style"})
        with patch.object(service, "e2e_workflow_catalog", return_value={"release_version": "v6", "items": []}):
            item = workflow_catalog(service)["items"][0]
            self.assertFalse(item["modelsInstalled"])
            self.assertEqual(item["missingModelCount"], 2)
            for model in manifest.models:
                (service.comfy_root / model["relative_path"]).write_bytes(b"x" * model["size"])
            item = workflow_catalog(service)["items"][0]
            self.assertTrue(item["modelsInstalled"])

    def test_user_inactive_and_unrelated_presets_do_not_add_downloads(self):
        path = self.write_preset(fields={})
        document = json.loads(path.read_bytes())
        document["presets"][0]["workflows"]["v3"] = {"fields": {"ILXL_CHECKPOINT": "unused.safetensors"}}
        path.write_text(json.dumps(document), encoding="utf-8")
        config = self.root / "config"
        config.mkdir()
        (config / path.name).write_bytes(path.read_bytes())
        self.write_preset("emotion_model", workflow="unrelated.json")
        self.create_pack()
        self.import_pack()
        selected = selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
        self.assertEqual(selected["model_ids"], ["fixed-vae"])

    def test_unknown_preset_model_keeps_workflow_visible_but_blocks_incomplete_install(self):
        path = self.write_preset(fields={"STYLE_LORA_DATA": {"list": [{"lora_path": "missing.safetensors"}]}})
        self.create_pack()
        self.import_pack()
        service = ComfyInstallerService(project_root=self.source, manifest=load_install_manifest(self.catalog_path))
        with self.assertRaisesRegex(WorkflowLibraryError, "missing.safetensors"):
            selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
        with patch.object(service, "e2e_workflow_catalog", return_value={"release_version": "v6", "items": []}):
            item = workflow_catalog(service)["items"][0]
            self.assertFalse(item["modelsInstalled"])
            self.assertIn("missing.safetensors", item["installBlockedReason"])
        self.write_preset(fields={})
        self.assertEqual(selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])["model_ids"], ["fixed-vae"])

    def installed_cleanup_fixture(self):
        other = self.root / "other.json"
        other.write_text(json.dumps({"vae": self.workflow_data["vae"]}), encoding="utf-8")
        self.bindings["other"] = other
        self.data["workflows"]["required_bindings"].append("other")
        self.data["workflows"]["items"].append({"id": "other", "bindings": ["other"], "model_ids": ["fixed-vae"]})
        self.save_catalog()
        self.write_preset()
        self.create_pack()
        self.import_pack()
        service = ComfyInstallerService(project_root=self.source, manifest=load_install_manifest(self.catalog_path))
        models = {}
        for model in self.data["models"]:
            path = service.comfy_root / model["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * model["size"])
            models[model["id"]] = path
        installed = install_managed_workflows(comfy_root=service.comfy_root, library_root=self.library,
                                              release_version="v6", selected_item_ids=["fixture", "other"])
        workflows = {Path(path): Path(path).read_bytes() for path in installed.workflow_files}
        return service, models, workflows

    def test_cleanup_removes_exclusive_models_and_loras_but_keeps_workflows_and_shared_models(self):
        service, models, workflows = self.installed_cleanup_fixture()
        selection = {"release_version": "v6", "selected_item_ids": ["fixture"]}
        plan = service.cleanup_workflow_models(**selection)
        self.assertEqual({item["id"] for item in plan["removable"]}, {"selected-model", "selected-style"})
        self.assertEqual({item["id"] for item in plan["retained"]}, {"fixed-vae"})
        self.assertEqual(plan["removableBytes"], 256)
        result = service.cleanup_workflow_models(**selection, revision=plan["revision"])
        self.assertEqual(result["deletedCount"], 2)
        self.assertFalse(result["failed"])
        self.assertFalse(models["selected-model"].exists())
        self.assertFalse(models["selected-style"].exists())
        self.assertTrue(models["fixed-vae"].exists())
        for store in (service, ComfyInstallerService(project_root=self.source, manifest=load_install_manifest(self.catalog_path))):
            self.assertEqual(len(store.workflow_library_status()["workflow_files"]), 2)
            for path, payload in workflows.items():
                self.assertEqual(path.read_bytes(), payload)
                self.assertFalse(path.stat().st_mode & stat.S_IWUSR)
        with patch.object(service, "e2e_workflow_catalog", return_value={"release_version": "v6", "items": []}):
            items = {item["id"]: item for item in workflow_catalog(service)["items"]}
        self.assertTrue(items["other"]["modelsInstalled"])
        self.assertFalse(items["fixture"]["modelsInstalled"])
        self.assertTrue(items["fixture"]["workflowInstalled"])
        self.assertEqual(items["fixture"]["cleanupBlockedReason"], "")

    def test_cleanup_protects_runtime_json_and_other_workflows_user_preset_references(self):
        service, models, _ = self.installed_cleanup_fixture()
        local = service.comfy_root / "user/default/workflows/personal.json"
        local.write_text(json.dumps({"prompt": {"class_type": "PrimitiveString", "inputs": {"value": json.dumps({
            "schema": "soya.illustration.v2", "fields": {"ANIMA_DIFFUSION_MODEL": "selected.safetensors"}})}}}), encoding="utf-8")
        path = self.write_preset("emotion_model", workflow=str(self.root / "other.json"), fields={
            "STYLE_LORA_DATA": {"list": [{"lora_path": "style.safetensors", "str": 1}]}})
        user = self.root / "config" / path.name
        user.parent.mkdir(exist_ok=True)
        user.write_bytes(path.read_bytes())
        path.unlink()  # User-only catalogs must still protect their consumers.
        plan = service.cleanup_workflow_models(release_version="v6", selected_item_ids=["fixture"])
        self.assertEqual(plan["removable"], [])
        self.assertEqual({item["id"] for item in plan["retained"]}, set(models))
        result = service.cleanup_workflow_models(release_version="v6", selected_item_ids=["fixture"], revision=plan["revision"])
        self.assertEqual(result["deletedCount"], 0)
        self.assertTrue(all(path.exists() for path in models.values()))

    def test_cleanup_rechecks_new_shared_references_before_deleting(self):
        service, models, _ = self.installed_cleanup_fixture()
        selection = {"release_version": "v6", "selected_item_ids": ["fixture"]}
        plan = service.cleanup_workflow_models(**selection)
        self.write_preset("emotion_model", workflow=str(self.root / "other.json"))
        with self.assertRaises(InstallerServiceError):
            service.cleanup_workflow_models(**selection, revision=plan["revision"])
        self.assertTrue(all(path.exists() for path in models.values()))

    def test_explicit_loader_dependency_is_kept_even_for_an_editable_model(self):
        self.workflow_data["fixed-model"] = {
            "class_type": "UNETLoader", "inputs": {"unet_name": "selected.safetensors"},
        }
        self.workflow.write_text(json.dumps(self.workflow_data), encoding="utf-8")
        items = build_workflow_items(self.bindings, load_install_manifest(self.catalog_path))
        self.assertEqual(items[0]["model_ids"], ["fixed-vae", "selected-model"])

    def test_pack_import_preserves_project_catalog(self):
        self.create_pack()
        self.data["models"].append(self.model("new-default-style", "loras/new.safetensors"))
        self.save_catalog()
        before = self.catalog_path.read_bytes()
        self.import_pack()
        self.assertEqual(self.catalog_path.read_bytes(), before)
        extracted = extract_workflow_pack(self.pack, self.root / "extracted", "fixture-key")
        self.assertIsNone(extracted.install_manifest)
        self.assertEqual(extracted.workflow_items[0]["model_ids"], ["fixed-vae"])
        state = json.loads((self.library / "SOYA_DISTRIBUTION/v6/.soya-pack.json").read_text(encoding="utf-8"))
        self.assertNotIn("install_manifest", state)

    def test_catalog_updates_apply_without_rebuilding_or_importing_pack(self):
        self.create_pack()
        self.import_pack()
        before_hash = hashlib.sha256(self.pack.read_bytes()).hexdigest()
        self.data["models"].append(self.model("new-default-style", "loras/new.safetensors"))
        self.data["models"][2].update(size=1024, url="https://example.com/new-vae", sha256="b" * 64)
        self.data["workflows"]["items"][0]["model_ids"] = ["selected-model"]
        self.save_catalog()
        manifest = release_install_manifest(library_root=self.library, release_version="v6")
        self.assertIn("new-default-style", default_install_model_ids(manifest.models))
        self.assertEqual(manifest.models[2]["url"], "https://example.com/new-vae")
        self.assertEqual(manifest.workflow_items[0]["model_ids"], ["fixed-vae"])
        requirements = selection_requirements(library_root=self.library, release_version="v6", selected_item_ids=["fixture"])
        self.assertEqual(requirements["model_ids"], ["fixed-vae"])
        self.assertEqual(requirements["model_bytes"], 1024)
        self.assertEqual(hashlib.sha256(self.pack.read_bytes()).hexdigest(), before_hash)

    def test_refreshed_pack_and_installed_files_match_and_missing_locks_are_repaired(self):
        self.create_pack()
        self.import_pack()
        comfy = self.root / "data/comfy"
        def install():
            return install_managed_workflows(comfy_root=comfy, library_root=self.library,
                                            release_version="v6", selected_item_ids=["fixture"], allow_missing_models=True)
        installed = Path(install().workflow_files[0])
        self.assertEqual(installed.read_bytes(), self.workflow.read_bytes())
        self.workflow_data["vae"]["_meta"] = {"title": "Updated VAE"}
        self.workflow.write_text(json.dumps(self.workflow_data), encoding="utf-8")
        self.create_pack()
        self.import_pack()
        install()
        self.assertEqual(installed.read_bytes(), self.workflow.read_bytes())
        self.assertFalse(installed.stat().st_mode & stat.S_IWUSR)
        user = installed.with_name("user.json")
        user.write_text('{"nodes":[]}', encoding="utf-8")
        renamed = installed.with_name("renamed-pack.json")
        renamed.write_bytes(installed.read_bytes())
        installed.chmod(installed.stat().st_mode | stat.S_IWUSR)
        state = library_status(comfy, self.library)
        self.assertFalse(state["workflow_files_error"])
        for path in (installed, renamed):
            self.assertFalse(path.stat().st_mode & stat.S_IWUSR)
        self.assertTrue(user.stat().st_mode & stat.S_IWUSR)
        self.assertEqual(json.loads(user.read_bytes()), {"nodes": []})
