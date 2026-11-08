"""Workflow packs use fixed dependencies without owning the model catalog."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from comfy_installer.crypto import create_workflow_pack, extract_workflow_pack
from comfy_installer.manifest import default_install_model_ids, load_install_manifest
from comfy_installer.pack_cli import build_workflow_items
from comfy_installer.service import ComfyInstallerService
from comfy_installer.workflow_library import release_install_manifest, selection_requirements


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
