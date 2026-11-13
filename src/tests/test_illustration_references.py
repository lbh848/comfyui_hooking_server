"""CPU checks for the distribution illustration reference/LoRA contract."""
import ast
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from illustration_json import build_illustration_json

RESOURCES = SRC.parent / "data/comfy/custom_nodes/comfyui-soya-custom-nodes"
SOYA = SRC.parent / "data/comfy/custom_nodes/comfyui-soya-custom-nodes"
spec = importlib.util.spec_from_file_location("illustration_json_nodes", RESOURCES / "soya_illustration_json_v2.py")
json_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(json_nodes)


def definitions(source, namespace):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<stage functions>", "exec"), namespace)
    return namespace


class ReferenceTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.bot = self.root / "bot"
        self.ref = self.input / "soya_bot/Story/Alice"
        self.face = self.bot / "Story/Alice"
        self.ref.mkdir(parents=True)
        self.face.mkdir(parents=True)
        Image.new("RGB", (8, 8), "red").save(self.ref / "representation.png")
        Image.new("RGB", (8, 8), "green").save(self.ref / "sub_1.png")
        Image.new("RGB", (8, 8), "blue").save(self.face / "_face_image.webp", lossless=True)

    def payload(self, mode):
        source = {"schema": "soya.illustration.v2", "assets": {}, "fields": {
            "CHAR_LIST": ["Alice"],
            "FACE_ID_DIR": {"list": [{"CHAR": "Alice", "ipa_path": "soya_bot/Story/Alice/cache.ipadpt"}]},
            "SNAPREF": {"enabled": True, "strength_scale": "linear", "strength": 1, "secondary_reference": mode},
        }}
        return json.loads(build_illustration_json(json.dumps(source), "", input_dir=self.input,
                          bot_dir=self.bot, include_face_embeddings=False))

    def colors(self, payload, stage):
        return [tuple(round(float(x) * 255) for x in tensor[0, 0, 0])
                for tensor in json_nodes.stage_reference_images(payload, stage).get("Alice", [])]

    def test_modes_route_images_to_every_anima_stage(self):
        for mode in ("none", "face_only", "face", "representative", "stage"):
            payload = self.payload(mode)
            self.assertIn("face_asset", payload["fields"]["SNAPREF"]["references"][0])
            for stage in ("first", "hires", "hand", "face", "eye"):
                with self.subTest(mode=mode, stage=stage):
                    expected = [(255, 0, 0)]
                    if mode == "face_only":
                        expected = [(0, 0, 255)]
                    elif mode == "face":
                        expected += [(0, 0, 255)]
                    elif mode == "representative":
                        expected += [(0, 128, 0)]
                    elif mode == "stage" and stage in {"face", "eye"}:
                        expected = [(0, 0, 255)]
                    self.assertEqual(self.colors(payload, stage), expected)

    def test_missing_data_patch_uses_the_recorded_representative(self):
        (self.ref / "representation.png").unlink()
        (self.face / "_face_image_source.json").write_text(
            json.dumps({"source_filename": "Alice_source.png"}), encoding="utf-8")
        Image.new("RGB", (8, 8), (0, 255, 0)).save(self.face / "Alice_source.png")
        payload = self.payload("none")
        self.assertEqual([item["CHAR"] for item in payload["fields"]["SNAPREF"]["references"]], ["Alice"])
        self.assertEqual(self.colors(payload, "first"), [(0, 255, 0)])

    def test_bot_representative_is_read_without_data_patch(self):
        Image.new("RGB", (8, 8), (255, 0, 255)).save(self.face / "Alice_current.png")
        source = {"schema": "soya.illustration.v2", "assets": {}, "fields": {
            "CHAR_LIST": ["Alice"],
            "FACE_ID_DIR": {"list": [{"CHAR": "Alice", "ipa_path": "soya_bot/Story/Alice/cache.ipadpt",
                                      "snapref_images": ["Alice_current.png"]}]},
            "SNAPREF": {"enabled": True, "strength_scale": "linear", "strength": 1,
                        "secondary_reference": "none"},
        }}
        payload = json.loads(build_illustration_json(json.dumps(source), "", input_dir=None,
                          bot_dir=self.bot, include_face_embeddings=False))
        self.assertEqual(self.colors(payload, "first"), [(255, 0, 255)])

    def test_missing_secondary_uses_primary_for_all_stages(self):
        (self.ref / "sub_1.png").unlink()
        (self.face / "_face_image.webp").unlink()
        for mode in ("face_only", "face", "representative", "stage"):
            payload = self.payload(mode)
            for stage in ("first", "hires", "hand", "face", "eye"):
                self.assertEqual(self.colors(payload, stage), [(255, 0, 0)])

    def test_selected_card_embeds_both_originals_and_its_face_without_input(self):
        card = self.face / "_visual_profiles/card_2"
        card.mkdir(parents=True)
        Image.new("RGB", (8, 8), "yellow").save(card / "_face_image.webp", lossless=True)
        Image.new("RGB", (8, 8), "red").save(self.face / "first.png")
        Image.new("RGB", (8, 8), "green").save(self.face / "second.png")
        source = {"schema": "soya.illustration.v2", "fields": {
            "CHAR_LIST": ["Alice"], "SNAPREF": {"enabled": True, "secondary_reference": "representative"},
            "FACE_ID_DIR": {"list": [{"CHAR": "Alice", "ipa_path": "soya_bot/Story/Alice/cache.ipadpt",
                "snapref_path": "soya_bot/Story/Alice/_visual_profiles/card_2",
                "snapref_images": ["first.png", "second.png"]}]}}}
        payload = json.loads(build_illustration_json(json.dumps(source), "", input_dir=None,
                                                    bot_dir=self.bot, include_face_embeddings=False))
        self.assertEqual(self.colors(payload, "first"), [(255, 0, 0), (0, 128, 0)])
        payload["fields"]["SNAPREF"]["secondary_reference"] = "stage"
        self.assertEqual(self.colors(payload, "face"), [(255, 255, 0)])
        source["fields"]["FACE_ID_DIR"]["list"][0]["snapref_images"] = []
        payload = json.loads(build_illustration_json(json.dumps(source), "", input_dir=None,
                                                    bot_dir=self.bot, include_face_embeddings=False))
        self.assertEqual(self.colors(payload, "first"), [(255, 255, 0)])

    def test_saved_legacy_face_reference_and_disabled_assets(self):
        payload = self.payload("face")
        refs = payload["fields"]["SNAPREF"]["references"]
        refs[0].pop("face_asset")
        refs[1].pop("role")
        self.assertEqual(self.colors(payload, "hand"), [(255, 0, 0), (0, 0, 255)])
        payload["fields"]["SNAPREF"]["secondary_reference"] = "stage"
        # New JSON uses the explicit face field, independent of the old second slot.
        refs[0]["face_asset"] = refs[1]["asset"]
        self.assertEqual(self.colors(payload, "first"), [(255, 0, 0)])
        self.assertEqual(self.colors(payload, "face"), [(0, 0, 255)])
        payload["assets"] = {}
        payload["fields"]["SNAPREF"]["strength"] = 0
        for stage in ("first", "hires", "hand", "face", "eye"):
            self.assertEqual(json_nodes.stage_reference_images(payload, stage), {})

    def test_face_asset_survives_embedding_cleanup(self):
        payload = self.payload("stage")
        face = payload["fields"]["SNAPREF"]["references"][0]["face_asset"]
        payload["fields"]["FACE_EMBEDDINGS"] = [{"CHAR": "Alice", "asset": face}]
        saved = json.loads(build_illustration_json(json.dumps(payload), "", input_dir=self.input,
                           include_face_embeddings=False))
        self.assertIn(face, saved["assets"])


class StageModelTests(unittest.TestCase):
    def setUp(self):
        self.spatial = Mock(return_value="regional model")
        self.reference = Mock(side_effect=lambda *args: SimpleNamespace(images=args[3]))
        self.load_lora = Mock(side_effect=lambda path, **kwargs: "loaded:" + path)
        self.ns = definitions((RESOURCES / "soya_illustration_stages.py").read_text(encoding="utf-8"), {
            "math": math, "torch": torch, "F": F, "SnapRef": self.reference,
            "apply_spatial_loras": self.spatial,
            "comfy": SimpleNamespace(utils=SimpleNamespace(load_torch_file=self.load_lora)),
            "load_character_lora_file": self.load_lora,
            "_resolve_lora_path": lambda path: path,
            "_parse_multi_char": lambda value: value,
            "_parse_lora_entries": lambda enabled, data, names: data if enabled else [],
        })
        self.clip = SimpleNamespace(tokenize=lambda text: text, encode_from_tokens_scheduled=lambda tokens: tokens)
        self.settings = {"enabled": True, "strength_scale": "linear", "strength": 1}
        self.images = {"Bob": ["bob reference"], "Alice": ["alice reference"]}
        self.masks = [torch.tensor([[[1., 1., 0., 0.]]]), torch.tensor([[[0., 0., 1., 1.]]])]
        self.multi = {"enable": True, "char_num": 2, "char_name_list": ["Alice", "Bob"],
                      "char_trigger_list": [["alice trigger"], ["bob trigger"]],
                      "char_inform": ["alice description", "bob description"],
                      "shared_before": ["quality"], "shared_after": ["lighting"],
                      "composition_prompt": "two people", "char_composition_list": ["two people", "two people"],
                      "background_prompt": "park", "background_trigger_list": []}

    def hires(self, **overrides):
        values = dict(multi=self.multi, masks=self.masks, lora_activate=True,
                      lora_data=[{"character": "Bob", "lora_path": "b.safetensors", "strength": .8},
                                 {"character": "Alice", "lora_path": "a.safetensors", "strength": .6}],
                      images=self.images, settings=self.settings)
        values.update(overrides)
        return self.ns["prepare_hires_model"]("clean model", self.clip, "vae", "positive",
                                              torch.zeros(1, 16, 32, 3), **values)

    def test_hires_routes_loras_and_references_by_character_not_list_order(self):
        self.assertEqual(self.hires(), "regional model")
        args, kwargs = self.spatial.call_args
        self.assertEqual(args[1], [(1, "loaded:b.safetensors", .8), (0, "loaded:a.safetensors", .6)])
        self.assertIs(args[2], self.masks)
        self.assertIn("alice description", args[3][0])
        self.assertIn("bob description", args[3][1])
        self.assertEqual(kwargs["snapref"].images, {1: ["alice reference"], 2: ["bob reference"]})
        self.assertNotIn("face_tracking", kwargs)

    def test_hires_loras_survive_disabled_references(self):
        self.hires(images={}, settings={"enabled": False})
        self.reference.assert_not_called()
        self.assertEqual(len(self.spatial.call_args.args[1]), 2)
        self.assertIsNone(self.spatial.call_args.kwargs["snapref"])

    def test_hires_uses_only_the_matching_characters_composition(self):
        self.multi["char_composition_list"] = ["The 1girl on the left", "The 1boy on the right"]
        self.multi["composition_prompt"] = ", ".join(self.multi["char_composition_list"])
        self.hires()
        prompts = self.spatial.call_args.args[3]
        self.assertIn("The 1girl on the left", prompts[0])
        self.assertNotIn("The 1boy on the right", prompts[0])
        self.assertIn("The 1boy on the right", prompts[1])
        self.assertNotIn("The 1girl on the left", prompts[1])
        self.assertIn(self.multi["composition_prompt"], self.spatial.call_args.args[4])

    def test_solo_does_not_apply_character_lora_twice(self):
        self.hires(multi={"enable": False}, images={"Alice": ["ref"]})
        self.load_lora.assert_not_called()
        self.assertEqual(self.spatial.call_args.args[1], [])
        self.assertEqual(self.hires(multi={"enable": False}, images={}), "clean model")

    def test_detailer_references_are_local_to_matched_character_and_mask(self):
        mask = torch.ones(1, 8, 8)
        result = self.ns["prepare_detailer_model"]("face lora model", self.clip, "vae", "positive", mask,
                                                   "Alice", self.images, self.settings)
        self.assertEqual(result, "regional model")
        self.assertIs(self.spatial.call_args.args[2][0], mask)
        self.assertEqual(self.reference.call_args.args[3], {1: ["alice reference"]})
        self.assertEqual(self.ns["prepare_detailer_model"]("clean", self.clip, "vae", "pos", mask,
                                                          "unknown", self.images, self.settings), "clean")

    def test_hand_crops_rgb_ownership_before_upscaling(self):
        # A left-hand crop must never receive Bob's right-side reference.
        mask = torch.ones(1, 8, 8)
        self.ns["prepare_hand_model"]("clean", "vae", "hand prompt", mask, (0, 0, 2, 1), (1, 4),
                                      clip=self.clip, masks=self.masks, names=["Alice", "Bob"],
                                      images=self.images, settings=self.settings)
        args, kwargs = self.spatial.call_args
        self.assertTrue(torch.equal(args[2][0], mask))
        self.assertEqual(float(args[2][1].sum()), 0)
        self.assertEqual(kwargs["snapref"].images, {1: ["alice reference"]})


@unittest.skipUnless((SOYA / "soya_first_sampler_v2_core.py").is_file(), "Requires installed SOYA sampler source")
class RegionalNaturalLanguageTests(unittest.TestCase):
    def setUp(self):
        source = (SOYA / "soya_first_sampler_v2_core.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_as_prompt_parts", "_parse_multi_char"}]
        sampler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SoyaFirstSampler_mdsoya")
        nodes.append(next(node for node in sampler.body if isinstance(node, ast.FunctionDef) and node.name == "sample"))
        self.ns = {"__package__": "regional_natural_language_test", "json": json, "traceback": traceback,
                   "_SAMPLER_MODE_FAST": "fast", "_parse_lora_entries": lambda *args: [],
                   "_load_channel_masks": lambda *args: ["left mask", "right mask"]}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<regional sampler>", "exec"), self.ns)
        self.spatial = Mock(return_value="regional model")
        self.spatial_module = ModuleType("regional_natural_language_test.soya_spatial_lora")
        self.spatial_module.apply_spatial_loras = self.spatial
        self.clip = SimpleNamespace(tokenize=lambda text: text, encode_from_tokens_scheduled=lambda text: text)
        self.node = SimpleNamespace(_sample_selected=Mock(return_value=("sampled",)))
        self.multi = {"enable": True, "char_num": 2, "char_name_list": ["Alice", "Bob"],
                      "char_inform": ["blue dress", "red jacket"], "char_trigger_list": [[], []],
                      "background_prompt": "garden", "composition_prompt": "The 1girl on the left, The 1boy on the right",
                      "shared_tag": {"before_char": ["watercolor", "garden"], "after_char": []},
                      "mask_fingerprint": "a" * 64, "mask_render_version": 1}

    def sample(self):
        with patch.dict(sys.modules, {self.spatial_module.__name__: self.spatial_module}):
            self.ns["sample"](self.node, model="base model", positive="whole scene", negative="negative", clip=self.clip,
                              multi_char=self.multi, seed=1, latent_image={}, LORA_ACT=False, LORA_DATA="", mask_location="",
                              steps=1, cfg=1, sampler_name="euler", scheduler="simple", denoise=1, split_mode="",
                              spd_scale=1, spd_sigma=1, adaptive_smc_alpha=1)
        return self.spatial.call_args.args[3], self.spatial.call_args.args[4]

    def test_first_sampler_keeps_each_natural_language_local(self):
        self.multi["char_composition_list"] = ["The 1girl on the left", "The 1boy on the right"]
        prompts, background = self.sample()
        self.assertEqual(prompts, ["watercolor, garden, blue dress, The 1girl on the left",
                                   "watercolor, garden, red jacket, The 1boy on the right"])
        self.assertIn(self.multi["composition_prompt"], background)

    def test_illustration_without_regional_natural_language_keeps_shared_composition(self):
        prompts, background = self.sample()
        for prompt in [*prompts, background]:
            self.assertIn(self.multi["composition_prompt"], prompt)

    def test_empty_natural_language_does_not_inherit_the_common_text(self):
        self.multi["char_composition_list"] = ["", ""]
        prompts, background = self.sample()
        self.assertEqual(prompts, ["watercolor, garden, blue dress", "watercolor, garden, red jacket"])
        self.assertIn(self.multi["composition_prompt"], background)

    def test_common_text_can_be_empty_and_character_can_use_only_natural_language(self):
        self.multi.update(composition_prompt="", char_inform=["", ""], char_composition_list=["reading", "waving"])
        prompts, background = self.sample()
        self.assertEqual(prompts, ["watercolor, garden, reading", "watercolor, garden, waving"])
        self.assertEqual(background, "watercolor, garden")

    def test_malformed_regional_text_is_rejected_without_misassigning_characters(self):
        for values in (None, "text", [], ["left"], ["left", 1]):
            self.multi["char_composition_list"] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "char_composition_list"):
                self.ns["_parse_multi_char"](self.multi)


class JsonNodeRoutingTests(unittest.TestCase):
    setUp = ReferenceTransportTests.setUp
    payload = ReferenceTransportTests.payload

    def test_real_json_wrappers_prepare_each_stage_and_bypass_ilxl(self):
        class Base:
            __module__ = "reference_test.base"
            FUNCTION = "execute"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {
                    "steps": ("INT", {"min": 1}), "cfg": ("FLOAT", {}),
                    "sampler_name": (["euler"],), "scheduler": (["simple"],),
                    "denoise": ("FLOAT", {"min": 0, "max": 1}),
                }}

        package = ModuleType("reference_test")
        package.__path__ = [str(RESOURCES)]
        modules = {"reference_test": package}
        originals = {}
        for base_name, stem in {
            "SoyaFirstSampler_mdsoya": "soya_first_sampler_v2_core",
            "SoyaCharLoraFaceDetailer_mdsoya": "soya_char_lora_face_detailer_v2_core",
            "SoyaCharLoraEyeDetailer_mdsoya": "soya_char_lora_eye_detailer_v2_core",
            "SoyaHandDetailerToggle_mdsoya": "soya_hand_detailer_toggle_v2_core",
            "SoyaHiresfixToggle_mdsoya": "hires",
            "SoyaHiresfixSpectrumToggle_mdsoya": "spectrum_hires",
        }.items():
            originals[base_name] = Base
            module = ModuleType(f"reference_test.{stem}")
            setattr(module, base_name, Base)
            modules[module.__name__] = module
        spatial = ModuleType("reference_test.soya_spatial_face_context")
        spatial.NODE_NAME = "SpatialTest"
        spatial.SoyaSpatialFaceContextV2 = Base
        modules[spatial.__name__] = spatial
        node_spec = importlib.util.spec_from_file_location("reference_test.json_nodes", RESOURCES / "soya_illustration_json_v2.py")
        module = importlib.util.module_from_spec(node_spec)
        node_spec.loader.exec_module(module)
        specs = json.loads((RESOURCES / "illustration_v2_specs.json").read_text(encoding="utf-8"))
        with patch.dict(sys.modules, modules):
            registry = dict(originals)
            module.register_illustration_v2(registry, {})
            for mode in ("face_only", "face", "representative", "stage"):
                for name, binding in specs.items():
                    if binding["base"] not in originals:
                        continue
                    with self.subTest(mode=mode, node=name):
                        payload = self.payload(mode)
                        fields = payload["fields"]
                        for input_name, field in binding["json_inputs"].items():
                            fields.setdefault(field["key"], 1 if field["type"] in {"INT", "FLOAT"} else "")
                        fields.update(MULTI_CHAR={"enable": False}, LORA_DATA={"list": []}, LORA_ACTIVATE=True)
                        if "enable" in binding["json_inputs"]:
                            fields[binding["json_inputs"]["enable"]["key"]] = True
                        sampling = binding["sampling"]
                        fields[sampling["key"]] = dict(steps=2, cfg=1., sampler_name="euler", scheduler="simple", denoise=.2)
                        is_anima = sampling["key"].startswith("ANIMA")
                        kwargs = {"base_model": "anima" if is_anima else "sdxl"}
                        node = registry[name]()
                        result = node.prepare_inputs(json.dumps(payload), kwargs)
                        expected = 1 if mode in {"face_only", "stage"} else 2
                        if "FirstSampler" in name:
                            self.assertEqual(len(result["_embedded_reference_images"][1]), expected)
                        elif "Hiresfix" in name and is_anima:
                            self.assertEqual(len(result["_hires_regions"]["images"]["Alice"]), expected)
                        elif "HandDetailer" in name and is_anima:
                            self.assertEqual(len(result["_hand_references"]["images"]["Alice"]), expected)
                        elif is_anima:
                            self.assertEqual(len(result["_reference_images"]["Alice"]), expected)
                        else:
                            self.assertFalse(any(key.startswith(("_reference", "_hand", "_hires")) for key in result))
                        if "enable" in binding["json_inputs"]:
                            fields[binding["json_inputs"]["enable"]["key"]] = False
                            payload["assets"] = {}  # Bypassed stages must not decode images.
                            node.prepare_inputs(json.dumps(payload), kwargs)


class WorkflowReferenceTests(unittest.TestCase):
    @unittest.skipUnless(SOYA.is_dir(), "Installed SOYA package required")
    def test_installed_detailer_hooks_compile(self):
        for stem in ("soya_char_lora_face_detailer_v2_core", "soya_char_lora_eye_detailer_v2_core",
                     "soya_hand_detailer_toggle_v2_core"):
            source = (SOYA / f"{stem}.py").read_text(encoding="utf-8")
            compile(source, stem, "exec")

    @unittest.skipUnless(SOYA.is_dir(), "Installed distribution workflows required")
    def test_both_distribution_ui_workflows_have_valid_links(self):
        count = 0
        for path in (SRC.parent / "data/comfy_workflow_library/SOYA_DISTRIBUTION/v6").glob("*.json"):
            if "삽화_ANIMA" not in path.name:
                continue
            count += 1
            original = json.loads(path.read_text(encoding="utf-8"))
            by_id = {node["id"]: node for node in original["nodes"]}
            for link in original["links"]:
                self.assertEqual(by_id[link[3]]["inputs"][link[4]]["link"], link[0])
                self.assertIn(link[0], by_id[link[1]]["outputs"][link[2]]["links"])
        self.assertEqual(count, 2)


@unittest.skipUnless((SOYA / "soya_hand_detailer_toggle_v2_core.py").is_file(), "Installed V2 stage hooks required")
class InstalledStageExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.append(str(SOYA.parents[1]))
        package = ModuleType("installed_reference_tests")
        package.__path__ = [str(SOYA)]
        sys.modules[package.__name__] = package
        cls.face = importlib.import_module(package.__name__ + ".soya_char_lora_face_detailer_v2_core")
        cls.eye = importlib.import_module(package.__name__ + ".soya_char_lora_eye_detailer_v2_core")
        cls.hand = importlib.import_module(package.__name__ + ".soya_hand_detailer_toggle_v2_core")

    def setUp(self):
        self.vae = SimpleNamespace(encode=lambda x: x, decode=lambda x: x, spacial_compression_encode=lambda: 8)
        self.clip = SimpleNamespace(tokenize=lambda x: x, encode_from_tokens_scheduled=lambda x: x)
        self.image = torch.zeros(1, 32, 32, 3)

    def test_face_and_eye_crop_sampling_receives_reference_model(self):
        for module, class_name in ((self.face, "SoyaCharLoraFaceDetailer_mdsoya"),
                                   (self.eye, "SoyaCharLoraEyeDetailer_mdsoya")):
            with self.subTest(node=class_name):
                patched = object()
                reference = Mock(return_value=patched)
                sample = Mock(side_effect=lambda *args: args[8]["samples"] + .1)
                kwargs = dict(enable="true", image=self.image, model=object(), clip=self.clip, vae=self.vae,
                              face_context={"matches": [{"name": "Alice", "crop": [8, 8, 24, 24], "bbox": [8, 8, 24, 24], "score": 1.}]},
                              char_tags=json.dumps({"list": [{"CHAR": "Alice", "FACE_TAGS": "face", "EYE_TAGS": "eyes"}]}),
                              quality_tags="quality", artist_tags="", negative="negative", lora_list='{"list":[]}', base_model="anima",
                              crop_expand_factor=1., upscale_factor=1., seed=42, steps=2, cfg=1., sampler_name="euler", scheduler="simple",
                              denoise=.2, feather=0., noise_mask=True,
                              _reference_images={"Alice": ["primary", "face"]}, _reference_settings={"enabled": True})
                if module is self.face:
                    kwargs["corner_roundness"] = 0.
                else:
                    kwargs.update(eye_context={"faces": [{"eye_mask": np.ones((16, 16), dtype=np.uint8),
                                                          "bbox_clamped": [8, 8, 24, 24]}]},
                                  eyebrow_restore=False, eyebrow_opacity=0., eyebrow_th=.5)
                with patch.object(module, "prepare_detailer_model", reference), patch.object(module, "_run_ksampler", sample):
                    result = getattr(module, class_name)().execute(**kwargs)
                self.assertIs(sample.call_args.args[0], patched)
                self.assertEqual(reference.call_args.args[5], "Alice")
                self.assertEqual(reference.call_args.args[4].shape, (1, 16, 16))
                self.assertGreater(float(result[0].sum()), 0)
                self.assertEqual(float(result[0][0, :8].sum()), 0)

    def test_hand_sampler_uses_per_crop_reference_callback(self):
        node = self.hand.SoyaHandDetailerToggle_mdsoya()
        patched = object()
        reference = Mock(return_value=patched)
        sample = Mock(side_effect=lambda *args: args[8]["samples"] + .1)
        with patch.object(node, "_detect_hands", return_value=[(8, 8, 24, 24)]), patch.object(node, "_run_ksampler", sample):
            result = node.doit(enable="true", image=self.image, model=SimpleNamespace(get_model_object=lambda key: None),
                               clip=self.clip, vae=self.vae, positive="hands", negative="negative", seed=42, steps=2,
                               cfg=1., sampler_name="euler", scheduler="simple", denoise=.2,
                               bbox_threshold=.4, mask_expand=1., crop_expand=1., upscale_factor=1., feather=0.,
                               corner_roundness=0., noise_mask=True, drop_size=1,
                               bbox_detector=SimpleNamespace(bbox_model=lambda: None), cycle=1, _reference_model=reference)
        self.assertIs(sample.call_args.args[0], patched)
        self.assertEqual(reference.call_args.args[4], (8, 8, 24, 24))
        self.assertEqual(reference.call_args.args[5], (32, 32))
        self.assertGreater(float(result[0].sum()), 0)

    def test_both_hires_implementations_sample_the_prepared_model(self):
        package = "installed_reference_tests."
        wrappers = importlib.import_module(package + "soya_illustration_json_v2")
        stages = importlib.import_module(package + "soya_illustration_stages")
        stock = importlib.import_module(package + "soya_hiresfix_toggle")
        spectrum = importlib.import_module(package + "soya_hiresfix_spectrum_toggle")
        registry = {"SoyaHiresfixToggle_mdsoya": stock.SoyaHiresfixToggle_mdsoya,
                    "SoyaHiresfixSpectrumToggle_mdsoya": spectrum.SoyaHiresfixSpectrumToggle_mdsoya}
        wrappers.register_illustration_v2(registry, {})
        payload = json.dumps({"schema": "soya.illustration.v2", "assets": {}, "fields": {
            "ANIMA_HRF_ACTIVATE": True, "SEED": 42, "MULTI_CHAR": {"enable": False},
            "ANIMA_HIRESFIX": {"steps": 2, "cfg": 1., "sampler_name": "euler", "scheduler": "simple", "denoise": .2},
        }})
        for name in ("SoyaHiresfixToggleV2_1b3a7a6c_mdsoya", "SoyaHiresfixSpectrumToggleV2_2ca1fb16_mdsoya"):
            with self.subTest(node=name):
                prepared = object()
                sampler = Mock(return_value=({"samples": self.image + .1},))
                stub_nodes = ModuleType("nodes")
                stub_nodes.common_ksampler = sampler
                with patch.object(stages, "prepare_hires_model", return_value=prepared) as prepare, \
                     patch.object(spectrum, "sample_spectrum_mod_guidance", sampler), patch.dict(sys.modules, {"nodes": stub_nodes}):
                    result = registry[name]().execute(payload, image=self.image, model=object(), clip=self.clip,
                                                     vae=self.vae, positive="positive", negative="negative", tiled_vae="false", tile_size=512)
                self.assertIs(sampler.call_args.args[0], prepared)
                self.assertIs(prepare.call_args.args[1], self.clip)
                self.assertGreater(float(result[0].sum()), 0)


if __name__ == "__main__":
    unittest.main()
