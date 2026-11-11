import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from illustration_json import (build_illustration_json, illustration_edit_document,
                               illustration_edit_prompts, is_moss_prompt,
                               update_workflow_json_settings, workflow_json_defaults)
from modes.moss_prompt_builder import (MossNpcPromptBuilder, MossPromptBuilder,
                                       build_controller, build_moss_layout,
                                       layout_bbox, merge_lora_lists, preview_layout)

RESOURCE = Path(__file__).resolve().parents[2] / "data" / "comfy" / "custom_nodes" / "comfyui-soya-custom-nodes"


def moss_document(*, loras=None, width=1024, height=1024):
    return {
        "moss": {
            "summary": {"caption": "two girls talking"},
            "items": [{"label": "character", "caption": "a girl", "bbox": [-512, -512, 512, 512]}],
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        },
        "workflow": {
            "model": "moss.safetensors", "clip": "clip.safetensors", "vae": "vae.safetensors",
            "width": width, "height": height, "batch_size": 1, "seed": 0,
            "negative": "low quality",
            "sampling": {"steps": 8, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0},
            "loras": {"activate": bool(loras), "list": list(loras or []), "filter": ""},
        },
    }


def workflow_fixture(document=None):
    document = document or moss_document()
    return {
        "1": {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "긍정프롬프트"},
              "inputs": {"value": json.dumps(document, ensure_ascii=False)}},
        "2": {"class_type": "SoyaUNETLoaderV2_moss_mdsoya",
              "inputs": {"prompt_json": ["1", 0], "model_key": "workflow.model"}},
        "3": {"class_type": "SoyaCLIPLoaderV2_moss_mdsoya",
              "inputs": {"prompt_json": ["1", 0], "model_key": "workflow.clip"}},
        "4": {"class_type": "SoyaVAELoaderV2_moss_mdsoya",
              "inputs": {"prompt_json": ["1", 0], "model_key": "workflow.vae"}},
        "5": {"class_type": "SoyaKSamplerV2_moss_mdsoya",
              "inputs": {"prompt_json": ["1", 0], "sampling_key": "workflow.sampling"}},
        "6": {"class_type": "SoyaBatchLoraLoaderV2_moss_mdsoya",
              "inputs": {"prompt_json": ["1", 0]}},
    }


class LayoutBBoxTests(unittest.TestCase):
    def test_matches_native_centered_yxyx(self):
        # Normalized 0.1..0.9 of a 768x1024 canvas.
        self.assertEqual(layout_bbox(0.05, 0.1, 0.4, 0.8, 768, 1024),
                         [-410, -346, 410, -38])
        self.assertEqual(layout_bbox(0, 0, 1, 1, 1024, 1024), [-512, -512, 512, 512])

    def test_rejects_out_of_canvas(self):
        with self.assertRaises(ValueError):
            layout_bbox(0.8, 0.8, 0.4, 0.4, 1024, 1024)


class LayoutTests(unittest.TestCase):
    def test_regions_become_character_items(self):
        layout = {"composition_prompt": "two girls talking", "regions": [
            {"name": "A", "character_prompt": "silver hair", "x": 0.05, "y": 0.1,
             "width": 0.4, "height": 0.8},
            {"name": "B", "character_prompt": "brown hair", "x": 0.55, "y": 0.1,
             "width": 0.4, "height": 0.8},
        ]}
        moss = build_moss_layout(layout, "1girl", image_width=768, image_height=1024)
        self.assertEqual(moss["summary"]["caption"], "two girls talking")
        self.assertEqual([item["caption"] for item in moss["items"]],
                         ["silver hair", "brown hair"])
        self.assertEqual(moss["bbox_format"], "yxyx")
        self.assertEqual(moss["bbox_coord_space"], "abs_centralized")
        self.assertEqual(moss["items"][0]["bbox"], [-410, -346, 410, -38])

    def test_multi_char_without_layout_uses_full_canvas(self):
        fields = {"MULTI_CHAR": {"enable": True, "char_name_list": ["A"],
                                 "char_inform": ["a girl"]}}
        moss = build_moss_layout(None, "", fields=fields, image_width=512, image_height=512)
        self.assertEqual(moss["items"][0]["caption"], "a girl")
        self.assertEqual(moss["items"][0]["bbox"], [-256, -256, 256, 256])

    def test_solo_has_no_items(self):
        moss = build_moss_layout(None, "1girl, garden", image_width=1024, image_height=1024)
        self.assertEqual(moss, {"summary": {"caption": "1girl, garden"}})

    def test_single_subject_npc_uses_its_card_values_without_bbox(self):
        # The NPC generation node already decided these values. MOSS consumes
        # them as one summary caption and never asks for a regional box.
        fields = {"ANIMA_QUALITY": "score_9, best quality",
                  "MULTI_CHAR": {"enable": True, "char_name_list": ["Porter"],
                                 "char_inform": ["1boy, brown hair"]}}
        moss = build_moss_layout(None, "1boy, brown coat, short hair", fields=fields,
                                 image_width=768, image_height=1024, single_subject=True)
        self.assertEqual(moss, {"summary": {"caption":
                         "1boy, brown coat, short hair, score_9, best quality"}})
        self.assertNotIn("items", moss)

    def test_dedicated_npc_builder_inherits_card_values(self):
        fields = {"ANIMA_QUALITY": "score_9, best quality",
                  "MULTI_CHAR": {"enable": True, "char_name_list": ["Porter"],
                                 "char_inform": ["1boy, brown hair"]}}
        moss = MossNpcPromptBuilder.build_layout(
            "1boy, brown coat, short hair", fields=fields)
        self.assertEqual(moss, {"summary": {"caption":
                         "1boy, brown coat, short hair, score_9, best quality"}})
        self.assertEqual(set(moss), {"summary"})


class NpcPromptSelectionTests(unittest.TestCase):
    def setUp(self):
        from modes.illustration_prompt_builders import npc_generation_messages
        self.build = npc_generation_messages
        self.candidates = [{"name": "Porter", "aliases": [], "description": "porter",
                            "anchors": ["C001"]}]
        self.prompts = {"npc_generation": "ANIMA NPC INSTRUCTION",
                        "npc_generation_moss": "MOSS NPC INSTRUCTION"}

    def messages(self, toggles):
        return self.build(self.prompts, self.candidates, "", "", toggles)

    def test_moss_uses_its_own_instruction(self):
        system = self.messages({"illustration_model_family": "moss"})[0]["content"]
        self.assertIn("MOSS NPC INSTRUCTION", system)
        self.assertNotIn("ANIMA NPC INSTRUCTION", system)

    def test_anima_keeps_the_shared_instruction(self):
        system = self.messages({"illustration_model_family": "anima"})[0]["content"]
        self.assertIn("ANIMA NPC INSTRUCTION", system)
        self.assertNotIn("MOSS NPC INSTRUCTION", system)

    def test_missing_moss_file_falls_back_to_shared_instruction(self):
        self.prompts.pop("npc_generation_moss")
        system = self.messages({"illustration_model_family": "moss"})[0]["content"]
        self.assertIn("ANIMA NPC INSTRUCTION", system)


class PreviewLayoutTests(unittest.TestCase):
    def test_centered_boxes_become_normalized_regions(self):
        document = moss_document()
        document["moss"]["items"][0]["bbox"] = [-256, -256, 256, 256]
        layout = preview_layout(document)
        self.assertEqual(layout["mask_width"], 1024)
        self.assertEqual(layout["mask_height"], 1024)
        region = layout["regions"][0]
        self.assertEqual((region["x"], region["y"], region["width"], region["height"]),
                         (0.25, 0.25, 0.5, 0.5))
        self.assertEqual(region["name"], "character 1")
        self.assertEqual(region["caption"], "a girl")

    def test_bare_layout_and_workflow_size(self):
        moss = {
            "summary": {"caption": "two girls"},
            "items": [{"label": "character", "caption": "a girl",
                       "bbox": [-100, -200, 100, 200]}],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }
        layout = preview_layout({"moss": moss, "workflow": {"width": 768, "height": 1024}})
        self.assertEqual(layout["mask_width"], 768)
        self.assertEqual(layout["mask_height"], 1024)
        region = layout["regions"][0]
        self.assertEqual((region["x"], region["y"]), (0.239583, 0.402344))

    def test_clamps_to_canvas_and_skips_empty(self):
        document = moss_document()
        document["moss"]["items"] = [
            {"label": "character", "caption": "a", "bbox": [-999, -999, 999, 999]},
            {"label": "object", "caption": "b", "bbox": [500, 500, 500, 500]},
        ]
        layout = preview_layout(document)
        self.assertEqual(len(layout["regions"]), 1)
        region = layout["regions"][0]
        self.assertEqual((region["x"], region["y"], region["width"], region["height"]),
                         (0.0, 0.0, 1.0, 1.0))

    def test_legacy_fields_layout_is_not_a_preview(self):
        self.assertIsNone(preview_layout({"regions": [{"x": 0, "y": 0,
                                                       "width": 0.5, "height": 0.5}]}))


class DetailMossTests(unittest.TestCase):
    def test_absolute_layout_passes_through_with_objects(self):
        layout = {
            "summary": {"caption": "two girls at a bench"},
            "items": [
                {"label": "character", "name": "A", "caption": "silver hair",
                 "bbox": [-100, -200, 100, 200]},
                {"label": "object", "name": "bench", "caption": "wooden bench",
                 "bbox": [100, -300, 200, 300]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
            "image_size": {"width": 768, "height": 1024},
        }
        moss = build_moss_layout(layout, "", image_width=768, image_height=1024)
        self.assertEqual([item["label"] for item in moss["items"]],
                         ["character", "object"])
        # Reference-picture mapping resolves characters by this name.
        self.assertEqual(moss["items"][0]["name"], "A")
        self.assertEqual(moss["items"][1]["name"], "bench")
        self.assertEqual(moss["items"][0]["bbox"], [-100, -200, 100, 200])
        self.assertEqual(moss["bbox_coord_space"], "abs_centralized")
        self.assertEqual(moss["bbox_format"], "yxyx")

    def test_detail_moss_messages_pass_image_size(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        scene = {"slot": 1, "characters": [{"name": "A", "profile": {}}]}
        messages = detail_moss_messages(
            "MOSS DETAIL", [scene], toggles={"illustration_model_family": "moss"},
            canvas_sizes={"solo": (768, 1024)})
        self.assertEqual(messages[0], {"role": "system", "content": "MOSS DETAIL"})
        user = messages[1]["content"]
        self.assertIn('"width": 768', user)
        self.assertIn('"height": 1024', user)
        self.assertIn("bbox_coord_space", user)
        self.assertIn("abs_centralized", user)

    def test_detail_moss_includes_keyvis_contract(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        scene = {"slot": 1, "characters": [{"name": "A", "profile": {}}]}
        keyvis = [
            {"role": "system", "content": "KEYVIS SYSTEM\n\nOUTPUT FORMAT\nTOON shape"},
            {"role": "user", "content": "# KEY VISUAL TIMELINE\nA"},
        ]
        messages = detail_moss_messages(
            "MOSS DETAIL", [scene],
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)},
            keyvis_messages=keyvis)
        system = messages[0]["content"]
        self.assertIn("KEY VISUAL INSTRUCTIONS", system)
        self.assertIn("KEYVIS SYSTEM", system)
        self.assertNotIn("TOON shape", system)
        combined = messages[1]["content"]
        self.assertIn("keyvis", combined)
        self.assertIn("[1152, 768]", combined)
        self.assertIn("# KEY VISUAL TIMELINE", messages[-1]["content"])

    def test_moss_keyvis_parser_uses_layout_canvas(self):
        from modes.illustration_context_pipeline import _parse_call2_keyvis_output
        response = {"scenes": [], "keyvis": {
            "image_size": {"width": 1, "height": 1},
            "summary": {"caption": "cover art"},
            "items": [
                {"label": "character", "name": "A", "caption": "1girl",
                 "bbox": [-10, -10, 10, 10]},
                {"label": "character", "name": "B", "caption": "1boy",
                 "bbox": [-10, -10, 10, 10]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }}
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        descriptor, reason = _parse_call2_keyvis_output(
            text, {"key_visual": True}, ["A", "B"], "TEST",
            allow_scenes=True, moss=True,
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)})
        self.assertEqual(reason, "")
        self.assertEqual(descriptor["kind"], "keyvis")
        self.assertEqual(descriptor["slot"], -1)
        self.assertEqual(descriptor["moss_image_size"], {"width": 1152, "height": 768})
        self.assertEqual([entry["name"] for entry in descriptor["moss_layout"]["items"]],
                         ["A", "B"])

    def test_detail_moss_group_profile_uses_group_canvas(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        scene = {"slot": 1, "characters": [{"name": "A", "profile": {}},
                                           {"name": "B", "profile": {}}]}
        messages = detail_moss_messages(
            "MOSS DETAIL", [scene],
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)})
        user = messages[1]["content"]
        self.assertIn('"width": 1152', user)
        self.assertIn('"height": 768', user)

    def test_supplied_canvas_wins_over_model_echo(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        scene = {"slot": 1, "characters": [{"name": "A"}],
                 "image_size": {"width": 768, "height": 1024}}
        response = {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1024},
            "summary": {"caption": "park"},
            "items": [{"label": "character", "name": "A", "caption": "1girl",
                       "bbox": [-100, -200, 100, 200]}],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }]}
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [scene], moss=True)[0]
        self.assertEqual(descriptor["moss_image_size"], {"width": 768, "height": 1024})
        self.assertEqual(descriptor["moss_layout"]["image_size"],
                         {"width": 768, "height": 1024})

    def test_detail_parser_keeps_absolute_boxes_and_image_size(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        scene = {"slot": 1, "characters": [{"name": "A"}]}
        response = {"scenes": [{
            "slot": 1,
            "image_size": {"width": 768, "height": 1024},
            "summary": {"caption": "park"},
            "items": [
                {"label": "character", "name": "A", "caption": "1girl",
                 "bbox": [-100, -200, 100, 200]},
                {"label": "object", "name": "bench", "caption": "bench",
                 "bbox": [100, -300, 200, 300]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }]}
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [scene], moss=True)[0]
        self.assertEqual(descriptor["slot"], 1)
        self.assertEqual(descriptor["characters"][0]["positive"], "1girl")
        self.assertEqual(descriptor["moss_image_size"], {"width": 768, "height": 1024})
        self.assertEqual([item["label"] for item in descriptor["moss_layout"]["items"]],
                         ["character", "object"])
        self.assertEqual(descriptor["moss_layout"]["items"][1]["bbox"],
                         [100, -300, 200, 300])


class LoraAndControllerTests(unittest.TestCase):
    def test_merge_keeps_order_and_drops_duplicates(self):
        character = {"list": [{"lora_path": "a.safetensors", "str": 0.7, "BASE": "anima"}]}
        common = {"list": [{"lora_path": "a.safetensors", "str": 0.9, "BASE": "anima"},
                           {"lora_path": "b.safetensors", "str": 0.5, "BASE": "anima"}]}
        merged = merge_lora_lists(character, common)
        self.assertEqual([item["lora_path"] for item in merged], ["a.safetensors", "b.safetensors"])
        self.assertEqual(merged[0]["str"], 0.7)

    def test_controller_activates_when_any_lora_exists(self):
        empty = build_controller(model="m", lora_list=[])
        self.assertFalse(empty["loras"]["activate"])
        full = build_controller(model="m", lora_list=[{"lora_path": "a.safetensors"}])
        self.assertTrue(full["loras"]["activate"])

    def test_document_has_no_schema(self):
        document = MossPromptBuilder.build(moss_document()["moss"], moss_document()["workflow"])
        self.assertEqual(set(document), {"moss", "workflow"})
        self.assertNotIn("schema", document)
        self.assertEqual(json.loads(MossPromptBuilder.dumps(document))["moss"]["bbox_format"], "yxyx")


class EditDocumentTests(unittest.TestCase):
    def test_edit_document_keeps_moss_shape(self):
        document = moss_document()
        text = json.dumps(document, ensure_ascii=False)
        self.assertTrue(is_moss_prompt(text))
        self.assertEqual(illustration_edit_document(text, "bad"), document)

    def test_edit_prompts_round_trip_moss_for_redraw(self):
        document = moss_document()
        positive, negative = illustration_edit_prompts(document, "raw source")
        self.assertTrue(is_moss_prompt(positive))
        self.assertEqual(json.loads(positive), document)
        self.assertEqual(negative, document["workflow"]["negative"])


class ServerDocumentTests(unittest.TestCase):
    def test_build_illustration_json_emits_moss_controller(self):
        defaults = workflow_json_defaults(workflow_fixture())
        self.assertEqual(set(defaults), {"workflow.model", "workflow.clip", "workflow.vae",
                                         "workflow.sampling", "STYLE_LORA_DATA"})
        tagged = (
            "[ANIMA_CONTENT]\n1girl\n[IMG_W]\n768\n[IMG_H]\n1024\n[SEED]\n42\n"
            "[LORA_DATA]\n{\"list\":[{\"lora_path\":\"char.safetensors\",\"str\":0.7,"
            "\"BASE\":\"anima\"}]}\n[STYLE_LORA_DATA]\n{\"list\":[]}\n[END]"
        )
        layout = {"composition_prompt": "two girls", "regions": [
            {"name": "A", "character_prompt": "a girl", "x": 0, "y": 0,
             "width": 1, "height": 1},
        ]}
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults,
            moss_layout=layout))
        self.assertEqual(set(payload), {"moss", "workflow"})
        self.assertEqual(payload["workflow"]["width"], 768)
        self.assertEqual(payload["workflow"]["height"], 1024)
        self.assertEqual(payload["workflow"]["seed"], 42)
        self.assertEqual(payload["workflow"]["model"], "moss.safetensors")
        self.assertEqual(payload["workflow"]["negative"], "low quality")
        self.assertTrue(payload["workflow"]["loras"]["activate"])
        self.assertEqual([item["lora_path"] for item in payload["workflow"]["loras"]["list"]],
                         ["char.safetensors"])

    def test_moss_defaults_without_layout_still_use_moss_builder(self):
        # Manual draw can arrive without a Regional layout; the MOSS workflow
        # must still receive its layout/controller document, not legacy fields.
        defaults = workflow_json_defaults(workflow_fixture())
        tagged = "[ANIMA_CONTENT]\n1girl, garden\n[IMG_W]\n768\n[IMG_H]\n1024\n[END]"
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults))
        self.assertEqual(set(payload), {"moss", "workflow"})
        self.assertEqual(payload["moss"], {"summary": {"caption": "1girl, garden"}})
        self.assertEqual(payload["workflow"]["width"], 768)
        self.assertEqual(payload["workflow"]["negative"], "low quality")

    def test_npc_representative_inherits_card_values_without_bbox(self):
        defaults = workflow_json_defaults(workflow_fixture())
        tagged = (
            "[ANIMA_CONTENT]\n1boy, brown coat, short hair\n"
            "[ANIMA_QUALITY]\nscore_9, best quality\n"
            "[IMG_W]\n768\n[IMG_H]\n1024\n[SEED]\n7\n"
            '[MULTI_CHAR]\n{"enable":true,"char_name_list":["Porter"],'
            '"char_inform":["1boy, brown hair"]}\n[END]'
        )
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults,
            npc_representative=True))
        self.assertEqual(payload["moss"], {"summary": {"caption":
                         "1boy, brown coat, short hair, score_9, best quality"}})
        self.assertEqual(payload["workflow"]["seed"], 7)
        self.assertEqual(payload["workflow"]["width"], 768)
        self.assertEqual(payload["workflow"]["height"], 1024)
        self.assertEqual(payload["workflow"]["negative"], "low quality")

    def test_anima_defaults_keep_legacy_v2_document(self):
        defaults = {"ANIMA_FIRST_SAMPLER": {"steps": 23, "cfg": 4.0}}
        tagged = "[ANIMA_CONTENT]\n1girl\n[END]"
        payload = json.loads(build_illustration_json(
            tagged, "bad", input_dir=".", workflow_defaults=defaults))
        self.assertEqual(payload["schema"], "soya.illustration.v2")
        self.assertIn("fields", payload)

    def test_preset_round_trip_writes_loras_into_controller(self):
        workflow = workflow_fixture()
        fields = workflow_json_defaults(workflow)
        fields["STYLE_LORA_DATA"] = {"list": [
            {"lora_path": "@ix7.safetensors", "str": 0.8, "BASE": "anima", "trigger": "@ixy"},
        ]}
        updated = update_workflow_json_settings(workflow, fields)
        document = json.loads(updated["1"]["inputs"]["value"])
        self.assertNotIn("schema", document)
        self.assertTrue(document["workflow"]["loras"]["activate"])
        self.assertEqual(document["workflow"]["loras"]["list"][0]["lora_path"],
                         "@ix7.safetensors")
        self.assertEqual(workflow_json_defaults(updated)["STYLE_LORA_DATA"],
                         {"list": document["workflow"]["loras"]["list"]})

    def test_preset_migration_keeps_legacy_flat_values(self):
        workflow = workflow_fixture()
        fields = workflow_json_defaults(workflow)
        fields.pop("workflow.sampling")
        fields.pop("workflow.model")
        fields["MOSS_FIRST_SAMPLER"] = {
            "steps": 12, "cfg": 4.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
        }
        fields["MOSS_DIFFUSION_MODEL"] = "legacy-moss.safetensors"
        updated = update_workflow_json_settings(workflow, fields)
        migrated = workflow_json_defaults(updated)
        self.assertEqual(migrated["workflow.sampling"]["steps"], 12)
        self.assertEqual(migrated["workflow.model"], "legacy-moss.safetensors")


class MossCharRefDocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bot = self.root / "bot"
        for name, colors in (("Alice", (b"a1", b"a2")), ("Bob", (b"b1", b"b2"))):
            folder = self.bot / "Story" / name
            folder.mkdir(parents=True)
            for index, data in enumerate(colors, 1):
                (folder / f"rep{index}.png").write_bytes(data)
        self.defaults = workflow_json_defaults(workflow_fixture())
        self.layout = {"composition_prompt": "two girls", "regions": [
            {"name": "Alice", "character_prompt": "silver hair", "x": 0, "y": 0,
             "width": 0.5, "height": 1},
            {"name": "Bob", "character_prompt": "blue coat", "x": 0.5, "y": 0,
             "width": 0.5, "height": 1},
        ]}

    def tagged(self, enabled=True, strength=1.0):
        snapref = {"enabled": enabled, "strength_scale": "linear",
                   "strength": strength, "secondary_reference": "none"}
        face = {"list": [
            {"CHAR": "Alice", "snapref_path": "soya_bot/Story/Alice",
             "snapref_images": ["rep1.png", "rep2.png"]},
            {"CHAR": "Bob", "snapref_path": "soya_bot/Story/Bob",
             "snapref_images": ["rep1.png", "rep2.png"]},
        ]}
        return (
            "[ANIMA_CONTENT]\n2girls\n[IMG_W]\n1024\n[IMG_H]\n1024\n"
            "[CHAR_LIST]\nAlice,Bob\n"
            f"[FACE_ID_DIR]\n{json.dumps(face)}\n"
            f"[SNAPREF]\n{json.dumps(snapref)}\n"
            '[LORA_DATA]\n{"list":[{"lora_path":"char.safetensors","str":0.7,'
            '"BASE":"anima"}]}\n'
            '[STYLE_LORA_DATA]\n{"list":[{"lora_path":"@ix7.safetensors","str":0.8,'
            '"BASE":"anima","trigger":"@ixy"}]}\n[END]'
        )

    def build(self, enabled=True, strength=1.0):
        return json.loads(build_illustration_json(
            self.tagged(enabled, strength), "low quality", input_dir=".",
            workflow_defaults=self.defaults, moss_layout=self.layout,
            bot_dir=self.bot))

    def test_reference_off_keeps_the_old_document(self):
        payload = self.build(enabled=False)
        self.assertEqual(set(payload), {"moss", "workflow"})
        self.assertNotIn("reference", payload["workflow"])
        self.assertEqual(payload["workflow"]["model"], "moss.safetensors")

    def test_reference_on_keeps_moss_pure_and_preset_values(self):
        baseline = self.build(enabled=False)
        payload = self.build()
        self.assertNotIn("reference", payload)
        reference = payload["workflow"]["reference"]
        self.assertTrue(reference["activate"])
        self.assertEqual(payload["moss"], baseline["moss"])
        self.assertEqual([item["name"] for item in reference["pictures"]],
                         ["Alice", "Bob"])
        self.assertEqual([item["picture"] for item in reference["pictures"]], [1, 2])
        self.assertEqual(len(payload["assets"]), 2)
        for picture in reference["pictures"]:
            self.assertIn(picture["asset"], payload["assets"])
        # Reference mode must not overwrite the selected model or sampling values.
        self.assertEqual(payload["workflow"]["model"], "moss.safetensors")
        self.assertEqual(payload["workflow"]["sampling"], {
            "steps": 8, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0,
        })
        paths = [item["lora_path"] for item in payload["workflow"]["loras"]["list"]]
        self.assertEqual(paths, ["char.safetensors", "@ix7.safetensors",
                                 "moss_image_charref_v2.safetensors"])

    def test_reference_on_zero_strength_stays_off(self):
        payload = self.build(strength=0)
        self.assertNotIn("reference", payload["workflow"])
        self.assertNotIn("assets", payload)

    def test_regeneration_preserves_reference_and_preset_values(self):
        payload = self.build()
        rebuilt = json.loads(build_illustration_json(
            json.dumps(payload, ensure_ascii=False), "ignored",
            input_dir=".", workflow_defaults=self.defaults, bot_dir=self.bot))
        self.assertEqual(rebuilt["workflow"]["reference"], payload["workflow"]["reference"])
        self.assertEqual(rebuilt["assets"], payload["assets"])
        self.assertEqual(rebuilt["workflow"]["model"], "moss.safetensors")
        self.assertEqual(rebuilt["workflow"]["sampling"]["steps"], 8)
        self.assertNotIn("reference", rebuilt)

    def test_regeneration_keeps_saved_values_when_the_preset_differs(self):
        payload = self.build()
        defaults = dict(self.defaults)
        defaults["workflow.model"] = "other.safetensors"
        defaults["workflow.clip"] = "other-clip.safetensors"
        defaults["workflow.vae"] = "other-vae.safetensors"
        defaults["workflow.sampling"] = {
            "steps": 20, "cfg": 5.0, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 0.8,
        }
        defaults["STYLE_LORA_DATA"] = {"list": [
            {"lora_path": "new-style.safetensors", "str": 0.9, "BASE": "anima"},
        ]}
        rebuilt = json.loads(build_illustration_json(
            json.dumps(payload, ensure_ascii=False), "ignored",
            input_dir=".", workflow_defaults=defaults, bot_dir=self.bot))
        self.assertEqual(rebuilt["workflow"]["model"], "moss.safetensors")
        self.assertEqual(rebuilt["workflow"]["clip"], "clip.safetensors")
        self.assertEqual(rebuilt["workflow"]["vae"], "vae.safetensors")
        self.assertEqual(rebuilt["workflow"]["sampling"], payload["workflow"]["sampling"])
        self.assertEqual(rebuilt["workflow"]["loras"], payload["workflow"]["loras"])

    def test_encode_prompt_adds_picture_markers_to_a_copy(self):
        payload = self.build()
        spec = importlib.util.spec_from_file_location(
            "moss_v2_charref", RESOURCE / "soya_illustration_json_v2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        read = module.read_payload(json.dumps(payload, ensure_ascii=False))
        reference = module.moss_reference(read)
        self.assertTrue(reference)
        text = module._moss_edit_prompt(read, reference)
        encoded = json.loads(text)
        captions = {item["name"]: item["caption"] for item in encoded["items"]}
        self.assertTrue(captions["Alice"].startswith(
            "Same character and art style as Picture 1."))
        self.assertTrue(captions["Bob"].startswith(
            "Same character and art style as Picture 2."))
        # The stored payload's moss layout is untouched.
        stored = module.read_payload(json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("Picture 1", json.dumps(stored["moss"], ensure_ascii=False))


class MossWorkflowCharRefTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def read_workflow(self, filename):
        # The live copy is hidden while its models are uninstalled, so fall
        # back to the unpacked distribution original used by the installer.
        candidates = (
            self.ROOT / "data/comfy/user/default/workflows" / filename,
            self.ROOT / "data/comfy_workflow_library/SOYA_DISTRIBUTION/v6" / filename,
        )
        for path in candidates:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        self.skipTest(f"workflow artifact not installed: {filename}")

    def test_charref_chain_is_wired(self):
        cases = (
            ("배포_공방_삽화_MOSS_8STEP_V1.json", "SoyaMossSkipLayerGuidanceV2_mdsoya", 22),
            ("배포_공방_삽화_MOSS_4STEP_V1.json", "SoyaMossOptionalLoraV2_mdsoya", 23),
        )
        for filename, guidance_type, reference_id in cases:
            with self.subTest(filename=filename):
                document = self.read_workflow(filename)
                by_id = {node["id"]: node for node in document["nodes"]}
                types = [node["type"] for node in document["nodes"]]
                for expected in ("SoyaMOSSEditModelPatch_mdsoya",
                                 "SoyaMossReferenceImagesV2_mdsoya",
                                 "SoyaMossPromptEncodeV2_mdsoya",
                                 "SoyaMossNegativeEncodeV2_mdsoya",
                                 guidance_type):
                    self.assertIn(expected, types)
                self.assertEqual(document["extra"]["soya_moss_charref"], 1)
                self.assertIn(reference_id, by_id)
                unet = by_id[10]
                patch_link = document["links"][next(
                    i for i, link in enumerate(document["links"]) if link[0] == 10)]
                self.assertEqual(unet["outputs"][0]["links"], [10])
                self.assertEqual(patch_link[3], next(
                    node["id"] for node in document["nodes"]
                    if node["type"] == "SoyaMOSSEditModelPatch_mdsoya"))
                positive = by_id[14]
                self.assertEqual(positive["type"], "SoyaMossPromptEncodeV2_mdsoya")
                self.assertEqual([entry["name"] for entry in positive["inputs"]],
                                 ["clip", "prompt_json", "vae", "image1", "image2", "image3"])
                for entry in positive["inputs"]:
                    self.assertIsNotNone(entry["link"])
                negative = by_id[15]
                self.assertEqual(negative["type"], "SoyaMossNegativeEncodeV2_mdsoya")


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "moss_soya_v2", RESOURCE / "soya_illustration_json_v2.py")
        cls.node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.node)

    def test_dotted_paths_and_compact_moss_json(self):
        document = moss_document(loras=[{"lora_path": "x.safetensors", "str": 0.5,
                                         "BASE": "anima"}], width=768, height=512)
        payload = self.node.read_payload(json.dumps(document, ensure_ascii=False))
        self.assertEqual(self.node.field_value(payload, "workflow.width", "INT"), 768)
        self.assertEqual(self.node.field_value(payload, "workflow.loras.activate", "STRING"), "true")
        self.assertEqual(json.loads(self.node.field_value(payload, "workflow.loras", "STRING"))["list"][0]["lora_path"],
                         "x.safetensors")
        moss_text = self.node.field_value(payload, "moss", "STRING")
        self.assertEqual(moss_text, json.dumps(json.loads(moss_text), ensure_ascii=False,
                                               separators=(",", ":")))
        self.assertEqual(json.loads(moss_text)["bbox_format"], "yxyx")

    def test_legacy_document_still_resolves_fields(self):
        payload = self.node.read_payload(json.dumps({
            "schema": "soya.illustration.v2", "fields": {"IMG_W": 512}, "assets": {}}))
        self.assertEqual(self.node.field_value(payload, "IMG_W", "INT"), 512)

    def test_reference_area_budget_is_shared_across_pictures(self):
        # Two 1MP references would double the cached CharRef K/V.
        self.assertEqual(self.node._moss_reference_megapixels(0), 1.0)
        self.assertEqual(self.node._moss_reference_megapixels(1), 1.0)
        self.assertEqual(self.node._moss_reference_megapixels(2), 0.5)
        self.assertAlmostEqual(self.node._moss_reference_megapixels(3), 1 / 3)
        self.assertAlmostEqual(self.node._moss_reference_megapixels(4), 1 / 3)


if __name__ == "__main__":
    unittest.main()
