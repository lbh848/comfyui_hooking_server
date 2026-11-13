import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from illustration_json import (build_illustration_json, disable_moss_detailer_branch,
                               disable_moss_dialogue_repair_branch,
                               illustration_edit_document, illustration_edit_prompts,
                               is_moss_prompt, update_workflow_json_settings,
                               workflow_json_defaults)
from modes.moss_prompt_builder import (MossNpcPromptBuilder, MossPromptBuilder,
                                       build_controller, build_moss_layout,
                                       build_swap_text_layout, build_text_detailer,
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


def text_workflow_fixture():
    workflow = workflow_fixture()
    document = json.loads(workflow["1"]["inputs"]["value"])
    document["workflow"]["detailer"] = {
        "model": "moss_image_sft_int8_convrot.safetensors",
        "layout": "{}",
        "resolution": 1024,
        "seed": 0,
        "lora": "moss_i2i_swap_text_v4.safetensors",
        "strength": 1.0,
        "negative": "",
        "uncond_skip_layers": "22",
        "sampling": {"steps": 50, "cfg": 3.0, "sampler_name": "er_sde",
                     "scheduler": "normal", "denoise": 1.0, "shift": 5.0},
    }
    workflow["1"]["inputs"]["value"] = json.dumps(document, ensure_ascii=False)
    workflow["7"] = {"class_type": "SoyaUNETLoaderV2_moss_mdsoya",
                     "inputs": {"prompt_json": ["1", 0],
                                "model_key": "workflow.detailer.model"}}
    workflow["8"] = {"class_type": "SoyaMossSpectrumKSamplerV2_moss_mdsoya",
                     "inputs": {"prompt_json": ["1", 0],
                                "sampling_key": "workflow.detailer.sampling"}}
    return workflow


class TextDetailerTests(unittest.TestCase):
    LAYOUT = {
        "summary": {"caption": "a page", "meta": ["manga"], "canvas": "multi_panel"},
        "items": [
            {"label": "character", "name": "A", "caption": "a girl", "bbox": [-400, -400, 0, 0]},
            {"label": "text", "caption": "white balloon", "text": "안녕?",
             "bbox": [10, 20, 110, 220]},
            {"label": "text", "caption": "white balloon", "text": "Hi!",
             "bbox": [-200, 30, -100, 230]},
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
        "image_size": {"width": 768, "height": 1024},
    }

    def test_filters_text_items_and_rebuilds_summary(self):
        layout = build_swap_text_layout(self.LAYOUT)
        self.assertEqual(layout["summary"], {"caption": "comic, korean text"})
        self.assertEqual([item["text"] for item in layout["items"]], ["안녕?", "Hi!"])
        self.assertTrue(all(item["label"] == "text" for item in layout["items"]))
        self.assertTrue(all(item["caption"] == "text" for item in layout["items"]))
        self.assertNotIn("image_size", layout)
        self.assertEqual(layout["items"][0]["bbox"], [10, 20, 110, 220])

    def test_language_detection(self):
        english = {"items": [{"label": "text", "text": "Hello", "bbox": [0, 0, 1, 1]}]}
        self.assertEqual(build_swap_text_layout(english)["summary"]["caption"],
                         "comic, english text")

    def test_empty_layout_needs_allow_empty(self):
        empty = {"items": [{"label": "character", "caption": "girl", "bbox": [0, 0, 1, 1]}]}
        self.assertIsNone(build_swap_text_layout(empty))
        layout = build_swap_text_layout(empty, allow_empty=True)
        self.assertEqual(layout["items"], [])
        self.assertEqual(layout["summary"]["caption"], "comic, korean text")

    def test_controller_keeps_user_sampling(self):
        sampling = {"steps": 12, "cfg": 2.5, "sampler_name": "euler",
                    "scheduler": "simple", "denoise": 1.0, "shift": 5.0}
        detailer = build_text_detailer(self.LAYOUT, sampling=sampling,
                                       width=768, height=1024, seed=7,
                                       negative="low quality")
        self.assertEqual(detailer["resolution"], 880)
        self.assertEqual(detailer["seed"], 7)
        self.assertEqual(detailer["sampling"], sampling)
        self.assertEqual(detailer["lora"], "moss_i2i_swap_text_v4.safetensors")
        self.assertEqual(detailer["negative"], "low quality")
        self.assertEqual(json.loads(detailer["layout"])["items"][0]["text"], "안녕?")

    def test_controller_scales_only_the_detailer_resolution(self):
        sampling = {"steps": 1}
        scaled = build_text_detailer(self.LAYOUT, sampling=sampling,
                                     width=768, height=1024, resolution_scale=0.5)
        # resolution is an area: sqrt(768×1024) × 0.5 → 448, whose target_size
        # canvas is exactly half of 768×1024.
        self.assertEqual(scaled["resolution"], 448)
        # The whole canvas is resized: every mask keeps its relative position
        # and relative size, not its own centre.
        self.assertEqual([item["bbox"] for item in json.loads(scaled["layout"])["items"]],
                         [[5, 10, 55, 110], [-100, 15, -50, 115]])
        # The node never upscales past the first-pass canvas, so x2 keeps the
        # native size and the masks stay in place.
        enlarged = build_text_detailer(self.LAYOUT, sampling=sampling,
                                       width=768, height=1024, resolution_scale=2.0)
        self.assertEqual(enlarged["resolution"], 1776)
        self.assertEqual(json.loads(enlarged["layout"])["items"][0]["bbox"],
                         [10, 20, 110, 220])
        # Out-of-range scales fall back to the original page size.
        for scale in (3.0, 0.1, 0, "bad", float("nan")):
            with self.subTest(scale=scale):
                original = build_text_detailer(self.LAYOUT, sampling=sampling,
                                               width=768, height=1024,
                                               resolution_scale=scale)
                self.assertEqual(original["resolution"], 880)
                self.assertEqual(json.loads(original["layout"])["items"][0]["bbox"],
                                 [10, 20, 110, 220])

    def test_controller_matches_the_node_target_canvas(self):
        # 1728×1152 at x0.5: the detailer samples at 864×576, so every mask is
        # scaled by the actual target/source ratios instead of the raw x0.5.
        sampling = {"steps": 1}
        detailer = build_text_detailer(self.LAYOUT, sampling=sampling,
                                       width=1728, height=1152,
                                       resolution_scale=0.5)
        self.assertEqual(detailer["resolution"], 704)
        self.assertEqual([item["bbox"] for item in json.loads(detailer["layout"])["items"]],
                         [[5, 10, 55, 110], [-100, 15, -50, 115]])

    def test_bubble_toggle_reaches_the_document(self):
        defaults = workflow_json_defaults(text_workflow_fixture())
        layout = {"summary": {"caption": "page"},
                  "items": [{"label": "text", "caption": "balloon", "text": "안녕?",
                             "bbox": [10, 20, 110, 220]}]}
        for raw, expected in (("false", False), ("true", True)):
            with self.subTest(raw=raw):
                tagged = (f"[ANIMA_CONTENT]\npage\n[MOSS_BUBBLE_ACTIVATE]\n{raw}\n[END]")
                payload = json.loads(build_illustration_json(
                    tagged, "low quality", input_dir=".", workflow_defaults=defaults,
                    moss_layout=layout))
                self.assertIs(payload["workflow"]["detailer"]["activate"], expected)

    def test_missing_toggle_keeps_the_old_detailer_active(self):
        defaults = workflow_json_defaults(text_workflow_fixture())
        payload = json.loads(build_illustration_json(
            "[ANIMA_CONTENT]\npage\n[END]", "low quality", input_dir=".",
            workflow_defaults=defaults))
        self.assertTrue(payload["workflow"]["detailer"]["activate"])

    def test_bubble_scale_reaches_the_detailer_resolution(self):
        defaults = workflow_json_defaults(text_workflow_fixture())
        layout = {"summary": {"caption": "page"},
                  "items": [{"label": "text", "caption": "balloon", "text": "안녕?",
                             "bbox": [10, 20, 110, 220]}]}
        tagged = ("[ANIMA_CONTENT]\npage\n[IMG_W]\n1024\n[IMG_H]\n1024\n"
                  "[MOSS_BUBBLE_SCALE]\n0.5\n[END]")
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults,
            moss_layout=layout))
        self.assertEqual(payload["workflow"]["detailer"]["resolution"], 512)


class MangaPanelFrameTests(unittest.TestCase):
    def diversify(self, items, caption="Clean black borders and white gutters."):
        from modes.illustration_scene_pipeline import _diversify_manga_panel_frames
        return _diversify_manga_panel_frames(items, caption)

    def test_all_rectangular_page_gets_angled_frames(self):
        items = [
            {"label": "object", "caption": "PANEL 1. Rectangular manga panel with a thin black border. Close-up.",
             "bbox": [-100, -100, 0, 100]},
            {"label": "object", "caption": "PANEL 2. Close-up, from side.", "bbox": [0, -100, 100, 100]},
            {"label": "object", "caption": "PANEL 3. Rectangular manga panel. Upper body.",
             "bbox": [100, -100, 200, 100]},
        ]
        caption = self.diversify(items)
        self.assertIn("FRAME: trapezoid", items[0]["caption"])
        self.assertIn("FRAME: parallelogram", items[2]["caption"])
        self.assertIn("PANEL 2. Close-up, from side.", items[1]["caption"])
        self.assertIn("angled panels", caption)
        self.assertNotIn("white diagonal gutters", caption)

    def test_page_with_an_angled_panel_is_untouched(self):
        items = [
            {"label": "object", "caption": "PANEL 1. FRAME: trapezoid, slanted edge, thin border.",
             "bbox": [-100, -100, 0, 100]},
            {"label": "object", "caption": "PANEL 2. Rectangular panel.", "bbox": [0, -100, 100, 100]},
            {"label": "object", "caption": "PANEL 3. Rectangular panel.", "bbox": [100, -100, 200, 100]},
        ]
        self.assertEqual(self.diversify(items), "Clean black borders and white gutters.")

    def test_two_panel_page_is_left_alone(self):
        items = [
            {"label": "object", "caption": "PANEL 1. Rectangular panel.", "bbox": [-100, -100, 0, 100]},
            {"label": "object", "caption": "PANEL 2. Rectangular panel.", "bbox": [0, -100, 100, 100]},
        ]
        self.diversify(items)
        self.assertEqual(items[0]["caption"], "PANEL 1. Rectangular panel.")


class DetailerBranchTests(unittest.TestCase):
    def workflow(self):
        return {
            "18": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["16", 0], "vae": ["17", 0]}},
            "19": {"class_type": "PreviewImage", "inputs": {"images": ["35", 0]}},
            "31": {"class_type": "SoyaMOSSi2iReferenceV2_moss_mdsoya",
                   "inputs": {"model": ["30", 0], "vae": ["17", 0],
                              "image": ["18", 0], "prompt_json": ["1", 0]}},
            "34": {"class_type": "SoyaMossSpectrumKSamplerV2_moss_mdsoya",
                   "inputs": {}},
            "35": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["34", 0], "vae": ["17", 0]}},
        }

    def document(self, activate):
        return {"workflow": {"detailer": {"activate": activate}}}

    def test_disabled_detailer_points_output_at_the_first_pass(self):
        workflow = self.workflow()
        self.assertTrue(disable_moss_detailer_branch(workflow, self.document(False)))
        self.assertEqual(workflow["19"]["inputs"]["images"], ["18", 0])
        self.assertEqual(workflow["35"]["inputs"]["samples"], ["34", 0])

    def test_active_detailer_keeps_the_branch(self):
        workflow = self.workflow()
        self.assertFalse(disable_moss_detailer_branch(workflow, self.document(True)))
        self.assertEqual(workflow["19"]["inputs"]["images"], ["35", 0])

    def test_missing_detailer_is_a_no_op(self):
        workflow = self.workflow()
        self.assertFalse(disable_moss_detailer_branch(workflow, {"workflow": {}}))


class DialogueRepairBranchTests(unittest.TestCase):
    def workflow(self):
        return {
            "1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "{}"}},
            "2": {"class_type": "CLIPLoader", "inputs": {}},
            "36": {"class_type": "SoyaMossDialogueRepairV2_mdsoya",
                   "inputs": {"prompt_json": ["1", 0], "clip": ["2", 0]}},
            "14": {"class_type": "SoyaMossPromptEncodeV2_mdsoya",
                   "inputs": {"prompt_json": ["36", 0], "clip": ["2", 0]}},
            "17": {"class_type": "SoyaKSamplerV2_moss_mdsoya",
                   "inputs": {"prompt_json": ["36", 0]}},
        }

    def document(self, enabled=True, activate=True):
        return {"workflow": {
            "dialogue_repair": {"enabled": enabled, "anchor": "C017", "source": "src"},
            "detailer": {"activate": activate},
        }}

    def test_disabled_repair_bypasses_the_node(self):
        workflow = self.workflow()
        self.assertTrue(disable_moss_dialogue_repair_branch(
            workflow, self.document(enabled=False)))
        self.assertEqual(workflow["14"]["inputs"]["prompt_json"], ["1", 0])
        self.assertEqual(workflow["17"]["inputs"]["prompt_json"], ["1", 0])

    def test_inactive_detailer_also_bypasses_the_node(self):
        workflow = self.workflow()
        self.assertTrue(disable_moss_dialogue_repair_branch(
            workflow, self.document(activate=False)))
        self.assertEqual(workflow["14"]["inputs"]["prompt_json"], ["1", 0])

    def test_enabled_repair_with_active_detailer_keeps_the_node(self):
        workflow = self.workflow()
        self.assertFalse(disable_moss_dialogue_repair_branch(workflow, self.document()))
        self.assertEqual(workflow["14"]["inputs"]["prompt_json"], ["36", 0])

    def test_missing_repair_node_is_a_no_op(self):
        workflow = {"1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "{}"}}}
        self.assertFalse(disable_moss_dialogue_repair_branch(workflow, self.document()))


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


class MangaSharedDividerTests(unittest.TestCase):
    def test_horizontal_and_vertical_shared_edges_are_emitted_once(self):
        from modes.illustration_scene_pipeline import _manga_shared_dividers
        panels = [
            {"shape": "regular", "bbox": [-100, -100, 100, 0]},
            {"shape": "regular", "bbox": [-100, 0, 0, 100]},
            {"shape": "regular", "bbox": [0, 0, 100, 100]},
        ]
        lines = _manga_shared_dividers(panels, 200, 200)
        self.assertEqual({tuple(item["bbox"]) for item in lines},
                         {(-100, -1, 0, 1), (0, -1, 100, 1), (-1, 0, 1, 100)})
        self.assertEqual(len(lines), 3)
        self.assertEqual(panels[0]["bbox"], [-100, -100, 100, 0])

    def test_gaps_slanted_panels_and_insets_do_not_get_invented_edges(self):
        from modes.illustration_scene_pipeline import _manga_shared_dividers
        first = {"shape": "regular", "bbox": [-100, -100, -1, 100]}
        for second in [
            {"shape": "regular", "bbox": [1, -100, 100, 100]},
            {"shape": "diagonal", "bbox": [-1, -100, 100, 100]},
            {"shape": "inset", "bbox": [-1, -100, 100, 100]},
        ]:
            with self.subTest(second=second):
                self.assertEqual(_manga_shared_dividers([first, second], 200, 200), [])

    def test_divider_negative_is_scoped_idempotent_and_keeps_input_unchanged(self):
        from manga_page_vocabulary import SHARED_DIVIDER_PREFIX
        controller = {"negative": "low quality, white gutters"}
        plain = {"items": [{"label": "object", "caption": "PANEL 1. ordinary frame"}]}
        self.assertEqual(MossPromptBuilder.build(plain, controller)["workflow"], controller)
        joined = {"items": [{"label": "object", "caption": SHARED_DIVIDER_PREFIX + " horizontal ink rule separating views"}]}
        first = MossPromptBuilder.build(joined, controller)
        self.assertIn("double panel borders", first["workflow"]["negative"])
        self.assertEqual(first["workflow"]["negative"].count("white gutters"), 1)
        self.assertEqual(controller["negative"], "low quality, white gutters")
        self.assertEqual(MossPromptBuilder.build(joined, first["workflow"]), first)


class DetailMangaMossTests(unittest.TestCase):
    """DETAIL-MANGA: one MOSS multi_panel page with panel objects and balloons."""

    def scene(self):
        return {"slot": 1, "anchor_segment": "C001",
                "characters": [{"name": "A"}],
                "image_size": {"width": 1024, "height": 1536}}

    def test_page_contract_sends_page_size_and_closed_vocabulary(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga",
                     "speak_language": "한국어"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)})
        user = messages[1]["content"]
        self.assertIn("hierarchy", user)
        self.assertIn("dominant_panel", user)
        self.assertIn("touching", user)
        self.assertIn("breaks_border", user)
        self.assertIn("page_size_options", user)
        self.assertIn("[1152, 1728]", user)
        self.assertIn("한국어", user)
        self.assertNotIn("layout_options", user)
        self.assertNotIn("meta:", user)
        # The server must not pre-pick a page size for the model.
        self.assertNotIn("layout_hint", user)
        self.assertNotIn('"image_size": {"width"', user)
        self.assertNotIn('"width": 768', user)

    def test_side_view_is_offered_for_junction_and_cross_section_panels(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        from manga_page_vocabulary import PANEL_CAMERA_ANGLE
        self.assertIn("side_view", PANEL_CAMERA_ANGLE)
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)})
        self.assertIn("side_view", messages[1]["content"])

    def test_page_contract_uses_layered_schema(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768)})
        user = messages[1]["content"]
        self.assertIn("panelId", user)
        self.assertIn("attachedTo", user)
        self.assertIn("spansPanels", user)
        self.assertIn("cross_panel_actors", user)
        self.assertIn("actors", user)
        self.assertIn("dialogue", user)

    def test_composition_hint_toggle_controls_conditional_rendering(self):
        from modes.illustration_context_pipeline import merged_toggles
        from modes.illustration_prompt_assembly import variable
        from modes.illustration_prompt_builders import detail_moss_messages
        shared = variable("base_payload", "shared")
        hint = "hint_payload"
        prompt = (shared + "{{#when::lb-xnai.manga_composition_hint::tis::1}}"
                  + hint + "{{/when}}")
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                toggles = merged_toggles({"illustration_model_family": "moss",
                    "illustration_output_mode": "manga",
                    "manga_composition_hint_enabled": "on" if enabled else "off"})
                self.assertIs(toggles["manga_composition_hint_enabled"], enabled)
                messages = detail_moss_messages(prompt, [self.scene()], toggles=toggles)
                system = messages[0]["content"]
                self.assertEqual(system, shared + hint if enabled else shared)
                self.assertIn(shared.variable_spans[0], system.variable_spans)
        self.assertFalse(merged_toggles({})["manga_composition_hint_enabled"])

    def test_parser_keeps_single_panel_and_bleed_character_bbox(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1536},
            "summary": {"canvas": "multi_panel", "caption": "one page"},
            "items": [
                {"label": "object",
                 "caption": "PANEL 1. FRAME: rectangular, thin black border. Close-up.",
                 "bbox": [-700, -480, 100, 480]},
                {"label": "character", "name": "A",
                 "caption": "PANEL 1. 1girl, hair and shoulder break past the frame into the gutter.",
                 "bbox": [-760, -500, 220, 500]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }]}
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.scene()], moss=True)[0]
        items = descriptor["moss_layout"]["items"]
        self.assertEqual([item["label"] for item in items], ["object", "character"])
        # A bleed body keeps its full bbox instead of being clipped to the panel.
        self.assertEqual(items[1]["bbox"], [-760, -500, 220, 500])

    def layered_scene(self):
        return {"slot": 1, "anchor_segment": "C001",
                "characters": [{"name": "A"}, {"name": "B"}],
                "image_size": {"width": 1024, "height": 1536}}

    def layered_response(self, *, duplicate_cross_in_panel=False):
        panel_one_actors = [
            {"actorId": "A", "bbox": [-700, -400, -80, 400],
             "instruction": "1girl, long hair, grin, upper body."},
        ]
        if duplicate_cross_in_panel:
            panel_one_actors.append(
                {"actorId": "C", "bbox": [-740, -480, 300, 480],
                 "instruction": "1girl, silver hair, leaning in."})
        return {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1536},
            "page": {"hierarchy": "asymmetric", "flow": "diagonal",
                     "boundaries": "touching",
                     "instruction": "Dominant top, reaction below.",
                     "caption": "page style"},
            "panels": [
                {"panelId": "top", "bbox": [-768, -512, -16, 512],
                 "attachedTo": [{"panelId": "bottom", "edge": "bottom"}],
                 "shape": "diagonal", "camera_distance": "close_up",
                 "camera_angle": "eye_level", "focus": "shallow_focus",
                 "visibleActorIds": ["A"], "instruction": "A leans in.",
                 "actors": panel_one_actors, "objects": [],
                 "dialogue": [{"speaker": "A", "text": "안녕",
                               "bbox": [-700, 40, -600, 380],
                               "balloon": "oval", "tail": "down"}]},
                {"panelId": "bottom", "bbox": [-16, -512, 768, 512],
                 "attachedTo": [{"panelId": "top", "edge": "top"}],
                 "shape": "irregular", "camera_distance": "medium",
                 "camera_angle": "high_angle", "focus": "deep_focus",
                 "visibleActorIds": ["B"], "instruction": "B watches.",
                 "actors": [{"actorId": "B", "bbox": [100, -100, 600, 400],
                             "instruction": "1boy, dark hair, cold stare, upper body."}],
                 "objects": [], "dialogue": []},
            ],
            "cross_panel_actors": [{
                "actorId": "C", "spansPanels": ["top", "bottom"],
                "relation": "breaks_border", "bbox": [-760, -500, 220, 500],
                "instruction": "Hair crosses the gutter."}],
        }]}

    def test_parser_flattens_layered_page_into_moss_items(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        text = "<lb-xnai>" + json.dumps(self.layered_response(), ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]
        layout = descriptor["moss_layout"]
        self.assertNotIn("canvas", layout["summary"])
        self.assertIn("page style", layout["summary"]["caption"])
        self.assertIn("Dominant top, reaction below.", layout["summary"]["caption"])
        self.assertIn("abutting borderless image regions", layout["summary"]["caption"])
        items = layout["items"]
        self.assertEqual([item["label"] for item in items],
                         ["object", "character", "text", "object", "character", "character"])
        self.assertTrue(items[0]["caption"].startswith(
            "PANEL 1. REGION: trapezoid with a slanted edge, edge-to-edge artwork"))
        self.assertIn("Shares one black dividing stroke with PANEL 2 at the bottom edge", items[0]["caption"])
        self.assertNotIn("thin black border", items[0]["caption"])
        self.assertTrue(items[1]["caption"].startswith("PANEL 1. 1girl"))
        self.assertIn("tail pointing down", items[2]["caption"])
        self.assertIn("Speaker: A.", items[2]["caption"])
        self.assertIn("horizontal left-to-right text", items[2]["caption"])
        self.assertEqual(items[2]["text"], "안녕")
        self.assertTrue(items[5]["caption"].startswith(
            "One continuous body crossing PANEL 1 and PANEL 2; the shared dividing line "
            "passes behind the body."))
        self.assertEqual(items[5]["bbox"], [-760, -500, 220, 500])
        self.assertEqual([item["name"] for item in descriptor["characters"]],
                         ["A", "B", "C"])
        self.assertEqual(descriptor["comic_page"]["panels"][0]["panelId"], "top")

    def test_numeric_composition_summary_preserves_the_page_image_description(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response()
        response["scenes"][0]["summary"] = 42
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]
        caption = descriptor["moss_layout"]["summary"]["caption"]
        self.assertIn("page style", caption)
        self.assertIn("Dominant top, reaction below.", caption)
        self.assertNotIn("42", caption)

    def test_cross_panel_actor_is_not_duplicated_in_panel_items(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response(duplicate_cross_in_panel=True)
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]
        cross = [item for item in descriptor["moss_layout"]["items"]
                 if item.get("name") == "C"]
        self.assertEqual(len(cross), 1)
        self.assertTrue(cross[0]["caption"].startswith("One continuous body crossing PANEL 1 and PANEL 2"))

    def test_cross_panel_actor_keeps_a_separate_unlinked_reaction(self):
        from copy import deepcopy
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response(duplicate_cross_in_panel=True)
        panel = deepcopy(response["scenes"][0]["panels"][1])
        panel.update(panelId="inset", shape="inset", attachedTo=[],
                     bbox=[300, -500, 700, -120], visibleActorIds=["C"],
                     actors=[{"actorId": "C", "bbox": [320, -480, 680, -140],
                              "instruction": "1girl, silver hair, surprised, close-up."}])
        response["scenes"][0]["panels"].append(panel)
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        layout = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]["moss_layout"]
        bodies = [item for item in layout["items"] if item.get("name") == "C"]
        self.assertEqual(len(bodies), 2)
        self.assertTrue(bodies[0]["caption"].startswith("PANEL 3."))
        self.assertEqual(bodies[0]["bbox"], [320, -480, 680, -140])
        self.assertEqual(bodies[1]["bbox"], [-760, -500, 220, 500])

    def test_layered_shared_edges_keep_authored_geometry_and_shapes(self):
        from copy import deepcopy
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response()
        scene = response["scenes"][0]
        template = scene["panels"][0]
        scene["panels"] = []
        scene["cross_panel_actors"] = []
        boxes = [[-768, -512, -256, 512], [-256, -512, 256, 512],
                 [256, -512, 768, 512]]
        for i, bbox in enumerate(boxes):
            panel = deepcopy(template)
            links = []
            if i:
                links.append({"panelId": f"shot_{i - 1}", "edge": "top"})
            if i < 2:
                links.append({"panelId": f"shot_{i + 1}", "edge": "bottom"})
            panel.update(panelId=f"shot_{i}", bbox=bbox, shape="regular", attachedTo=links,
                         actors=[], dialogue=[], visibleActorIds=[])
            scene["panels"].append(panel)
        scene["page"]["instruction"] = "Three equal views for a deliberate pause."
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        layout = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]["moss_layout"]
        self.assertIn(scene["page"]["instruction"], layout["summary"]["caption"])
        panel_items = [item for item in layout["items"] if item["caption"].startswith("PANEL ")]
        self.assertEqual([item["bbox"] for item in panel_items], boxes)
        self.assertTrue(all("REGION: rectangular" in item["caption"] for item in panel_items))
        self.assertTrue(all("thin black border" not in item["caption"] for item in panel_items))
        dividers = [item for item in layout["items"] if " ink rule separating " in item["caption"]]
        self.assertEqual(len(dividers), 2)
        self.assertNotIn("white diagonal gutters", layout["summary"]["caption"])

    def test_explicit_separated_layout_keeps_its_gutters(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response()
        scene = response["scenes"][0]
        scene["page"]["boundaries"] = "separated"
        scene["panels"][0]["bbox"][2] = -32
        scene["panels"][1]["bbox"][0] = 32
        for panel in scene["panels"]:
            panel["attachedTo"] = []
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        layout = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]["moss_layout"]
        self.assertIn("deliberate white gutters", layout["summary"]["caption"])
        self.assertNotIn("edge to edge", layout["summary"]["caption"])
        self.assertIn("thin black border", layout["items"][0]["caption"])
        self.assertEqual(layout["items"][0]["bbox"][2], -32)

    def test_touching_page_preserves_inset_frame_without_boxing_main_panels(self):
        from copy import deepcopy
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = self.layered_response()
        scene = response["scenes"][0]
        inset = deepcopy(scene["panels"][0])
        inset.update(panelId="detail", shape="inset", bbox=[300, 200, 600, 450],
                     attachedTo=[], actors=[], dialogue=[], visibleActorIds=[])
        scene["panels"].append(inset)
        text = "<lb-xnai>" + json.dumps(response) + "</lb-xnai>"
        layout = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]["moss_layout"]
        panels = [item for item in layout["items"] if item["label"] == "object"]
        self.assertEqual(len(panels), 3)
        for panel in panels[:2]:
            self.assertIn("REGION:", panel["caption"])
            self.assertNotIn("thin black border", panel["caption"])
        self.assertIn("FRAME: inset rectangular frame, thin black border", panels[2]["caption"])
        self.assertEqual(panels[2]["bbox"], [300, 200, 600, 450])

    def test_layered_page_accepts_single_panel(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1536},
            "page": {"caption": "one panel page"},
            "panels": [{
                "panelId": "only", "bbox": [-768, -512, 768, 512],
                "shape": "borderless", "camera_distance": "medium",
                "camera_angle": "eye_level", "focus": "deep_focus",
                "visibleActorIds": ["A"], "instruction": "A looks up.",
                "actors": [{"actorId": "A", "bbox": [-600, -400, 600, 400],
                            "instruction": "1girl, long hair, looking up, upper body."}],
                "objects": [], "dialogue": [{
                    "speaker": "A", "text": "응?", "bbox": [-500, 100, -350, 500],
                    "balloon": "oval", "tail": "down"}]}],
            "cross_panel_actors": [],
        }]}
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.layered_scene()], moss=True)[0]
        items = descriptor["moss_layout"]["items"]
        self.assertEqual([item["label"] for item in items], ["object", "character", "text"])
        self.assertIn("borderless, no border, no gutter", items[0]["caption"])
        self.assertEqual(items[2]["text"], "응?")
        self.assertIn("tail pointing down", items[2]["caption"])
        self.assertEqual(descriptor["moss_layout"]["summary"]["caption"], "one panel page")

    def test_layered_page_rejects_out_of_range_panel_count(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        panels = [{
            "panelId": f"p{index}", "bbox": [-768, -512, 768, 512],
            "shape": "regular", "camera_distance": "medium",
            "camera_angle": "eye_level", "focus": "deep_focus",
            "visibleActorIds": [], "instruction": "empty panel",
            "actors": [], "objects": [], "dialogue": [],
        } for index in range(1, 7)]
        response = {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1536},
            "page": {"caption": "too many"},
            "panels": panels, "cross_panel_actors": [],
        }]}
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        with self.assertRaises(ValueError):
            parse_detail_batch(text, [self.layered_scene()], moss=True)

    def test_manga_scale_multiplies_page_size_options(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768),
                          "manga_scale": 0.5})
        user = messages[1]["content"]
        # Every offered resolution is multiplied before the model picks one.
        self.assertIn("[576, 864]", user)
        self.assertIn("[528, 944]", user)
        self.assertNotIn("[1152, 1728]", user)

    def test_selected_page_sizes_narrow_the_offered_list(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768),
                          "manga_page_sizes": "1152x1728,1056x1888"})
        user = messages[1]["content"]
        # Only the resolutions selected in the bot settings are offered.
        self.assertIn("[1152, 1728]", user)
        self.assertIn("[1056, 1888]", user)
        self.assertNotIn("[1408, 1408]", user)
        self.assertNotIn("[1728, 1152]", user)

    def test_empty_page_size_selection_offers_every_default(self):
        from modes.illustration_prompt_builders import detail_moss_messages
        messages = detail_moss_messages(
            "MOSS MANGA", [self.scene()],
            toggles={"illustration_model_family": "moss",
                     "illustration_output_mode": "manga"},
            canvas_sizes={"solo": (768, 1024), "group": (1152, 768),
                          "manga_page_sizes": []})
        user = messages[1]["content"]
        self.assertIn("[1152, 1728]", user)
        self.assertIn("[1728, 1152]", user)

    def test_parser_keeps_panel_objects_text_and_repeated_character_boxes(self):
        from modes.illustration_scene_pipeline import parse_detail_batch
        response = {"scenes": [{
            "slot": 1, "image_size": {"width": 1024, "height": 1536},
            "summary": {"canvas": "multi_panel",
                        "caption": "page style"},
            "items": [
                {"label": "object", "caption": "PANEL 1. frame",
                 "bbox": [-768, -512, -24, 512]},
                {"label": "character", "name": "A", "caption": "PANEL 1. girl",
                 "bbox": [-600, -400, -100, 0]},
                {"label": "text", "caption": "PANEL 1. oval balloon",
                 "text": "안녕", "bbox": [-600, 50, -500, 400]},
                {"label": "object", "caption": "PANEL 2. frame",
                 "bbox": [24, -512, 768, 512]},
                {"label": "character", "name": "A", "caption": "PANEL 2. girl smiles",
                 "bbox": [100, -100, 600, 400]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }]}
        text = "<lb-xnai>" + json.dumps(response, ensure_ascii=False) + "</lb-xnai>"
        descriptor = parse_detail_batch(text, [self.scene()], moss=True)[0]
        layout = descriptor["moss_layout"]
        self.assertEqual(layout["summary"], {
            "caption": "page style",
            "meta": ["comic panel redraw", "full color manga", "nsfw"]})
        self.assertEqual([item["label"] for item in layout["items"]],
                         ["object", "character", "text", "object", "character"])
        self.assertEqual(layout["items"][2]["text"], "안녕")
        # The same canonical person repeats once per panel without losing a box.
        self.assertEqual([item["bbox"] for item in layout["items"] if item["label"] == "character"],
                         [[-600, -400, -100, 0], [100, -100, 600, 400]])
        self.assertEqual([item["name"] for item in descriptor["characters"]], ["A"])
        self.assertEqual(descriptor["canvas"], [1024, 1536])

    def test_build_moss_layout_reduces_meta_to_manga(self):
        layout = {
            "summary": {"canvas": "multi_panel",
                        "caption": "page style"},
            "items": [
                {"label": "object", "caption": "PANEL 1. frame", "bbox": [-10, -10, 10, 10]},
                {"label": "text", "caption": "balloon", "text": "hi",
                 "bbox": [-5, -5, 0, 0]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
            "image_size": {"width": 1024, "height": 1536},
        }
        moss = build_moss_layout(layout, image_width=1024, image_height=1536)
        self.assertEqual(moss["summary"], {
            "caption": "page style",
            "meta": ["comic panel redraw", "full color manga", "nsfw"],
            "canvas": "multi_panel"})
        self.assertEqual(moss["items"][1], {"label": "text", "caption": "balloon",
                                            "text": "hi", "bbox": [-5, -5, 0, 0]})
        self.assertEqual(moss["image_size"], {"width": 1024, "height": 1536})

    def test_manga_plan_is_disabled_for_moss_manga(self):
        from modes.illustration_context_pipeline import merged_toggles
        toggles = merged_toggles({
            "illustration_model_family": "moss",
            "illustration_output_mode": "manga",
            "call2_plan_enabled": True,
        }, for_generation=True)
        self.assertFalse(toggles["call2_plan_enabled"])
        # Ordinary MOSS illustration still honours the planner setting.
        ordinary = merged_toggles({
            "illustration_model_family": "moss",
            "illustration_output_mode": "illustration",
            "call2_plan_enabled": True,
        }, for_generation=True)
        self.assertTrue(ordinary["call2_plan_enabled"])


class LoraAndControllerTests(unittest.TestCase):
    def test_same_lora_keeps_distinct_bbox_owners(self):
        entries = [{"lora_path": "shared.safetensors", "CHAR": "Alice", "str": .7},
                   {"lora_path": "shared.safetensors", "CHAR": "Bob", "str": .5},
                   {"lora_path": "shared.safetensors", "CHAR": "Alice", "str": .9}]
        merged = merge_lora_lists(entries, [{"lora_path": "shared.safetensors", "str": 1.}])
        self.assertEqual([(item["CHAR"], item["str"]) for item in merged], [("Alice", .7), ("Bob", .5)])

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

    def test_text_detailer_uses_only_balloon_items(self):
        defaults = workflow_json_defaults(text_workflow_fixture())
        self.assertIn("workflow.detailer.sampling", defaults)
        tagged = ("[ANIMA_CONTENT]\npage\n[IMG_W]\n768\n[IMG_H]\n1024\n"
                  "[SEED]\n42\n[END]")
        layout = {
            "summary": {"caption": "page", "canvas": "multi_panel"},
            "items": [
                {"label": "character", "name": "A", "caption": "girl",
                 "bbox": [-400, -400, 0, 0]},
                {"label": "text", "caption": "balloon", "text": "안녕?",
                 "bbox": [10, 20, 110, 220]},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        }
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults,
            moss_layout=layout))
        detailer = payload["workflow"]["detailer"]
        # resolution is the node's area cap (sqrt(768×1024) rounded to 16).
        self.assertEqual(detailer["resolution"], 880)
        self.assertEqual(detailer["seed"], 42)
        self.assertEqual(detailer["model"], defaults["workflow.model"])
        self.assertEqual(detailer["sampling"], defaults["workflow.detailer.sampling"])
        self.assertEqual(detailer["negative"], "low quality")
        text = json.loads(detailer["layout"])
        self.assertEqual(text["summary"], {"caption": "comic, korean text"})
        self.assertEqual([item["text"] for item in text["items"]], ["안녕?"])

    def test_text_detailer_is_present_without_dialogue(self):
        defaults = workflow_json_defaults(text_workflow_fixture())
        tagged = "[ANIMA_CONTENT]\nlandscape\n[END]"
        payload = json.loads(build_illustration_json(
            tagged, "low quality", input_dir=".", workflow_defaults=defaults))
        detailer = payload["workflow"]["detailer"]
        self.assertEqual(json.loads(detailer["layout"])["items"], [])
        self.assertEqual(detailer["sampling"], defaults["workflow.detailer.sampling"])

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

    def test_duplicate_character_reuses_one_reference(self):
        self.layout = {"composition_prompt": "two panels", "regions": [
            {"name": "Alice", "character_prompt": "silver hair", "x": 0, "y": 0,
             "width": 0.5, "height": 1},
            {"name": "Alice", "character_prompt": "silver hair, close-up",
             "x": 0.5, "y": 0, "width": 0.5, "height": 0.5},
            {"name": "Bob", "character_prompt": "blue coat", "x": 0, "y": 0.5,
             "width": 0.5, "height": 0.5},
        ]}
        reference = self.build()["workflow"]["reference"]
        self.assertEqual([item["name"] for item in reference["pictures"]],
                         ["Alice", "Bob"])
        self.assertEqual([item["picture"] for item in reference["pictures"]], [1, 2])

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
                self.assertEqual(len(unet["outputs"][0]["links"]), 2)
                patch_link = document["links"][next(
                    i for i, link in enumerate(document["links"]) if link[0] == 10)]
                self.assertEqual(patch_link[3], next(
                    node["id"] for node in document["nodes"]
                    if node["type"] == "SoyaMOSSEditModelPatch_mdsoya"))
                reference = next(node for node in document["nodes"]
                                 if node["type"] == "SoyaMOSSi2iReferenceV2_moss_mdsoya")
                model_input = next(entry for entry in reference["inputs"]
                                   if entry["name"] == "model")
                self.assertEqual(
                    next(link for link in document["links"] if link[0] == model_input["link"])[1],
                    10)
                positive = by_id[14]
                self.assertEqual(positive["type"], "SoyaMossPromptEncodeV2_mdsoya")
                self.assertEqual([entry["name"] for entry in positive["inputs"]],
                                 ["clip", "prompt_json", "vae", "image1", "image2", "image3"])
                for entry in positive["inputs"]:
                    self.assertIsNotNone(entry["link"])
                negative = by_id[15]
                self.assertEqual(negative["type"], "SoyaMossNegativeEncodeV2_mdsoya")

    def test_moss_presets_expose_text_detailer(self):
        presets = json.loads((Path(__file__).resolve().parents[1] / "presets"
                              / "illustration_model_presets.json").read_text(encoding="utf-8"))
        for name in ("공방-MOSS-8step", "공방-MOSS-4step"):
            with self.subTest(name=name):
                preset = next(item for item in presets["presets"] if item["name"] == name)
                fields = preset["workflows"]["v3_anima"]["fields"]
                self.assertNotIn("workflow.detailer.model", fields)
                self.assertIn("workflow.detailer.sampling", fields)
                path = preset["config"]["illustration_workflow_source_paths"]["v3_anima"]
                self.assertIn("MOSS_", path)
                self.assertNotIn("_TEXT_", path)

    def test_text_detailer_chain_is_wired(self):
        specs = json.loads((Path(__file__).resolve().parents[1] / "comfy_installer"
                            / "illustration_v2_specs.json").read_text(encoding="utf-8"))
        for name in ("SoyaMOSSi2iReferenceV2_moss_mdsoya",
                     "SoyaMossSpectrumKSamplerV2_moss_mdsoya",
                     "SoyaCLIPTextEncodeV2_moss_detailer_negative_mdsoya"):
            self.assertIn(name, specs)
        for filename in ("배포_공방_삽화_MOSS_8STEP_V1.json",
                         "배포_공방_삽화_MOSS_4STEP_V1.json"):
            with self.subTest(filename=filename):
                document = self.read_workflow(filename)
                by_id = {node["id"]: node for node in document["nodes"]}
                types = [node["type"] for node in document["nodes"]]
                for expected in ("SoyaMOSSi2iReferenceV2_moss_mdsoya",
                                 "SoyaMossSwapTextEncodeV2_mdsoya",
                                 "SoyaMossSpectrumKSamplerV2_moss_mdsoya",
                                 "SoyaCLIPTextEncodeV2_moss_detailer_negative_mdsoya"):
                    self.assertIn(expected, types)
                self.assertEqual(document["extra"]["soya_moss_text_detailer"], 1)
                prompt = json.loads(by_id[1]["widgets_values"][0])
                detailer = prompt["workflow"]["detailer"]
                self.assertEqual(detailer["model"], prompt["workflow"]["model"])
                self.assertEqual(detailer["lora"],
                                 "moss_i2i_swap_text_v4.safetensors")
                self.assertIn("sampling", detailer)
                reference = next(node for node in document["nodes"]
                                 if node["type"] == "SoyaMOSSi2iReferenceV2_moss_mdsoya")
                self.assertEqual([entry["name"] for entry in reference["outputs"]],
                                 ["MODEL", "LATENT"])
                sampler = next(node for node in document["nodes"]
                               if node["type"] == "SoyaMossSpectrumKSamplerV2_moss_mdsoya")
                self.assertEqual(sampler["widgets_values"], ["workflow.detailer.sampling"])
                preview = by_id[19]
                preview_link = next(link for link in document["links"]
                                    if link[0] == preview["inputs"][0]["link"])
                final_decode = by_id[preview_link[1]]
                self.assertEqual(final_decode["type"], "VAEDecode")
                decode_link = next(link for link in document["links"]
                                   if link[0] == final_decode["inputs"][0]["link"])
                self.assertEqual(by_id[decode_link[1]]["type"],
                                 "SoyaMossSpectrumKSamplerV2_moss_mdsoya")


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
