"""Regional asset prompts keep box ownership, text and references aligned."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from asset_json import embed_asset_mask_references, prepare_asset_mask_document, prepare_asset_prompt
from illustration_json import build_illustration_json
from tests import test_asset_style_presets as style_tests


def document_fixture():
    return {"schema": "soya.illustration.v2", "assets": {}, "fields": {
        "IMG_W": 704, "IMG_H": 1024, "SEED": -1, "ANIMA_CONTENT": "whole scene",
        "SNAPREF": {"enabled": False, "references": []},
        "STYLE_LORA_DATA": {"list": [{"BASE": "anima", "trigger": "shared style"}]},
        "MULTI_CHAR": {"enable": True, "char_num": 2, "char_name_list": ["right", "left"],
                       "char_inform": ["red jacket", "blue dress"], "char_trigger_list": [["right trigger"], ["left trigger"]],
                       "background_trigger_list": [], "background_prompt": "garden", "composition_prompt": "standing together",
                       "shared_tag": {"before_char": ["watercolor", "garden"], "after_char": []},
                       "layout": {"regions": [{"name": "right", "x": .55, "y": .1, "width": .4, "height": .8},
                                              {"name": "left", "x": .05, "y": .1, "width": .4, "height": .8}]}},
    }}


class AssetMaskDocumentTests(unittest.TestCase):
    def test_reverse_order_boxes_keep_prompt_and_rgb_ownership_through_transport(self):
        document = prepare_asset_mask_document(document_fixture())
        multi = document["fields"]["MULTI_CHAR"]
        self.assertEqual(multi["char_name_list"], ["left", "right"])
        self.assertEqual(multi["char_inform"], ["blue dress", "red jacket"])
        self.assertEqual(multi["char_trigger_list"], [["left trigger", "shared style"], ["right trigger", "shared style"]])
        self.assertEqual(len(multi["mask_fingerprint"]), 64)
        self.assertEqual(document["fields"]["CHAR_LIST"], ["left", "right"])
        asset = document["assets"][document["fields"]["REGION_MASK"]]
        raw = base64.b64decode(asset["data"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), asset["sha256"])
        with Image.open(io.BytesIO(raw)) as mask:
            self.assertEqual(mask.getpixel((100, 500)), (255, 0, 0))
            self.assertEqual(mask.getpixel((600, 500)), (0, 255, 0))
            self.assertEqual(mask.getpixel((0, 0)), (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            prompt = prepare_asset_prompt(json.dumps(document), "")
            delivered = json.loads(build_illustration_json(prompt, "", input_dir=directory, include_face_embeddings=False))
        self.assertEqual(delivered["fields"]["MULTI_CHAR"], multi)
        self.assertEqual(delivered["fields"]["REGION_MASK"], document["fields"]["REGION_MASK"])
        self.assertEqual(delivered["assets"][asset["sha256"]], asset)
        self.assertGreaterEqual(delivered["fields"]["SEED"], 0)

    def test_single_person_position_control_remains_enabled(self):
        document = document_fixture()
        multi = document["fields"]["MULTI_CHAR"]
        multi.update(char_num=1, char_name_list=["right"], char_inform=["red jacket"], char_trigger_list=[[]])
        multi["layout"]["regions"] = multi["layout"]["regions"][:1]
        result = prepare_asset_mask_document(document)
        self.assertTrue(result["fields"]["MULTI_CHAR"]["enable"])
        self.assertEqual(result["fields"]["MULTI_CHAR"]["layout"]["regions"][0]["x"], .55)

    def test_regional_composition_follows_rgb_order_and_survives_transport(self):
        document = document_fixture()
        multi = document["fields"]["MULTI_CHAR"]
        multi["char_composition_list"] = ["The 1boy on the right", "The 1girl on the left"]
        multi["composition_prompt"] = ", ".join(multi["char_composition_list"])
        result = prepare_asset_mask_document(document)
        self.assertEqual(multi["char_composition_list"], ["The 1girl on the left", "The 1boy on the right"])
        with tempfile.TemporaryDirectory() as directory:
            delivered = json.loads(build_illustration_json(prepare_asset_prompt(json.dumps(result), ""), "",
                                   input_dir=directory, include_face_embeddings=False))
        self.assertEqual(delivered["fields"]["MULTI_CHAR"], multi)

    def test_invalid_regional_composition_fails_before_queueing(self):
        for values in (None, "wrong", [], ["only one"], ["left", None], ["left", 1]):
            document = document_fixture()
            document["fields"]["MULTI_CHAR"]["char_composition_list"] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "마스크별 자연어"):
                prepare_asset_mask_document(document)

    def test_optional_natural_language_and_natural_language_only_regions(self):
        document = document_fixture()
        multi = document["fields"]["MULTI_CHAR"]
        multi.update(composition_prompt="", char_composition_list=["", ""])
        prepare_asset_mask_document(document)
        self.assertEqual(multi["char_composition_list"], ["", ""])
        multi.update(char_inform=["", ""], char_composition_list=["reading a book", "waving"])
        prepare_asset_mask_document(document)
        self.assertEqual(multi["char_composition_list"], ["reading a book", "waving"])

    def test_invalid_boxes_and_mismatched_prompts_fail_before_queueing(self):
        for patch in ({"x": -.1}, {"x": .8}, {"width": 0}, {"height": float("nan")}, {"x": True}, {"name": "unknown"}):
            document = document_fixture()
            document["fields"]["MULTI_CHAR"]["layout"]["regions"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                prepare_asset_mask_document(document)
        document = document_fixture()
        document["fields"]["MULTI_CHAR"]["char_inform"] = ["missing second"]
        with self.assertRaises(ValueError):
            prepare_asset_mask_document(document)

    def test_references_are_embedded_per_character_and_preserved(self):
        document = prepare_asset_mask_document(document_fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.new("RGB", (16, 16), "red").save(path)
            text = embed_asset_mask_references(json.dumps(document), [{"CHAR": "right", "images": [{"local_path": str(path)}]}])
            result = json.loads(prepare_asset_prompt(text, "", snapref_enabled=True))
            self.assertEqual(result["fields"]["SNAPREF"]["references"][0]["CHAR"], "right")
            self.assertTrue(result["fields"]["SNAPREF"]["enabled"])
            delivered = json.loads(build_illustration_json(json.dumps(result), "", input_dir=directory, include_face_embeddings=False))
            self.assertEqual(delivered["fields"]["SNAPREF"]["references"][0]["CHAR"], "right")


class AssetMaskApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = style_tests.AssetStyleGenerationTests.asyncSetUp
    job = style_tests.AssetStyleGenerationTests.job

    async def test_mask_preview_and_asset_generation_queue_use_the_same_document(self):
        job = self.job(prompt_json=document_fixture())
        preview = await self.client.post("/api/asset/preview", json=job)
        self.assertEqual(preview.status, 200, await preview.text())
        expected = (await preview.json())["prompt_json"]
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [job]})
        self.assertEqual(response.status, 200, await response.text())
        queued = self.queue.await_args.args[0][0]
        self.assertEqual(queued["type"], "asset_generation")
        self.assertEqual(queued["params"]["body"]["prompt_json"], expected)
        self.assertEqual(json.loads(queued["params"]["body"]["positive_prompt"]), expected)

    async def test_one_invalid_region_rejects_entire_batch(self):
        invalid = document_fixture()
        invalid["fields"]["MULTI_CHAR"]["layout"]["regions"][1]["width"] = 2
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job(prompt_json=document_fixture()), self.job(prompt_json=invalid)]})
        self.assertEqual(response.status, 400)
        self.queue.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
