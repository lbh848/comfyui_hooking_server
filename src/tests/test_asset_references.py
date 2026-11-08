"""Asset reference modes use the same per-stage image contract as illustrations."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from PIL import Image, ImageDraw

from asset_json import prepare_asset_references, prepare_asset_prompt
from illustration_json import build_illustration_json
from queue_manager import QueueItem, QueueManager
from tests import test_asset_style_presets as style_tests

spec = importlib.util.spec_from_file_location("asset_reference_nodes", Path(__file__).resolve().parents[1] /
    "comfy_installer/resources/illustration_v2/soya_illustration_json_v2.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


def fixture(root, mode="none", preprocess="none"):
    images = []
    for name, color in (("primary", "red"), ("face", "blue"), ("second", "green")):
        path = root / f"{name}.png"
        image = Image.new("RGB", (32, 32), color)
        ImageDraw.Draw(image).line((8, 0, 8, 31), fill="black", width=2)
        image.save(path)
        images.append({"local_path": str(path)})
    document = {"schema": "soya.illustration.v2", "assets": {}, "fields": {"CHAR_LIST": ["asset_mode"], "SEED": 123,
        "SNAPREF": {"enabled": True, "strength_scale": "linear", "strength": .8, "fidelity": .4,
                    "mode": "cs", "megapixels": 1, "secondary_reference": mode, "preprocess": preprocess}}}
    job = {"snapref_enabled": True, "snapref_images": images[:1], "snapref_face_images": images[1:2],
           "snapref_secondary_images": images[2:]}
    return document, job


class AssetReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_all_delivery_modes_select_the_same_images_for_each_stage_as_illustrations(self):
        for mode in ("none", "face_only", "face", "representative", "stage"):
            with self.subTest(mode=mode):
                document, job = fixture(self.root, mode)
                prepare_asset_references(document, job)
                result = json.loads(prepare_asset_prompt(json.dumps(document), "", snapref_enabled=True))
                result = json.loads(build_illustration_json(json.dumps(result), "", input_dir=self.root, include_face_embeddings=False))
                refs = result["fields"]["SNAPREF"]["references"]
                primary, face = refs[0]["asset"], refs[0].get("face_asset")
                for stage in ("first", "hires", "hand", "face", "eye"):
                    expected = [face] if mode == "face_only" or mode == "stage" and stage in {"face", "eye"} else [primary]
                    if mode == "face": expected = [primary, face]
                    if mode == "representative": expected = [primary, refs[1]["asset"]]
                    self.assertEqual(nodes.stage_reference_assets(result, stage), {"asset_mode": list(dict.fromkeys(expected))})
                self.assertEqual(result["fields"]["SNAPREF"]["fidelity"], .4)

    def test_all_preprocessors_transform_embedded_pixels_once_and_keep_originals(self):
        for mode in ("none", "lineart", "lineart_color", "grayscale"):
            with self.subTest(mode=mode):
                document, job = fixture(self.root, "face", mode)
                prepare_asset_references(document, job)
                ref = document["fields"]["SNAPREF"]["references"][0]
                self.assertEqual("source_asset" in ref, mode != "none")
                self.assertEqual("face_source_asset" in ref, mode != "none")
                pixels = Image.open(io.BytesIO(base64.b64decode(document["assets"][ref["asset"]]["data"]))).convert("RGB")
                if mode == "grayscale":
                    red, green, blue = pixels.split()
                    self.assertEqual(red.tobytes(), green.tobytes())
                    self.assertEqual(red.tobytes(), blue.tobytes())
                if mode.startswith("lineart"):
                    self.assertEqual(pixels.getpixel((25, 25)), (255, 255, 255))
                delivered = json.loads(build_illustration_json(json.dumps(document), "", input_dir=self.root, include_face_embeddings=False))
                self.assertEqual(delivered["fields"]["SNAPREF"]["references"], document["fields"]["SNAPREF"]["references"])

    def test_optional_image_fallbacks_and_face_only_without_primary(self):
        for mode in ("face_only", "face", "representative", "stage"):
            document, job = fixture(self.root, mode)
            job.update(snapref_face_images=[], snapref_secondary_images=[])
            prepare_asset_references(document, job)
            primary = document["fields"]["SNAPREF"]["references"][0]["asset"]
            for stage in ("first", "face", "eye"):
                self.assertEqual(nodes.stage_reference_assets(document, stage), {"asset_mode": [primary]})
        document, job = fixture(self.root, "face_only")
        job["snapref_images"] = [{"local_path": "missing.png"}]
        prepare_asset_references(document, job)
        self.assertEqual(len(nodes.stage_reference_assets(document, "first")["asset_mode"]), 1)

    def test_disabled_and_zero_strength_do_not_read_missing_images(self):
        for enabled, strength in ((False, 1), (True, 0)):
            document, job = fixture(self.root)
            job.update(snapref_enabled=enabled, snapref_images=[{"local_path": "missing.png"}])
            document["fields"]["SNAPREF"]["strength"] = strength
            prepare_asset_references(document, job)
            delivered = json.loads(prepare_asset_prompt(json.dumps(document), "", snapref_enabled=enabled))
            self.assertEqual(nodes.stage_reference_assets(delivered, "first"), {})

    def test_invalid_settings_and_missing_required_images_fail_before_queueing(self):
        for key, value in (("mode", "bad"), ("preprocess", "bad"), ("secondary_reference", "bad"),
                           ("fidelity", 2), ("strength", -1), ("strength", True), ("megapixels", 0)):
            document, job = fixture(self.root)
            document["fields"]["SNAPREF"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prepare_asset_references(document, job)
        document, job = fixture(self.root)
        job["snapref_images"] = []
        with self.assertRaises(ValueError): prepare_asset_references(document, job)

    def test_regional_face_images_keep_character_ownership(self):
        document, job = fixture(self.root, "stage")
        document["fields"]["MULTI_CHAR"] = {"enable": True, "char_name_list": ["Alice", "Bob"]}
        job["regional_references"] = [
            {"CHAR": "Bob", "images": job["snapref_secondary_images"], "face_images": job["snapref_images"]},
            {"CHAR": "Alice", "images": job["snapref_images"], "face_images": job["snapref_face_images"]},
        ]
        prepare_asset_references(document, job)
        refs = document["fields"]["SNAPREF"]["references"]
        self.assertEqual(nodes.stage_reference_assets(document, "face"), {ref["CHAR"]: [ref["face_asset"]] for ref in refs})
        self.assertEqual(nodes.stage_reference_assets(document, "hand"), {ref["CHAR"]: [ref["asset"]] for ref in refs})


class AssetReferenceGatewayTests(style_tests.AssetStyleGenerationTests):
    async def test_preview_and_queue_preserve_images_after_source_deletion(self):
        document, references = fixture(self.config.parent, "face", "grayscale")
        job = self.job(prompt_json=document, **references)
        preview = await self.client.post("/api/asset/preview", json=job)
        self.assertEqual(preview.status, 200, await preview.text())
        self.assertTrue((await preview.json())["snapref_embedded"])
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [job]})
        self.assertEqual(response.status, 200, await response.text())
        body = self.queue.await_args.args[0][0]["params"]["body"]
        for path in self.config.parent.glob("*.png"): path.unlink()
        queue = QueueManager()
        queue.get_config = lambda: {}
        queue.asset_mode = SimpleNamespace(generate=AsyncMock(return_value={"success": True}))
        await queue._handle_asset_generation(QueueItem(id="ref", type="asset_generation", label="references", params={"body": body}))
        args = queue.asset_mode.generate.await_args.kwargs
        delivered = json.loads(prepare_asset_prompt(args["positive_prompt"], "", snapref_enabled=args["snapref_enabled"], snapref_dir=args["snapref_dir"]))
        self.assertEqual(len(nodes.stage_reference_assets(delivered, "first")["asset_mode"]), 2)

    async def test_invalid_reference_prevents_partial_batch_submission(self):
        document, references = fixture(self.config.parent)
        bad = copy.deepcopy(references)
        bad["snapref_images"] = [{"local_path": "missing.png"}]
        response = await self.client.post("/api/asset/enqueue", json={"jobs": [self.job(prompt_json=document, **references), self.job(prompt_json=document, **bad)]})
        self.assertEqual(response.status, 400)
        self.queue.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
