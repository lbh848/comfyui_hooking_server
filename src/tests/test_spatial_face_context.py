"""Spatial face ownership and FD/ED graph integration; no model downloads."""
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from illustration_json import build_illustration_json

RESOURCES = Path(__file__).resolve().parents[2] / "data/comfy/custom_nodes/comfyui-soya-custom-nodes"
package = ModuleType("spatial_test_nodes")
package.__path__ = [str(RESOURCES)]
sys.modules[package.__name__] = package
spatial = importlib.import_module("spatial_test_nodes.soya_spatial_face_context")


def detector_for(boxes, confidences=None):
    confidences = confidences or [0.95] * len(boxes)
    detections = [SimpleNamespace(conf=torch.tensor([confidence]), xyxy=torch.tensor([box], dtype=torch.float32))
                  for box, confidence in zip(boxes, confidences)]
    return SimpleNamespace(bbox_model=Mock(return_value=[SimpleNamespace(boxes=detections)]))


def payload_for(**fields):
    return json.dumps({"schema": "soya.illustration.v2", "fields": fields, "assets": {}})


class SpatialOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.left, self.right = [0.1, 0.1, 0.3, 0.3], [0.7, 0.1, 0.9, 0.3]
        self.mask = np.zeros((100, 100, 3), dtype=np.float32)
        self.mask[:, :50, 0], self.mask[:, 50:, 1] = 1, 1

    def test_mask_channel_order_is_independent_of_character_and_detection_order(self):
        assigned = spatial.assign_faces([self.right, self.left], ["Bob", "Alice"],
            mask=self.mask, mask_names=["Alice", "Bob"])
        self.assertEqual([(a[0], a[1]) for a in assigned], [(0, "Bob"), (1, "Alice")])

    def test_overlapping_fallback_masks_keep_global_one_to_one_assignment(self):
        boxes = [[.4, .1, .6, .3], self.left]
        assigned = spatial.assign_faces(boxes, ["Alice", "Bob"], mask=self.mask, mask_names=["Alice", "Bob"])
        self.assertEqual([(a[0], a[1]) for a in assigned], [(0, "Bob"), (1, "Alice")])
        self.assertEqual(spatial.assign_faces(boxes, ["Alice", "Bob"]), [])

    def test_solo_without_mask_selects_largest_face_once(self):
        assigned = spatial.assign_faces([[.1, .1, .15, .15], self.right], ["Alice"])
        self.assertEqual([(a[0], a[1]) for a in assigned], [(1, "Alice")])

    def test_manga_can_assign_same_character_in_two_panels(self):
        panels = [{"x": 0, "y": 0, "width": .5, "height": 1}, {"x": .5, "y": 0, "width": .5, "height": 1}]
        assigned = spatial.assign_faces([self.left, self.right], ["Alice"], panels=panels)
        self.assertEqual([a[1] for a in assigned], ["Alice", "Alice"])

    def test_context_uses_current_resolution_and_clamps_crop(self):
        assigned = [(0, "Alice", 1., "tracked")]
        context = spatial.make_context([[200, 100, 600, 300]], assigned, 2000, 1000, 3, 3)
        self.assertEqual(context["matches"][0]["bbox"], [200, 100, 600, 300])
        self.assertEqual(context["matches"][0]["crop"], [0, 0, 1000, 500])
        self.assertEqual((context["img_W"], context["img_H"], context["ipa_embeds"]), (2000, 1000, {}))

    def test_invalid_tracking_coordinates_are_rejected(self):
        for box in ([0, 0, float("nan"), 1], [0, 0, 2, 1], [0, 1, 1, 0], [1, 2]):
            with self.assertRaises(ValueError):
                spatial.normalized_box(box)

    def test_tracking_only_supplies_names_and_detector_boxes_supply_expanded_crops(self):
        image = torch.ones(1, 200, 400, 3)
        detector = detector_for([[285, 25, 355, 65], [45, 25, 115, 65]])
        payload = payload_for(CHAR_LIST=["Alice", "Bob"], FACE_CROP_TOP=4, FACE_CROP_BOTTOM=2)
        tracks = json.dumps([{"name": "Alice", "bbox": self.left}, {"name": "Bob", "bbox": self.right}])
        node = spatial.SoyaSpatialFaceContextV2()
        with patch.object(torch, "load", side_effect=AssertionError("cache must not be loaded")):
            output = node.build(image, payload, tracks, bbox_detector=detector, debug=True)
        self.assertEqual(output[2], ["Bob", "Alice"])
        self.assertEqual(output[4]["matches"][0]["bbox"], [285, 25, 355, 65])
        self.assertEqual(output[4]["matches"][0]["crop"], [215, 0, 400, 85])
        self.assertEqual(output[4]["matches"][1]["crop"], [0, 0, 185, 85])
        self.assertEqual(output[4]["matches"][0]["method"], "sampler")
        self.assertEqual(output[1][0].shape, (1, 85, 185, 3))
        self.assertTrue(torch.all(output[1][0] == 1), "hair/surrounding pixels must not be cut out by tracking masks")
        detector.bbox_model.assert_called_once()
        self.assertEqual(detector.bbox_model.call_args.kwargs, {"verbose": False})

    def test_close_tracked_faces_are_kept_with_their_original_names(self):
        image = torch.zeros(1, 100, 100, 3)
        payload = json.dumps({"schema": "soya.illustration.v2", "fields": {
            "CHAR_LIST": ["Alice", "Bob"], "MULTI_CHAR": {"enable": False}}, "assets": {}})
        tracks = json.dumps([{"name": "Alice", "bbox": [.3, .2, .55, .5]},
                             {"name": "Bob", "bbox": [.45, .2, .7, .5]}])
        detector = detector_for([[30, 20, 55, 50], [45, 20, 70, 50]])
        output = spatial.SoyaSpatialFaceContextV2().build(image, payload, tracks, detector)
        self.assertEqual([m["name"] for m in output[4]["matches"]], ["Alice", "Bob"])
        self.assertEqual(len(output[4]["matches"]), 2)

    def test_missing_tracking_uses_mask_fallback_without_reassigning_tracked_face(self):
        stream = io.BytesIO()
        Image.fromarray((self.mask * 255).astype(np.uint8)).save(stream, format="PNG")
        data = stream.getvalue()
        key = hashlib.sha256(data).hexdigest()
        payload = json.dumps({"schema": "soya.illustration.v2", "fields": {
            "CHAR_LIST": ["Alice", "Bob"], "MULTI_CHAR": {"enable": True, "char_name_list": ["Alice", "Bob"]},
            "REGION_MASK": key,
        }, "assets": {key: {"encoding": "base64", "sha256": key, "data": base64.b64encode(data).decode()}}})
        image = torch.zeros(1, 100, 100, 3)
        tracks = json.dumps([{"name": "Alice", "bbox": self.left}])
        detector = detector_for([[70, 10, 90, 30], [10, 10, 30, 30]])
        node = spatial.SoyaSpatialFaceContextV2()
        output = node.build(image, payload, tracks, detector)
        self.assertEqual([(m["name"], m["method"]) for m in output[4]["matches"]], [("Bob", "mask"), ("Alice", "sampler")])

    def test_detector_threshold_limit_and_no_detection_keep_legacy_behavior(self):
        payload = payload_for(CHAR_LIST=["Alice"])
        image = torch.zeros(1, 100, 100, 3)
        detector = detector_for([[10, 10, 50, 50], [30, 30, 90, 90], [70, 10, 90, 30]], [.95, .4, .8])
        node = spatial.SoyaSpatialFaceContextV2()
        context = node.build(image, payload, "[]", detector, max_face_count=1, yolo_confidence=.7)[4]
        self.assertEqual(context["matches"][0]["bbox"], [10, 10, 50, 50])
        empty = node.build(image, payload, json.dumps([{"name": "Alice", "bbox": self.left}]), detector_for([]))[4]
        self.assertEqual(empty["matches"], [], "tracking must not invent a detailer crop without a detected face")

    def test_fractional_box_asymmetric_crop_preserves_legacy_integer_rounding(self):
        box = np.array([80.3, 40.1, 120.7, 70.9], dtype=np.float32)
        context = spatial.make_context([box], [(0, "Alice", 1, "sampler")], 200, 100, 4, 2)
        self.assertEqual(context["matches"][0]["crop"], [39, 0, 161, 86])


class SpatialGraphTests(unittest.TestCase):
    def test_spatial_transport_never_opens_missing_character_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = '[CHAR_LIST]\nNew NPC\n[CACHE_PATH]\n{"list":[{"CHAR":"New NPC","emb_path":"missing/cache.pt"}]}\n[END]'
            payload = json.loads(build_illustration_json(prompt, "", input_dir=directory, include_face_embeddings=False))
            self.assertNotIn("FACE_EMBEDDINGS", payload["fields"])
            self.assertEqual(payload["assets"], {})
            with self.assertRaises(FileNotFoundError):
                build_illustration_json(prompt, "", input_dir=directory)

    def test_saved_embedding_assets_are_removed_but_image_assets_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"schema": "soya.illustration.v2", "fields": {
                "FACE_EMBEDDINGS": [{"CHAR": "Alice", "asset": "cache"}],
                "SNAPREF": {"enabled": True, "references": [{"CHAR": "Alice", "asset": "image"}]},
                "REGION_MASK": "mask",
            }, "assets": {key: {"fixture": key} for key in ("cache", "image", "mask")}}
            result = json.loads(build_illustration_json(json.dumps(payload), "", input_dir=directory, include_face_embeddings=False))
            self.assertNotIn("FACE_EMBEDDINGS", result["fields"])
            self.assertEqual(set(result["assets"]), {"image", "mask"})

    def test_new_character_uses_representative_when_secondary_face_is_not_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "soya_bot/Story/Alice").mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(root / "soya_bot/Story/Alice/representation.png")
            payload = {"schema": "soya.illustration.v2", "fields": {
                "CHAR_LIST": ["Alice"],
                "FACE_ID_DIR": {"list": [{"CHAR": "Alice", "ipa_path": "soya_bot/Story/Alice/cache.ipadpt"}]},
                "SNAPREF": {"enabled": True, "secondary_reference": "face"},
            }, "assets": {}}
            result = json.loads(build_illustration_json(json.dumps(payload), "", input_dir=root, include_face_embeddings=False))
            self.assertEqual(len(result["fields"]["SNAPREF"]["references"]), 1)
            self.assertEqual(len(result["assets"]), 1)
            legacy = json.loads(build_illustration_json(json.dumps(payload), "", input_dir=root))
            self.assertEqual(len(legacy["fields"]["SNAPREF"]["references"]), 1)


if __name__ == "__main__":
    unittest.main()
