import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "comfy"
    / "custom_nodes"
    / "comfyui-soya-custom-nodes"
    / "soya_minimax_h3_ref2image.py"
)
MODULE_NAME = "soya_minimax_h3_ref2image_test"


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _common_upscale(samples, width, height, _method, _crop):
    return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")


def _conditioning_set_values(conditioning, values):
    output = []
    for tensor, metadata in conditioning:
        merged = metadata.copy()
        merged.update(values)
        output.append([tensor, merged])
    return output


def _load_module():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.model_management.intermediate_device = lambda: torch.device("cpu")
    comfy.nested_tensor = types.ModuleType("comfy.nested_tensor")
    comfy.nested_tensor.NestedTensor = FakeNestedTensor
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.common_upscale = _common_upscale

    node_helpers = types.ModuleType("node_helpers")
    node_helpers.conditioning_set_values = _conditioning_set_values
    nodes = types.ModuleType("nodes")
    nodes.MAX_RESOLUTION = 16384

    modules = {
        "comfy": comfy,
        "comfy.model_management": comfy.model_management,
        "comfy.nested_tensor": comfy.nested_tensor,
        "comfy.utils": comfy.utils,
        "node_helpers": node_helpers,
        "nodes": nodes,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


MODULE = _load_module()
Node = MODULE.SoyaMiniMaxH3ReferenceToImage_mdsoya


class FakeVAE:
    def __init__(self):
        self.encoded_shapes = []

    def encode(self, image):
        self.encoded_shapes.append(tuple(image.shape))
        return torch.zeros(
            (1, 24, 1, image.shape[1] // 16, image.shape[2] // 16)
        )


class FakeClip:
    def __init__(self):
        self.prompt = None
        self.ref_items = None

    def tokenize(self, prompt, minimax_ref_items):
        self.prompt = prompt
        self.ref_items = minimax_ref_items
        return "tokens"

    def encode_from_tokens_scheduled(self, tokens):
        if tokens != "tokens":
            raise AssertionError("unexpected token payload")
        return [[torch.zeros((1, 1, 1)), {"original": True}]]


class H3Ref2ImageTests(unittest.TestCase):
    def test_schema_exposes_one_required_and_eight_optional_references(self):
        input_types = Node.INPUT_TYPES()

        self.assertIn("ref_image_1", input_types["required"])
        self.assertEqual(
            [name for name in input_types["optional"] if name.startswith("ref_image_")],
            [f"ref_image_{index}" for index in range(2, 10)],
        )

    def test_prepare_builds_t1_zero_audio_latent_and_reference_blocks(self):
        node = Node()
        clip = FakeClip()
        vae = FakeVAE()
        first = torch.zeros((1, 512, 768, 3))
        second = torch.ones((1, 1024, 512, 3))

        conditioning, latent, diagnostics = node.prepare(
            clip=clip,
            vae=vae,
            prompt="Use <Picture 1> for the character and <Picture 2> for the outfit.",
            width=1344,
            height=768,
            ref_image_size="match",
            ref_image_1=first,
            ref_image_2=second,
        )

        video, audio = latent["samples"].unbind()
        self.assertEqual(tuple(video.shape), (1, 24, 1, 48, 84))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, 0))
        self.assertEqual(len(clip.ref_items), 2)
        self.assertEqual(len(conditioning[0][1]["minimax_refs"]), 2)
        self.assertEqual(conditioning[0][1]["minimax_refs"][0]["kind"], "image")
        self.assertIn("references: 2", diagnostics)
        self.assertIn("T=1", diagnostics)

    def test_match_mode_does_not_exceed_target_area_beyond_alignment(self):
        width, height = MODULE._aligned_reference_size(
            source_width=4000,
            source_height=3000,
            target_width=1344,
            target_height=768,
            mode="match",
        )

        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertLessEqual(width * height, 1344 * 768 + 32 * (width + height))

    def test_reference_slots_must_be_contiguous(self):
        with self.assertRaisesRegex(ValueError, "연속"):
            Node._reference_sequence(
                torch.zeros((1, 64, 64, 3)),
                {"ref_image_3": torch.zeros((1, 64, 64, 3))},
            )

    def test_invalid_target_alignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "32의 배수"):
            Node._validate_target(1000, 768)

    def test_non_h3_vae_latent_is_rejected_with_shape_context(self):
        with self.assertRaisesRegex(ValueError, "MiniMax H3"):
            Node._validate_reference_latent(torch.zeros((1, 4, 64, 64)), 1)

    def test_empty_prompt_is_logged_and_rejected(self):
        with self.assertRaisesRegex(ValueError, "비어"):
            Node().prepare(
                clip=FakeClip(),
                vae=FakeVAE(),
                prompt="  ",
                width=1344,
                height=768,
                ref_image_size="match",
                ref_image_1=torch.zeros((1, 64, 64, 3)),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
