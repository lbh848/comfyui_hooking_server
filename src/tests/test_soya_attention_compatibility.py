"""CPU smoke checks against installed SOYA hooks; no model loading or generation."""
import ast
import importlib.util
from pathlib import Path
from typing import Optional
import unittest

spec = importlib.util.spec_from_file_location("soya_attention_installer",
    Path(__file__).resolve().parents[1] / "comfy_installer/illustration_v2_nodes.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
patch_soya_attention_source = installer.patch_soya_attention_source

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

COMFY = Path(__file__).resolve().parents[2] / "data" / "comfy"
SOYA = COMFY / "custom_nodes" / "comfyui-soya-custom-nodes"


def definitions(path, names, namespace, *, patch_source=False):
    source = path.read_text(encoding="utf-8")
    if patch_source:
        source = patch_soya_attention_source(path.name, source)
        if patch_soya_attention_source(path.name, source) != source:
            raise AssertionError("Compatibility installation must be idempotent")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing attention definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@unittest.skipUnless(torch is not None and SOYA.is_dir(), "Requires the installed ComfyUI Python environment")
class SoyaAttentionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {"torch": torch, "F": F, "Optional": Optional}
        definitions(COMFY / "comfy/ldm/modules/attention.py", {"AttentionTensorContainer"}, namespace)
        definitions(SOYA / "anima_regional_conditioning.py", {"_masked_attn_op"}, namespace, patch_source=True)
        definitions(SOYA / "soya_snapref.py", {"_reference_attention", "_masked_reference_attention"}, namespace, patch_source=True)
        cls.hooks = namespace

    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(2, 3, 2, 4)
        self.k = torch.randn(2, 3, 2, 4)
        self.v = torch.randn(2, 3, 2, 4)
        self.preferred = object()
        self.options = {"test": True}
        self.calls = []

    def attention(self, q, k, v, *, transformer_options, preferred_attention):
        self.assertIs(preferred_attention, self.preferred)
        self.assertIs(transformer_options, self.options)
        self.calls.append((q, k, v))
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)

    def wrapped(self):
        return [self.hooks["AttentionTensorContainer"](value) for value in (self.q, self.k, self.v)]

    def test_reference_record_keeps_tensors_and_forwards_backend(self):
        record = [None]
        wrapped = self.wrapped()
        actual = self.hooks["_reference_attention"](
            self.attention, record, 0, *wrapped,
            transformer_options=self.options, preferred_attention=self.preferred,
        )
        self.assertIs(record[0][0], self.k)
        self.assertIs(record[0][1], self.v)
        self.assertTrue(all(value.tensor is None for value in wrapped))
        torch.testing.assert_close(actual, self.attention(self.q, self.k, self.v,
                                  transformer_options=self.options, preferred_attention=self.preferred))

    def test_regional_bias_stays_enforced_with_containers_and_backend_hint(self):
        bias = torch.full((1, 1, 3, 3), float("-inf"))
        bias[..., 0] = 0
        for inputs in ((self.q, self.k, self.v), self.wrapped()):
            actual = self.hooks["_masked_attn_op"](*inputs, transformer_options=self.options,
                                                  attn_bias=bias, preferred_attention=self.preferred)
            expected = self.v[:, :1].expand(-1, 3, -1, -1).flatten(2)
            torch.testing.assert_close(actual, expected)

    def test_snapref_changes_only_selected_positive_region(self):
        refs = {0: (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))}
        weights = torch.tensor([[1.0, 0.0, 0.0]])
        baseline = self.attention(self.q, self.k, self.v, transformer_options=self.options,
                                  preferred_attention=self.preferred)
        for strength in (0.0, 0.7):
            actual = self.hooks["_masked_reference_attention"](
                self.attention, refs, weights, [0, 1], strength, *self.wrapped(),
                transformer_options=self.options, preferred_attention=self.preferred,
            )
            torch.testing.assert_close(actual[1], baseline[1])
            torch.testing.assert_close(actual[0, 1:], baseline[0, 1:])
            ref_k, ref_v = refs[0]
            keys = torch.cat((ref_k, self.k[:1]), dim=1)
            values = torch.cat((ref_v, self.v[:1]), dim=1)
            bias = torch.zeros(1, 1, 1, keys.shape[1])
            bias[..., :2] = strength
            expected = F.scaled_dot_product_attention(self.q[:1, :1].transpose(1, 2),
                keys.transpose(1, 2), values.transpose(1, 2), attn_mask=bias).transpose(1, 2).flatten(2)
            torch.testing.assert_close(actual[:1, :1], expected)

    def test_snapref_without_references_preserves_attention(self):
        actual = self.hooks["_masked_reference_attention"](
            self.attention, {}, torch.zeros(0, 3), [0, 1], 0.0, *self.wrapped(),
            transformer_options=self.options, preferred_attention=self.preferred,
        )
        torch.testing.assert_close(actual, self.attention(self.q, self.k, self.v,
                                  transformer_options=self.options, preferred_attention=self.preferred))


if __name__ == "__main__":
    unittest.main()
