import importlib.util
import json
from pathlib import Path
import unittest

from illustration_json import build_illustration_json
from modes.illustration_prompt_builders import detail_moss_messages

NODE_PATH = (Path(__file__).resolve().parents[2] / "data" / "comfy" / "custom_nodes"
             / "comfyui-soya-custom-nodes" / "soya_moss_dialogue_repair.py")
spec = importlib.util.spec_from_file_location("soya_moss_dialogue_repair", NODE_PATH)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

SOURCE = """[C016]
룬이 조심스러운 발걸음으로 다가와 도윤의 손에 들린 첫 그림을 두 손으로 받아 들었다.

[C017]
"저, 저기... 신입 씨. 이 그림은 제가 마감해서 잘 말려둘게요. ...앞으로 잘 부탁드려요."

"아, 네. 저야말로 잘 부탁드립니다, 룬 씨."

[C018]
도윤이 마주 인사하며 살짝 눈웃음을 지으려 애쓰자, 룬의 녹색 눈동자가 동그랗게 커지더니 꼬리가 살랑 흔들렸다."""

BROKEN = [
    "저의 서리민렜 잨 부탁드립니다, rune 닉.",
    "아, 네! 기뻐 받심벤다.",
]
FIXED = [
    "저야말로 잘 부탁드립니다, 룬 씨.",
    "아, 네! 기쁘게 받겠습니다.",
]


def document(*, enabled=True):
    return {
        "moss": {
            "summary": {"caption": "page"},
            "items": [
                {"label": "object", "caption": "panel", "bbox": [0, 0, 10, 10]},
                {"label": "text", "caption": "balloon", "text": BROKEN[0],
                 "bbox": [1, 2, 3, 4]},
                {"label": "text", "caption": "balloon", "text": BROKEN[1],
                 "bbox": [5, 6, 7, 8]},
            ],
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        },
        "workflow": {
            "model": "moss.safetensors",
            "dialogue_repair": {
                "enabled": enabled,
                "anchor": "C017",
                "source": SOURCE,
            },
            "detailer": {
                "layout": json.dumps({"items": [
                    {"label": "text", "caption": "text", "text": BROKEN[0],
                     "bbox": [1, 2, 3, 4]},
                    {"label": "text", "caption": "text", "text": BROKEN[1],
                     "bbox": [5, 6, 7, 8]},
                ]}, ensure_ascii=False),
            },
        },
    }


class FakeClip:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def tokenize(self, text, **kwargs):
        return {"moss": [[(index, 1.0) for index in range(len(text) // 64 + 1)]]}

    def generate(self, tokens, **kwargs):
        self.calls += 1
        return [1, 2]

    def decode(self, ids):
        return self.reply


class SourceBlockTests(unittest.TestCase):
    def test_blocks_and_anchor_order(self):
        blocks = repair.source_blocks(SOURCE)
        self.assertEqual([block_id for block_id, _ in blocks], ["C016", "C017", "C018"])
        self.assertEqual(repair.order_blocks(blocks, "C017"), [1, 0, 2])

    def test_select_blocks_returns_source_order_and_includes_the_anchor(self):
        blocks = repair.source_blocks(SOURCE)
        items = repair.text_items(document()["moss"])
        picked = repair.select_blocks(blocks, "C017", items, 4096,
                                      count_tokens=lambda text: len(text) // 4)
        self.assertEqual([block_id for block_id, _ in picked], ["C016", "C017", "C018"])

    def test_select_blocks_rejects_a_budget_below_the_anchor(self):
        blocks = repair.source_blocks(SOURCE)
        items = repair.text_items(document()["moss"])
        self.assertEqual(repair.select_blocks(blocks, "C017", items, 1,
                                              count_tokens=lambda text: 10_000), [])


class ParseOutputTests(unittest.TestCase):
    def test_plain_lines_ignore_noise(self):
        raw = ("Sure, here are the lines:\n"
               "0. 저야말로 잘 부탁드립니다, 룬 씨.\n"
               "noise\n"
               "1. 아, 네! 기쁘게 받겠습니다.\n")
        self.assertEqual(repair.parse_output(raw, 2), {0: FIXED[0], 1: FIXED[1]})

    def test_newline_escape_round_trip(self):
        self.assertEqual(repair.parse_output("0. 하나\\n둘", 1), {0: "하나\n둘"})

    def test_valid_korean_rejects_latin_and_json(self):
        self.assertTrue(repair.is_valid_korean(FIXED[0], BROKEN[0]))
        self.assertFalse(repair.is_valid_korean("rune 닉", BROKEN[0]))
        self.assertFalse(repair.is_valid_korean('{"texts":["a"]}', BROKEN[0]))
        self.assertFalse(repair.is_valid_korean("", BROKEN[0]))


class RepairNodeTests(unittest.TestCase):
    def test_disabled_passes_through_without_calling_the_model(self):
        clip = FakeClip("0. x")
        raw = json.dumps(document(enabled=False), ensure_ascii=False)
        self.assertEqual(repair.SoyaMossDialogueRepairV2_mdsoya().repair(raw, clip), (raw,))
        self.assertEqual(clip.calls, 0)

    def test_missing_source_passes_through(self):
        payload = document()
        payload["workflow"]["dialogue_repair"]["source"] = ""
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(repair.SoyaMossDialogueRepairV2_mdsoya().repair(
            raw, FakeClip("0. x")), (raw,))

    def test_valid_output_updates_both_copies(self):
        clip = FakeClip("\n".join(f"{index}. {text}" for index, text in enumerate(FIXED)))
        raw = json.dumps(document(), ensure_ascii=False)
        result = json.loads(repair.SoyaMossDialogueRepairV2_mdsoya().repair(raw, clip)[0])
        texts = [item["text"] for item in result["moss"]["items"] if item["label"] == "text"]
        self.assertEqual(texts, FIXED)
        detailer = json.loads(result["workflow"]["detailer"]["layout"])
        self.assertEqual([item["text"] for item in detailer["items"]], FIXED)

    def test_invalid_output_keeps_the_original(self):
        raw = json.dumps(document(), ensure_ascii=False)
        result = json.loads(repair.SoyaMossDialogueRepairV2_mdsoya().repair(
            raw, FakeClip("0. rune 닉\n1. ???"))[0])
        texts = [item["text"] for item in result["moss"]["items"] if item["label"] == "text"]
        self.assertEqual(texts, BROKEN)

    def test_malformed_document_passes_through(self):
        raw = "{not json"
        self.assertEqual(repair.SoyaMossDialogueRepairV2_mdsoya().repair(
            raw, FakeClip("0. x")), (raw,))


class DetailRequestTests(unittest.TestCase):
    def test_source_context_never_reaches_the_detail_prompt(self):
        scene = {"anchor_segment": "C017", "slot": 1, "characters": [],
                 "source_context": "SHOULD_NOT_APPEAR_IN_THE_PROMPT"}
        messages = detail_moss_messages("system", [scene],
                                        toggles={"illustration_output_mode": "manga"})
        joined = "\n".join(message["content"] for message in messages)
        self.assertNotIn("SHOULD_NOT_APPEAR_IN_THE_PROMPT", joined)
        self.assertEqual(scene["source_context"], "SHOULD_NOT_APPEAR_IN_THE_PROMPT")


class DocumentControlBlockTests(unittest.TestCase):
    def test_dialogue_repair_block_is_embedded(self):
        layout = {
            "summary": {"caption": "page"},
            "items": [{"label": "text", "caption": "balloon", "text": BROKEN[0],
                       "bbox": [-100, -100, 100, 100]}],
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        defaults = {
            "workflow.model": "moss.safetensors",
            "workflow.clip": "clip.safetensors",
            "workflow.vae": "vae.safetensors",
            "workflow.sampling": {"steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                  "scheduler": "simple", "denoise": 1.0},
        }
        control = {"enabled": False, "anchor": "C017", "source": SOURCE}
        raw = build_illustration_json(
            "[ANIMA_CONTENT]\npage\n[IMG_W]\n1024\n[IMG_H]\n1024\n[MOSS_BUBBLE_ACTIVATE]\nfalse\n[END]",
            "", input_dir=".", workflow_defaults=defaults, moss_layout=layout,
            dialogue_repair=control)
        document = json.loads(raw)
        self.assertEqual(document["workflow"]["dialogue_repair"], control)

    def test_saved_document_uses_the_current_toggle(self):
        defaults = {
            "workflow.model": "moss.safetensors",
            "workflow.clip": "clip.safetensors",
            "workflow.vae": "vae.safetensors",
            "workflow.sampling": {"steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                  "scheduler": "simple", "denoise": 1.0},
        }
        saved = document(enabled=True)
        control = {"enabled": False, "anchor": "C017", "source": SOURCE}
        raw = build_illustration_json(
            json.dumps(saved, ensure_ascii=False), "", input_dir=".",
            workflow_defaults=defaults, dialogue_repair=control)
        self.assertFalse(json.loads(raw)["workflow"]["dialogue_repair"]["enabled"])


if __name__ == "__main__":
    unittest.main()
