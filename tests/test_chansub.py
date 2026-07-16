import io
import unittest
import zipfile

from modes.chansub_prompt_builder import ChansubPromptBuilder
from modes.chansub_service import build_request_body, extract_image_from_response
from modes import llm_prompt_edit


class ChansubPromptBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tags = {
            "artist_presets": {
                "artist-a": ["artist:sample"],
                "artist-sdxl": ["artist:sdxl"],
            },
            "quality_presets": {
                "quality-a": ["best quality", "amazing quality"],
                "quality-sdxl": ["sdxl quality"],
            },
            "negative_presets": {
                "negative-a": ["bad anatomy", "lowres"],
                "negative-sdxl": ["sdxl negative"],
            },
            "anima_quality": ["default quality"],
            "anima_negative": ["default negative"],
            "quality": ["default sdxl quality"],
            "negative": ["default sdxl negative"],
        }
        self.settings = {
            "anima_artist_preset": "artist-a",
            "anima_quality_preset": "quality-a",
            "anima_negative_preset": "negative-a",
            "sdxl_artist_preset": "artist-sdxl",
            "sdxl_quality_preset": "quality-sdxl",
            "sdxl_negative_preset": "negative-sdxl",
            "img_w": 832,
            "img_h": 1216,
        }

    def test_build_uses_anima_presets_scene_and_size_only(self):
        result = ChansubPromptBuilder().build(
            "night, (dramatic lighting)",
            r"shifty \(nikke\), standing",
            "rain",
            self.tags,
            self.settings,
        )
        self.assertEqual(result["width"], 832)
        self.assertEqual(result["height"], 1216)
        self.assertEqual(
            result["positive"],
            r"best quality, amazing quality, artist:sample, night, (dramatic lighting), "
            r"shifty \(nikke\), standing, rain",
        )
        self.assertEqual(result["negative"], "bad anatomy, lowres")
        self.assertNotIn("[LORA", result["positive"])
        self.assertNotIn("[ANIMA", result["positive"])

    def test_build_preserves_comfy_weight_and_character_separator_syntax(self):
        result = ChansubPromptBuilder().build(
            "(dramatic lighting:1.2)",
            r"alice \(series\) | bob \(series\)",
            "",
            self.tags,
            self.settings,
        )
        self.assertIn("(dramatic lighting:1.2)", result["positive"])
        self.assertIn(r"alice \(series\) | bob \(series\)", result["positive"])

    def test_build_uses_selected_sdxl_presets(self):
        self.settings["chansub_workflow_type"] = "sdxl"

        result = ChansubPromptBuilder().build(
            "outdoors", "1girl", "sunlight", self.tags, self.settings
        )

        self.assertEqual(
            result["positive"],
            "sdxl quality, artist:sdxl, outdoors, 1girl",
        )
        self.assertEqual(result["negative"], "sdxl negative")
        self.assertNotIn("artist:sample", result["positive"])
        self.assertNotIn("sunlight", result["positive"])

    def test_sdxl_without_presets_uses_existing_default_tags(self):
        self.settings.update(
            {
                "chansub_workflow_type": "sdxl",
                "sdxl_quality_preset": "",
                "sdxl_negative_preset": "",
            }
        )

        result = ChansubPromptBuilder().build(
            "outdoors", "1girl", "", self.tags, self.settings
        )

        self.assertIn("default sdxl quality", result["positive"])
        self.assertEqual(result["negative"], "default sdxl negative")

    def test_invalid_workflow_type_falls_back_to_anima(self):
        self.settings["chansub_workflow_type"] = "unknown"

        result = ChansubPromptBuilder().build(
            "outdoors", "1girl", "", self.tags, self.settings
        )

        self.assertIn("best quality, amazing quality", result["positive"])
        self.assertEqual(result["negative"], "bad anatomy, lowres")


class ChansubServiceTest(unittest.TestCase):
    def test_request_body_matches_nai_shape(self):
        body = build_request_body("positive", "negative", 640, 960)
        self.assertEqual(body["action"], "generate")
        self.assertEqual(body["input"], "positive")
        self.assertEqual(body["parameters"]["negative_prompt"], "negative")
        self.assertEqual(body["parameters"]["width"], 640)
        self.assertEqual(body["parameters"]["height"], 960)
        self.assertEqual(
            body["parameters"]["v4_prompt"]["caption"]["base_caption"], "positive"
        )

    def test_extracts_first_image_from_zip(self):
        png = b"\x89PNG\r\n\x1a\n" + b"image-data"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("image_0.png", png)
        self.assertEqual(
            extract_image_from_response(buffer.getvalue(), "application/zip"), png
        )


class ChansubLlmEditTest(unittest.TestCase):
    def test_provider_metadata_detects_flat_chansub_prompt(self):
        self.assertEqual(
            llm_prompt_edit.detect_format("plain, nai, tags", provider="chansub"),
            "chansub",
        )

    def test_chansub_messages_and_reassembly(self):
        messages = llm_prompt_edit.build_chansub_llm_messages(
            "비를 추가", "1girl, outdoors", "lowres"
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("1girl, outdoors", messages[1]["content"])
        self.assertIn("lowres", messages[1]["content"])

        positive, negative, meta = llm_prompt_edit.reassemble_chansub(
            "old positive",
            "old negative",
            {"plan": "비 추가", "positive": "new positive", "negative": "new negative"},
        )
        self.assertEqual(positive, "new positive")
        self.assertEqual(negative, "new negative")
        self.assertEqual(meta["plan"], "비 추가")


if __name__ == "__main__":
    unittest.main()
