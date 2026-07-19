import io
import unittest
import zipfile
import copy
from unittest.mock import AsyncMock, patch

import aiohttp

from modes.chansub_prompt_builder import ChansubPromptBuilder, build_v1_prompt
from modes import chansub_service
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
        self.assertEqual(result["quality_tag_start"], 1)
        self.assertEqual(result["quality_tag_count"], 2)
        self.assertEqual(
            result["positive"],
            r"artist:sample, best quality, amazing quality, night, (dramatic lighting), "
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
            "artist:sdxl, sdxl quality, outdoors, 1girl",
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

    def test_build_v1_prompt_uses_anima_quality_and_negative_only(self):
        """V1 조립: ANIMA 품질/부정 프리셋만 사용, 아티스트 없음, [Positive]/[ILXL] 구조."""
        result = build_v1_prompt(
            "night, classroom",
            "1girl, hana, black hair",
            "sunset lighting",
            self.tags,
            self.settings,
        )
        # [Positive] = ANIMA 품질 프리셋 + setup + char + supplement, 아티스트 없음
        self.assertIn("[Positive]\nbest quality, amazing quality, night, classroom, "
                      "1girl, hana, black hair, sunset lighting", result["positive"])
        # [ILXL] = setup + char
        self.assertIn("[ILXL]\nnight, classroom, 1girl, hana, black hair", result["positive"])
        # [Negative] = ANIMA 부정 프리셋
        self.assertEqual(result["negative"], "bad anatomy, lowres")
        # 아티스트/LoRA 없음
        self.assertNotIn("artist:sample", result["positive"])
        self.assertNotIn("[LORA", result["positive"])

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

    def test_empty_image_response_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "비어 있습니다"):
            extract_image_from_response(b"", "image/png")

    def test_retryable_http_statuses(self):
        self.assertTrue(chansub_service._is_retryable_http_status(408))
        self.assertTrue(chansub_service._is_retryable_http_status(429))
        self.assertTrue(chansub_service._is_retryable_http_status(500))
        self.assertFalse(chansub_service._is_retryable_http_status(400))
        self.assertFalse(chansub_service._is_retryable_http_status(401))

    def test_retry_reorders_only_quality_area_of_positive_prompt(self):
        positive = r"artist:name, best quality, amazing quality, 1girl, (red dress, blue ribbon)"

        retry_positive, swapped = chansub_service.reorder_positive_prompt_for_retry(
            positive, 1, quality_tag_start=1, quality_tag_count=2
        )

        self.assertEqual(swapped, (1, 2))
        self.assertEqual(
            retry_positive,
            r"artist:name, amazing quality, best quality, 1girl, (red dress, blue ribbon)",
        )

    def test_first_ten_retries_use_distinct_quality_orders(self):
        positive = "artist:name, q1, q2, q3, q4, q5, 1girl"

        retries = [
            chansub_service.reorder_positive_prompt_for_retry(
                positive, retry_number, quality_tag_start=1, quality_tag_count=5
            )[0]
            for retry_number in range(1, 11)
        ]

        self.assertEqual(len(set(retries)), 10)
        self.assertTrue(all(value.startswith("artist:name, ") for value in retries))
        self.assertTrue(all(value.endswith(", 1girl") for value in retries))


class ChansubRetryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        chansub_service.update_api_key("test-key")

    def tearDown(self):
        chansub_service.update_api_key("")

    async def test_transient_failure_retries_until_success(self):
        request_mock = AsyncMock(
            side_effect=[
                aiohttp.ClientConnectionError("temporary-1"),
                aiohttp.ClientConnectionError("temporary-2"),
                b"image-data",
            ]
        )
        sleep_mock = AsyncMock()

        with patch.object(chansub_service, "_post_generate_request", request_mock), patch.object(
            chansub_service.asyncio, "sleep", sleep_mock
        ):
            image, result = await chansub_service.generate_image(
                "positive", "negative", 640, 960, max_retries=2, retry_delay_sec=3
            )

        self.assertEqual(image, b"image-data")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(request_mock.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)
        sleep_mock.assert_awaited_with(3.0)

    async def test_non_retryable_http_failure_stops_immediately(self):
        request_mock = AsyncMock(
            side_effect=chansub_service.ChansubRequestError(
                "챈섭 HTTP 401: unauthorized", retryable=False
            )
        )
        sleep_mock = AsyncMock()

        with patch.object(chansub_service, "_post_generate_request", request_mock), patch.object(
            chansub_service.asyncio, "sleep", sleep_mock
        ):
            image, error = await chansub_service.generate_image(
                "positive", "negative", 640, 960, max_retries=2, retry_delay_sec=3
            )

        self.assertIsNone(image)
        self.assertIn("HTTP 401", error)
        self.assertEqual(request_mock.await_count, 1)
        sleep_mock.assert_not_awaited()

    async def test_retryable_failure_stops_after_configured_retries(self):
        request_mock = AsyncMock(
            side_effect=chansub_service.ChansubRequestError(
                "챈섭 HTTP 503: unavailable", retryable=True
            )
        )
        sleep_mock = AsyncMock()

        with patch.object(chansub_service, "_post_generate_request", request_mock), patch.object(
            chansub_service.asyncio, "sleep", sleep_mock
        ):
            image, error = await chansub_service.generate_image(
                "positive", "negative", 640, 960, max_retries=2, retry_delay_sec=1.5
            )

        self.assertIsNone(image)
        self.assertIn("HTTP 503", error)
        self.assertEqual(request_mock.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)
        sleep_mock.assert_awaited_with(1.5)

    async def test_retry_updates_both_positive_prompt_fields_only_after_failure(self):
        captured_bodies = []

        async def capture_request(body, headers):
            captured_bodies.append(copy.deepcopy(body))
            if len(captured_bodies) == 1:
                raise chansub_service.ChansubRequestError(
                    "챈섭 HTTP 500: failed", retryable=True
                )
            return b"image-data"

        with patch.object(
            chansub_service, "_post_generate_request", side_effect=capture_request
        ), patch.object(chansub_service.asyncio, "sleep", new=AsyncMock()):
            image, result = await chansub_service.generate_image(
                "artist:name, best quality, amazing quality, 1girl, outdoors",
                "lowres, bad hands, blurry",
                640,
                960,
                max_retries=1,
                retry_delay_sec=0,
                quality_tag_start=1,
                quality_tag_count=2,
            )

        first_params = captured_bodies[0]["parameters"]
        retry_params = captured_bodies[1]["parameters"]
        self.assertEqual(
            captured_bodies[0]["input"],
            "artist:name, best quality, amazing quality, 1girl, outdoors",
        )
        self.assertEqual(
            captured_bodies[1]["input"],
            "artist:name, amazing quality, best quality, 1girl, outdoors",
        )
        self.assertEqual(
            retry_params["v4_prompt"]["caption"]["base_caption"],
            "artist:name, amazing quality, best quality, 1girl, outdoors",
        )
        self.assertEqual(first_params["negative_prompt"], "lowres, bad hands, blurry")
        self.assertEqual(retry_params["negative_prompt"], "lowres, bad hands, blurry")
        self.assertEqual(first_params["seed"], retry_params["seed"])
        self.assertEqual(image, b"image-data")
        self.assertEqual(result["attempts"], 2)


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
