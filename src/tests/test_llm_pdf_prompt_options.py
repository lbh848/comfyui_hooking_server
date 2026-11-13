"""Image PDF transport options and preserved system-prompt coverage."""
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

from app_paths import SOURCE_DIR
from llm_settings import SLOT_DEFAULTS
from modes.llm_pdf_prompt import (
    PDF_IMAGE_DEFAULT_OPTIONS,
    build_text_pdf,
    normalize_image_options,
    prepare_pdf_prompt_messages,
)
from settings_store import LlmSlot


def load_engine():
    """Load llm_service without constructing the unrelated mode singletons."""
    package = types.ModuleType("modes")
    package.__path__ = [str(SOURCE_DIR / "modes")]
    spec = importlib.util.spec_from_file_location(
        "llm_pdf_options_test_engine", SOURCE_DIR / "modes/llm_service.py"
    )
    engine = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modes": package, spec.name: engine}):
        spec.loader.exec_module(engine)
    return engine


class ImagePdfOptionsTest(unittest.TestCase):
    def test_defaults_match_image_panel(self):
        options = normalize_image_options()
        self.assertEqual(options["media_resolution"], "medium")
        self.assertEqual(options["page_width"], 1191)
        self.assertEqual(options["page_height"], 1684)
        self.assertEqual(options["margin"], 12)
        self.assertEqual(options["font_size"], 16)
        self.assertEqual(options["line_spacing"], 1.2)
        self.assertEqual(options["line_break"], "character")
        self.assertEqual(options["color_mode"], "grayscale")
        self.assertTrue(options["preserve_system_prompt"])
        self.assertIn("Malgun Gothic", options["font_family"])

    def test_settings_schema_and_runtime_defaults_match(self):
        slot = LlmSlot(id="llm7", providerId="vertex")
        self.assertEqual(slot.pdf_image_page_width, PDF_IMAGE_DEFAULT_OPTIONS["page_width"])
        self.assertEqual(slot.pdf_image_font_size, PDF_IMAGE_DEFAULT_OPTIONS["font_size"])
        self.assertTrue(slot.pdf_image_preserve_system_prompt)
        for key, value in PDF_IMAGE_DEFAULT_OPTIONS.items():
            self.assertEqual(SLOT_DEFAULTS[f"pdf_image_{key}"], value)

    def test_image_pdf_uses_configured_page_size(self):
        pdf, page_count = build_text_pdf(
            "hello world",
            prompt_format="image",
            image_options={
                "page_width": 800,
                "page_height": 1000,
                "margin": 20,
                "font_size": 20,
                "line_spacing": 1.5,
            },
        )
        self.assertGreaterEqual(page_count, 1)
        self.assertIn(b"/MediaBox [0 0 400 500]", pdf)

    def test_image_pdf_wraps_by_word_and_character(self):
        long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        for line_break in ("character", "word"):
            pdf, page_count = build_text_pdf(
                long_text,
                prompt_format="image",
                image_options={
                    "page_width": 256,
                    "page_height": 256,
                    "margin": 12,
                    "font_size": 16,
                    "line_spacing": 1.2,
                    "line_break": line_break,
                },
            )
            self.assertGreaterEqual(page_count, 1)
            self.assertTrue(pdf.startswith(b"%PDF-1.7"))

    def test_system_prompt_is_preserved_outside_pdf(self):
        messages = [
            {"role": "system", "content": "SYSTEM RULES"},
            {"role": "user", "content": "hello"},
        ]
        transformed, metadata = prepare_pdf_prompt_messages(
            messages,
            encode_transcript=False,
            prompt_format="image",
            image_options={"preserve_system_prompt": True},
        )
        self.assertEqual(metadata["preserved_system_chars"], len("SYSTEM RULES"))
        preserved = [message for message in transformed
                     if "PRESERVED ORIGINAL SYSTEM PROMPT" in str(message.get("content"))]
        self.assertEqual(len(preserved), 1)
        self.assertIn("SYSTEM RULES", preserved[0]["content"])

        _, without = prepare_pdf_prompt_messages(
            messages,
            encode_transcript=False,
            prompt_format="image",
            image_options={"preserve_system_prompt": False},
        )
        self.assertEqual(without["preserved_system_chars"], 0)

    def test_preserved_system_stays_plain_when_base64_is_enabled(self):
        messages = [
            {"role": "system", "content": "SYSTEM RULES"},
            {"role": "user", "content": "hello"},
        ]
        transformed, _ = prepare_pdf_prompt_messages(
            messages,
            encode_transcript=True,
            prompt_format="image",
            image_options={"preserve_system_prompt": True},
        )
        preserved = [message for message in transformed if message.get("_base64_exempt")]
        self.assertEqual(len(preserved), 1)
        engine = load_engine()
        encoded = engine._encode_gemini_base64_messages(transformed)
        plain = [message for message in encoded
                 if "PRESERVED ORIGINAL SYSTEM PROMPT" in str(message.get("content"))]
        self.assertEqual(len(plain), 1)
        self.assertIn("SYSTEM RULES", plain[0]["content"])
        self.assertNotIn("_base64_exempt", plain[0])

    def test_image_mode_uses_configured_media_resolution(self):
        engine = load_engine()
        messages = [{
            "role": "user",
            "content": [{
                "type": "file",
                "file": {
                    "filename": "llm-conversation.pdf",
                    "file_data": "data:application/pdf;base64,AA==",
                },
            }],
        }]
        token = engine._request_config_override_ctx.set({
            "llm_pdf_prompt_format": "image",
            "llm_pdf_image_media_resolution": "medium",
        })
        try:
            body = engine._build_gemini_request_body(
                messages, "gemini-4-argon", service="vertex"
            )
        finally:
            engine._request_config_override_ctx.reset(token)
        self.assertEqual(
            body["generationConfig"]["mediaResolution"],
            "MEDIA_RESOLUTION_MEDIUM",
        )
        token = engine._request_config_override_ctx.set({
            "llm_pdf_prompt_format": "pdf",
        })
        try:
            body = engine._build_gemini_request_body(
                messages, "gemini-4-argon", service="vertex"
            )
        finally:
            engine._request_config_override_ctx.reset(token)
        self.assertEqual(
            body["generationConfig"]["mediaResolution"],
            "MEDIA_RESOLUTION_LOW",
        )


if __name__ == "__main__":
    unittest.main()
