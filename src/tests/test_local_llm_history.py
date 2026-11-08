"""Local Gemma history fitting preserves required input and reserves output."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from llm_sp_workflow import build_request, stream
from comfy_installer.resources.gemma4_vision.soya_gemma4_context import fit_history


spec = importlib.util.spec_from_file_location(
    "history_prompt_assembly", Path(__file__).resolve().parents[1] / "modes/illustration_prompt_assembly.py")
assembly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembly)


class LocalHistoryTests(unittest.TestCase):
    def counter(self, context, output=20):
        self.counted = []
        def count(text):
            self.counted.append(text)
            tokens = len(text) + 3  # A deterministic tokenizer with chat-template overhead.
            return {"input_tokens": tokens, "context_limit": context,
                    "max_tokens": output, "remaining": context - tokens - output}
        return count

    def test_only_marked_history_survives_flattening_with_unicode_and_images(self):
        history = "[USER]\n옛 대화😀\n\n[CHAR]\n최근 대화"
        content = "  # HISTORY\n" + assembly.variable(history, "과거 대화") + "\n# CURRENT\n현재 입력  "
        messages = [{"role": "system", "content": "Keep these instructions."},
                    {"role": "user", "content": [{"type": "text", "text": content},
                     {"type": "text", "text": "  Final instruction.  "},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}]}]
        original = copy.deepcopy(messages)
        request = build_request(copy.deepcopy(messages))
        self.assertEqual(len(request["history_ranges"]), 1)
        start, end = request["history_ranges"][0]
        self.assertEqual(request["prompt"][start:end], history)
        self.assertTrue(request["prompt"].startswith("SYSTEM:\nKeep these instructions.\n\nUSER:\n# HISTORY\n"))
        self.assertTrue(request["prompt"].endswith("Final instruction."))
        self.assertEqual(request["image"]["data"], "YQ==")
        self.assertEqual(messages, original)

    def test_exact_fit_keeps_all_history_and_counts_only_once(self):
        history = "[USER]\nOld\n\n[CHAR]\nRecent"
        prompt = "Rules\n" + history + "\nCurrent"
        count = self.counter(len(prompt) + 3 + 20)
        adjusted, metrics = fit_history(prompt, [[6, 6 + len(history)]], count)
        self.assertEqual(adjusted, prompt)
        self.assertEqual(metrics["history_turns_removed"], 0)
        self.assertEqual(len(self.counted), 1)

    def test_oldest_turns_removed_with_current_and_response_preserved(self):
        prefix, suffix = "SYSTEM instructions\nHISTORY\n", "\nCURRENT scene and profile\nOUTPUT schema"
        oldest = "[USER]\nOld request\n\n[CHAR]\nOld response\n\n"
        recent = "[USER]\nRecent request\n\n[CHAR]\nRecent response"
        prompt = prefix + oldest + recent + suffix
        ranges = [[len(prefix), len(prefix) + len(oldest + recent)]]
        original = copy.deepcopy(ranges)
        expected = prefix + recent + suffix
        count = self.counter(len(expected) + 3 + 20)
        adjusted, metrics = fit_history(prompt, ranges, count)
        self.assertEqual(adjusted, expected)
        self.assertEqual(metrics["history_turns_removed"], 2)
        self.assertEqual(metrics["max_tokens"], 20)
        self.assertEqual(metrics["input_tokens"] + 20, metrics["context_limit"])
        self.assertEqual(ranges, original)

    def test_combined_scene_and_keyvisual_history_use_same_recent_suffix(self):
        older, recent = "[CHAR]\nOld facts\n\n", "[USER]\nRecent facts"
        history = older + recent
        prefix, middle, suffix = "Scene\n", "\nKey Visual\n", "\nCurrent narrative"
        prompt = prefix + history + middle + history + suffix
        second = len(prefix + history + middle)
        expected = prefix + recent + middle + recent + suffix
        adjusted, metrics = fit_history(prompt, [[len(prefix), len(prefix + history)],
                                                [second, second + len(history)]],
                                        self.counter(len(expected) + 23))
        self.assertEqual(adjusted, expected)
        self.assertEqual(metrics["history_chars"], len(recent) * 2)

    def test_history_can_be_empty_without_truncating_current_input(self):
        history = "[CHAR]\n" + "Past conversation. " * 10
        prefix, suffix = "Rules\n", "\nCurrent input"
        expected = prefix + "(none)" + suffix
        adjusted, metrics = fit_history(prefix + history + suffix,
                                        [[len(prefix), len(prefix + history)]],
                                        self.counter(len(expected) + 23))
        self.assertEqual(adjusted, expected)
        self.assertEqual(metrics["history_chars"], 0)

    def test_repeated_individual_messages_still_keep_the_latest_occurrence(self):
        history = "[USER]\nA repeated request"
        prompt = "Rules\n" + history + "\nSeparator\n" + history + "\nCurrent"
        second = len("Rules\n" + history + "\nSeparator\n")
        expected = "Rules\n(none)\nSeparator\n" + history + "\nCurrent"
        adjusted, _ = fit_history(prompt, [[6, 6 + len(history)], [second, second + len(history)]],
                                  self.counter(len(expected) + 23))
        self.assertEqual(adjusted, expected)

    def test_required_input_overflow_is_actionable_and_never_truncated(self):
        history = "[CHAR]\nPast"
        prompt = "Rules\n" + history + "\nCurrent " + "x" * 100
        with self.assertRaisesRegex(RuntimeError, "히스토리를 제외한 필수 입력.*응답 한도.*컨텍스트"):
            fit_history(prompt, [[6, 6 + len(history)]], self.counter(60))
        self.assertTrue(all(text.endswith("Current " + "x" * 100) for text in self.counted))

    def test_unmarked_history_like_text_is_not_trimmed(self):
        request = build_request([{"role": "user", "content": "# PAST HISTORY\n[CHAR]\n" + "x" * 100}])
        self.assertNotIn("history_ranges", request)
        with self.assertRaisesRegex(RuntimeError, "필수 입력"):
            fit_history(request["prompt"], [], self.counter(60))
        self.assertEqual(self.counted, [request["prompt"]])


class LocalStreamErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_error_does_not_duplicate_llm_failure_prefix(self):
        with patch("llm_sp_workflow.analyze", side_effect=RuntimeError("Gemma 컨텍스트 한도 초과")):
            events = [event async for event in stream([{"role": "user", "content": "Hello"}])]
        self.assertEqual(events[-1], {"type": "error", "error": "Gemma 컨텍스트 한도 초과"})


if __name__ == "__main__":
    unittest.main()
