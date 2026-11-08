"""Locale-independent installer logging, including redirected and embedded streams."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from console_encoding import configure_utf8_stdio, utf8_subprocess_env
from comfy_installer.node_compatibility import remove_instant_lora_python_compatibility
from comfy_installer.operations import isolated_subprocess_env, run_command


SOURCE = Path(__file__).resolve().parents[1]
MESSAGE = "한글 — 日本語 😀 café"


class FixedEncodingStream:
    """An embedding application can provide a stream without reconfigure()."""
    def __init__(self, encoding):
        self.encoding = encoding
        self.buffer = io.BytesIO()

    def write(self, text):
        self.buffer.write(text.encode(self.encoding))
        return len(text)

    def flush(self):
        pass


class ConsoleEncodingTests(unittest.TestCase):
    def test_redirected_streams_preserve_unicode_under_different_system_encodings(self):
        for encoding in ("cp949", "cp1252", "cp932", "ascii", "utf-8"):
            with self.subTest(encoding=encoding), patch.dict(os.environ):
                output, errors = io.BytesIO(), io.BytesIO()
                stdout = io.TextIOWrapper(output, encoding=encoding, errors="strict")
                stderr = io.TextIOWrapper(errors, encoding=encoding, errors="strict")
                with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                    configure_utf8_stdio()
                    print(MESSAGE)
                    print(MESSAGE, file=sys.stderr)
                    print("invalid surrogate: \ud800")
                    stdout.flush()
                    stderr.flush()
                self.assertIn(MESSAGE, output.getvalue().decode("utf-8"))
                self.assertIn(MESSAGE, errors.getvalue().decode("utf-8"))
                self.assertIn(r"\ud800", output.getvalue().decode("utf-8"))
                stdout.detach()
                stderr.detach()

    def test_fixed_encoding_capture_escapes_only_unsupported_characters(self):
        for encoding in ("cp949", "cp1252", "ascii"):
            with self.subTest(encoding=encoding), patch.dict(os.environ):
                stream = FixedEncodingStream(encoding)
                with patch.object(sys, "stdout", stream), patch.object(sys, "stderr", stream):
                    configure_utf8_stdio()
                    first = sys.stdout
                    configure_utf8_stdio()
                    self.assertIs(sys.stdout, first)
                    print(MESSAGE)
                self.assertEqual(stream.buffer.getvalue().decode(encoding).strip(), MESSAGE.encode(encoding, errors="backslashreplace").decode(encoding))

    def test_missing_standard_streams_are_supported(self):
        with patch.dict(os.environ), patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            configure_utf8_stdio()
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def test_reported_instant_lora_failure_is_reproduced_and_fixed_without_losing_ui_logs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            output = io.BytesIO()
            stream = io.TextIOWrapper(output, encoding="cp949", errors="strict")
            with self.assertRaises(UnicodeEncodeError):
                stream.write("패치 해제 생략 — missing")
            messages = []
            with patch.object(sys, "stdout", stream):
                configure_utf8_stdio()
                result = remove_instant_lora_python_compatibility(
                    comfy_root=root / "comfy", requirements_dir=root / "state",
                    allow_missing=True, log=messages.append,
                )
                stream.flush()
            self.assertEqual(result["status"], "missing")
            self.assertIn("패치 해제 생략 —", messages[0])
            self.assertIn("패치 해제 생략 —", output.getvalue().decode("utf-8"))
            stream.detach()

    def test_child_environment_overrides_conflicting_locale_settings(self):
        with patch.dict(os.environ, {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp949:strict", "PYTHONPATH": "unwanted"}):
            env = isolated_subprocess_env({"PYTHONIOENCODING": "ascii", "TEST_KEEP": "yes"})
            self.assertEqual(env["PYTHONUTF8"], "1")
            self.assertEqual(env["PYTHONIOENCODING"], "utf-8:backslashreplace")
            self.assertEqual(env["TEST_KEEP"], "yes")
            self.assertNotIn("PYTHONPATH", env)
        explicit = {"PYTHONIOENCODING": "cp932"}
        self.assertEqual(utf8_subprocess_env(explicit)["PYTHONIOENCODING"], "utf-8:backslashreplace")
        self.assertEqual(explicit, {"PYTHONIOENCODING": "cp932"})

    def test_real_subprocess_keeps_unicode_with_default_and_explicit_environments(self):
        code = f"import sys; print({MESSAGE!r}); print({MESSAGE!r}, file=sys.stderr)"
        with patch.dict(os.environ, {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp949:strict"}):
            for env in (None, dict(os.environ)):
                with self.subTest(explicit=env is not None):
                    lines = run_command([sys.executable, "-c", code], cwd=SOURCE, env=env)
                    self.assertEqual(lines, [MESSAGE, MESSAGE])

    def test_direct_installer_import_initializes_utf8_before_the_failed_node_branch(self):
        code = """
import sys, tempfile
from pathlib import Path
from comfy_installer.node_compatibility import remove_instant_lora_python_compatibility
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    result = remove_instant_lora_python_compatibility(comfy_root=root, requirements_dir=root, allow_missing=True)
    assert result['status'] == 'missing'
print(sys.stdout.encoding, sys.stdout.errors)
"""
        for encoding in ("cp949", "cp1252", "cp932", "ascii"):
            with self.subTest(encoding=encoding):
                env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": encoding + ":strict"}
                result = subprocess.run([sys.executable, "-c", code], cwd=SOURCE, env=env, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
                text = result.stdout.decode("utf-8")
                self.assertIn("패치 해제 생략 —", text)
                self.assertIn("utf-8 backslashreplace", text)

    def test_settings_and_app_launchers_initialize_utf8_without_batch_files(self):
        for module in ("settings_server", "launch"):
            with self.subTest(module=module):
                env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii:strict"}
                code = f"import {module}; print({MESSAGE!r})"
                result = subprocess.run([sys.executable, "-c", code], cwd=SOURCE, env=env, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
                self.assertIn(MESSAGE, result.stdout.decode("utf-8"))
