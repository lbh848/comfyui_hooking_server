"""face_detector v9c 주/v8m 폴백 + postprocess face_fallback 설정 키 단위테스트.

실제 ONNX 세션 없이, _resolve_primary_model / detect_faces 폴백 분기만 검증한다.
(추론 자체는 face_yolo_benchmark 의 FakeSession 테스트가 이미 커버)
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modes import face_detector as fd  # noqa: E402
from modes import postprocess as pp  # noqa: E402


def _img():
    return Image.new("RGB", (64, 64), (120, 120, 120))


class TestResolvePrimaryModel(unittest.TestCase):
    def test_prefers_v9c_when_available(self):
        with patch.object(fd, "_ensure_model", side_effect=lambda k: k == "v9c"):
            self.assertEqual(fd._resolve_primary_model(), "v9c")

    def test_falls_back_to_v8m_when_v9c_missing(self):
        with patch.object(fd, "_ensure_model", side_effect=lambda k: k == "v8m"):
            self.assertEqual(fd._resolve_primary_model(), "v8m")

    def test_none_when_no_model_available(self):
        with patch.object(fd, "_ensure_model", return_value=False):
            self.assertIsNone(fd._resolve_primary_model())


class TestDetectFacesFallback(unittest.TestCase):
    """detect_faces 의 v9c-주/v8m-폴백 분기를 _run_with_cpu_fallback monkeypatch 로 검증."""

    def test_no_fallback_when_disabled(self):
        with patch.object(fd, "_resolve_primary_model", return_value="v9c"), \
                patch.object(fd, "_run_with_cpu_fallback", return_value=([], [])) as run:
            out = fd.detect_faces(_img(), conf_thres=0.3, device="cpu",
                                  face_fallback=False)
            self.assertEqual(out, [])
            self.assertEqual(run.call_count, 1)  # v9c 1회만

    def test_fallback_runs_v8m_on_miss(self):
        def fake_run(model_key, device_key, cpu_threads, run_fn):
            # v9c → 0건, v8m → 1건 회복
            if model_key == "v9c":
                return ([], [])
            return ([(10, 10, 30, 30)], [0.8])

        with patch.object(fd, "_resolve_primary_model", return_value="v9c"), \
                patch.object(fd, "_ensure_model", return_value=True), \
                patch.object(fd, "_run_with_cpu_fallback", side_effect=fake_run) as run:
            out = fd.detect_faces(_img(), conf_thres=0.3, device="cpu",
                                  face_fallback=True)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0]["conf"], 0.8)
            # v9c 실패 → v8m 폴백 1회 추가 = 총 2회
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], "v9c")
            self.assertEqual(run.call_args_list[1].args[0], "v8m")

    def test_no_fallback_when_v9c_already_detects(self):
        def fake_run(model_key, device_key, cpu_threads, run_fn):
            return ([(10, 10, 30, 30)], [0.9])

        with patch.object(fd, "_resolve_primary_model", return_value="v9c"), \
                patch.object(fd, "_run_with_cpu_fallback", side_effect=fake_run) as run:
            out = fd.detect_faces(_img(), conf_thres=0.3, device="cpu",
                                  face_fallback=True)
            self.assertEqual(len(out), 1)
            self.assertEqual(run.call_count, 1)  # v9c가 잡았으니 v8m 미실행

    def test_fallback_skipped_when_primary_is_v8m(self):
        # v9c 모델이 없어 primary=v8m 인 경우, 폴백 분기가 다시 v8m 을 부르지 않음.
        with patch.object(fd, "_resolve_primary_model", return_value="v8m"), \
                patch.object(fd, "_run_with_cpu_fallback", return_value=([], [])) as run:
            out = fd.detect_faces(_img(), conf_thres=0.3, device="cpu",
                                  face_fallback=True)
            self.assertEqual(out, [])
            self.assertEqual(run.call_count, 1)


class TestBubbleFaceFallbackKey(unittest.TestCase):
    """postprocess 말풍선 설정에 face_fallback 키가 누락되지 않는지 회귀 가드."""

    def test_default_bubble_has_face_fallback_false(self):
        b = pp._default_bubble()
        self.assertIn("face_fallback", b)
        self.assertIs(b["face_fallback"], False)

    def test_face_models_specs(self):
        # imgsz 가 모델별로 분리되어 있는지(960/640).
        self.assertEqual(fd._FACE_MODELS["v8m"]["imgsz"], 960)
        self.assertEqual(fd._FACE_MODELS["v9c"]["imgsz"], 640)
        self.assertEqual(fd._PRIMARY_FACE_MODEL, "v9c")


if __name__ == "__main__":
    unittest.main()
