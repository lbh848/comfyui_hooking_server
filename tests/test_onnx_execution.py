import unittest
from unittest.mock import Mock, patch

from modes import onnx_execution
from modes.postprocess import _default_bubble, _default_vn


class OnnxExecutionTests(unittest.TestCase):
    def test_thread_dropdown_contains_every_logical_thread_count(self):
        with patch.object(onnx_execution.os, "cpu_count", return_value=16):
            options = onnx_execution.list_cpu_thread_options()

        self.assertEqual([item["value"] for item in options], list(range(17)))
        self.assertEqual(options[0]["label"], "자동 (ONNX Runtime)")
        self.assertEqual(options[8]["label"], "8 스레드")
        self.assertEqual(options[12]["label"], "12 스레드")
        self.assertEqual(options[16]["label"], "16 스레드")

    def test_thread_normalization_uses_current_environment_limit(self):
        with patch.object(onnx_execution.os, "cpu_count", return_value=16):
            self.assertEqual(onnx_execution.normalize_cpu_threads(0), 0)
            self.assertEqual(onnx_execution.normalize_cpu_threads(14), 14)
            self.assertEqual(onnx_execution.normalize_cpu_threads(99), 16)
            self.assertEqual(onnx_execution.normalize_cpu_threads(-3), 0)
            self.assertEqual(onnx_execution.normalize_cpu_threads("bad"), 0)

    def test_device_dropdown_tracks_installed_execution_providers(self):
        with patch.object(
            onnx_execution,
            "installed_providers",
            return_value={
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
            },
        ):
            devices = onnx_execution.list_devices()

        self.assertEqual(
            [item["key"] for item in devices],
            ["auto", "cpu", "cuda0", "dml0"],
        )

    def test_cpu_session_receives_selected_intra_op_thread_count(self):
        fake_session = Mock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        with patch.object(
            onnx_execution,
            "installed_providers",
            return_value={"CPUExecutionProvider"},
        ), patch(
            "onnxruntime.InferenceSession",
            return_value=fake_session,
        ) as inference_session:
            session, active = onnx_execution.create_session(
                "dummy.onnx",
                device_key="cpu",
                cpu_threads=12,
                log_prefix="TEST_ONNX",
            )

        self.assertIs(session, fake_session)
        self.assertEqual(active, "cpu")
        options = inference_session.call_args.kwargs["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 12)
        self.assertEqual(options.inter_op_num_threads, 1)

    def test_both_postprocess_modes_have_independent_thread_defaults(self):
        vn = _default_vn()
        bubble = _default_bubble()

        self.assertEqual(vn["face_device"], "auto")
        self.assertEqual(vn["face_cpu_threads"], 0)
        self.assertEqual(bubble["onnx_device"], "auto")
        self.assertEqual(bubble["cpu_threads"], 0)


if __name__ == "__main__":
    unittest.main()
