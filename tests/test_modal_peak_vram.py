"""생성 중 피크 VRAM 보고.

GPU 등급마다 VRAM 이 달라 어떤 워크플로우에 어떤 등급이 필요한지 추측이었다.
계측은 부가 정보라 실패해도 생성을 망치면 안 된다.
"""

import re
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
).read_text(encoding="utf-8")
SAMPLER = re.search(
    r"def _sample_gpu_memory\(.*?(?=\n@|\ndef |\nclass )", SOURCE, re.S
).group(0)


def test_peak_vram_is_returned_with_the_result():
    assert '"peak_vram"' in SOURCE


def test_device_wide_probe_is_used_not_torch():
    """ComfyUI 는 워커 안에서 별도 프로세스로 돈다. 워커의 torch 통계로는
    실제 사용량이 보이지 않아 장치 전체를 봐야 한다."""
    assert "nvidia-smi" in SAMPLER
    assert "torch.cuda.max_memory" not in SAMPLER


def test_peak_is_kept_not_the_last_sample():
    assert "max(" in SAMPLER or "peak =" in SAMPLER


def test_sampling_failure_is_swallowed():
    assert "except" in SAMPLER


def test_sampler_runs_as_a_daemon_thread():
    """생성이 끝났는데 표집 스레드가 종료를 막으면 안 된다."""
    assert "daemon=True" in SOURCE
    assert "vram_thread.join(timeout=" in SOURCE
