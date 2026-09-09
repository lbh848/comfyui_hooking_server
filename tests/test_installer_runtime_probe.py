"""독립 환경 검증 프로브가 종료 경합으로 실패하던 문제 회귀 테스트.

무설치 상태에서 설치를 완주시키는 시험(macOS, cloud_direct) 중 **9단계
runtime_isolation 에서 설치 전체가 실패**했다. 원인은 검증 실패가 아니었다:

    명령 실패(code=-11)   ← SIGSEGV

프로브는 JSON 을 **정상적으로 다 찍은 뒤** 죽었다. macOS 크래시 리포트의
크래시 스레드가 원인을 정확히 가리켰다:

    onnxruntime_pybind11_state.so
      Microsoft::Applications::Events::PlatformAbstraction::WorkerThread::threadFunc
      Microsoft::Applications::Events::TransmissionPolicyManager::uploadAsync
      Microsoft::Applications::Events::HttpRequestEncoder::handleEncode
    EXC_BAD_ACCESS (SIGSEGV) at 0x56946

즉 onnxruntime 이 import 시 띄우는 **텔레메트리 업로드 워커 스레드**가 업로드
도중인데 메인 스레드가 인터프리터를 종료하면서 겹친 것이다. 경합이라 간헐적이고
(수동 재현 8회 중 0회), 값은 이미 다 찍힌 뒤라 **결과는 멀쩡한데 종료 코드만
실패**가 된다.

정리할 상태가 없는 일회성 측정 스크립트이므로 종료 경로를 건너뛴다.
플랫폼 무관 — 같은 경합은 어느 OS 에서도 성립한다.
"""

import re

from comfy_installer.dependency_installer import runtime_probe_script


def _script(**kwargs) -> str:
    base = {
        "requires_nvidia": False,
        "requires_triton": False,
        "requires_sageattention": False,
    }
    base.update(kwargs)
    return runtime_probe_script(**base)


def test_probe_skips_interpreter_shutdown():
    """종료 경로를 타면 onnxruntime 텔레메트리 스레드와 경합한다."""
    lines = [line.strip() for line in _script().strip().splitlines() if line.strip()]
    assert lines[-1] == "os._exit(0)"


def test_probe_flushes_before_hard_exit():
    """os._exit 는 버퍼를 비우지 않는다 — 먼저 flush 하지 않으면 결과가 사라진다."""
    lines = [line.strip() for line in _script().strip().splitlines() if line.strip()]
    exit_index = lines.index("os._exit(0)")
    flushed = lines[:exit_index]
    assert "sys.stdout.flush()" in flushed
    assert "sys.stderr.flush()" in flushed


def test_probe_prints_result_before_exiting():
    """측정값을 찍기 전에 나가면 검증 자체가 무의미해진다."""
    script = _script()
    print_index = script.index("print(json.dumps(result")
    exit_index = script.index("os._exit(0)")
    assert print_index < exit_index


def test_probe_imports_os_for_the_hard_exit():
    """import 를 빠뜨리면 NameError 로 죽어 종료 코드가 다시 실패가 된다."""
    first_line = _script().strip().splitlines()[0]
    assert re.match(r"^import .*\bos\b", first_line), first_line


def test_probe_still_reports_every_measured_field():
    """종료 처리를 바꾸면서 측정 항목이 사라지면 안 된다."""
    script = _script()
    for field in (
        "'torch':torch.__version__",
        "'cuda_available':torch.cuda.is_available()",
        "'numpy':numpy.__version__",
        "'opencv':cv2.__version__",
        "'onnxruntime':onnxruntime.__version__",
        "'insightface':insightface.__version__",
        "'user_site_enabled':site.ENABLE_USER_SITE",
    ):
        assert field in script, field


def test_cpu_profile_skips_gpu_validation():
    """NVIDIA 없는 머신에서 GPU 검증을 요구하면 설치가 통과할 수 없다."""
    script = _script()
    assert "'skipped: CPU profile'" in script
    assert "requires_nvidia=False" in script


def test_nvidia_profile_still_validates_acceleration():
    """CPU 폴백을 넣느라 NVIDIA 경로의 검증이 약해지면 안 된다."""
    script = _script(requires_nvidia=True, requires_sageattention=True)
    assert "requires_nvidia=True" in script
    assert "requires_sageattention=True" in script
    assert "import sageattention" in script
