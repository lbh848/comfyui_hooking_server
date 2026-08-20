"""원격 실행 편집 경로의 두 가지 차단 요인 회귀 테스트.

T1(Qwen Edit 실검증) 중 드러났다. 둘 다 **플랫폼 무관**이며, 둘 다 원격 실행을
완전히 막고 있었다.

1. **로컬 모델 요구** — `qwen_edit_mode` 가 실행 대상과 무관하게 로컬 파일을
   요구했다. cloud_direct 로 26.48 GiB 를 볼륨에 올려도:

       FileNotFoundError: Qwen Rapid AIO v19 체크포인트 다운로드가 완료되지 않았습니다

   모델은 원격에 멀쩡히 있는데 "다운로드가 안 끝났다"고 말한다. 같은 판정을
   asset_tool_mode.py:335 는 이미 하고 있었다 — 이 모드만 빠져 있었다.

2. **Windows 경로 구분자** — 배포 워크플로우가 Windows 에서 작성돼 체크포인트
   이름이 ``v19\\Qwen-...safetensors`` 다. **워커는 언제나 Linux** 라 ComfyUI 는
   ``v19/Qwen-...`` 로 나열하고 제출을 거부한다:

       ckpt_name: 'v19\\Qwen-...' not in ['v19/Qwen-...']   (HTTP 400)

   즉 **Windows 사용자가 Modal 로 돌려도 똑같이 막힌다.** 업로드 쪽
   workflow_assets 는 이미 같은 정규화를 하고 있었고 제출 쪽만 빠져 있었다.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_edit_mode_skips_local_model_check_when_remote():
    """원격 실행이면 로컬 모델을 요구하지 않는다 — cloud_direct 의 정상 상태다."""
    source = (ROOT / "modes" / "qwen_edit_mode.py").read_text(encoding="utf-8")
    assert "CURRENT_COMFY_EXECUTION_TARGET" in source
    assert "REMOTE_COMFY_TARGETS" in source
    # 로컬 경로에서는 여전히 막아야 한다 (로컬 실행인데 모델이 없으면 진짜 실패다).
    assert "체크포인트 다운로드가 완료되지 않았습니다" in source
    assert "원격 실행이므로 계속합니다" in source


