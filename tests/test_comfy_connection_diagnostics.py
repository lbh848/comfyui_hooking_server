"""로컬 ComfyUI 미실행 시 진단 품질 회귀 테스트.

배경: Modal 전용 구성(macOS 등)에서는 로컬 ComfyUI가 아예 없다. 그런데
- 변환 경로가 aiohttp 영문 원문을 그대로 노출했고(제출 경로만 한국어 안내를 썼다),
- "ComfyUI가 실행 중인지 확인하세요"라는 안내는 켤 대상이 없는 구성에선 오답이며,
- 기동 시 배분 점검이 없어 눌러봐야만 실패를 알 수 있었다.
"""

import io
import re
from pathlib import Path

import pytest

SERVER = Path(__file__).parents[1] / "server.py"


def _source() -> str:
    return io.open(SERVER, encoding="utf-8-sig").read()


def _slice_function(source: str, name: str) -> str:
    """def 부터 다음 최상위 def/class 직전까지 잘라낸다.

    여러 줄 시그니처는 닫는 `):` 가 0열에 오므로 "들여쓰기 여부"만으로는
    끊을 수 없다. 다음 최상위 정의를 만날 때까지 이어 붙인다.
    """

    start = source.index(f"def {name}(")
    lines = source[start:].splitlines(keepends=True)
    body = [lines[0]]
    for line in lines[1:]:
        if re.match(r"^(async\s+def|def|class)\s", line):
            break
        body.append(line)
    return "".join(body)


def test_conversion_failure_uses_korean_helper_not_raw_aiohttp_text():
    """변환 실패가 str(e) 대신 한국어 안내 헬퍼를 쓴다."""
    body = _slice_function(_source(), "convert_workflow_via_endpoint")
    assert "_comfy_connection_error_message(" in body, (
        "변환 경로가 한국어 안내 헬퍼를 쓰지 않는다 — aiohttp 영문 원문이 노출된다"
    )
    tail = body[body.index("except aiohttp.ClientError"):]
    assert "return None, str(e)" not in tail, (
        "변환 실패가 aiohttp 예외 문자열을 그대로 반환한다"
    )


def test_connection_message_is_allocation_aware():
    """원격 지원 작업이면 'MODAL로 바꾸라'고 안내한다."""
    body = _slice_function(_source(), "_comfy_connection_error_message")
    assert "task_key" in body, "안내가 배분(task_key)을 보지 않는다"
    assert "MODAL_SUPPORTED_COMFY_TASK_KEYS" in body, (
        "원격 실행 가능 여부를 판단하지 않는다"
    )
    assert "MODAL" in body


def test_startup_preflight_exists_and_is_wired():
    """기동 시 배분 점검이 있고 on_startup에 연결돼 있다."""
    source = _source()
    assert "def _comfy_allocation_preflight(" in source
    startup = _slice_function(source, "on_startup")
    assert "_log_comfy_allocation_preflight" in startup, (
        "프리플라이트가 on_startup에 연결돼 있지 않다"
    )


def test_preflight_never_blocks_startup():
    """진단 코드가 기동을 깨뜨리면 안 된다 — 예외를 삼켜야 한다."""
    body = _slice_function(_source(), "_log_comfy_allocation_preflight")
    assert "except Exception" in body
    assert "raise" not in body


def test_preflight_skips_remote_targets():
    """Modal/Vast로 배분된 작업은 로컬 포트를 검사하지 않는다."""
    body = _slice_function(_source(), "_comfy_allocation_preflight")
    assert "REMOTE_COMFY_TARGETS" in body
    assert "continue" in body
