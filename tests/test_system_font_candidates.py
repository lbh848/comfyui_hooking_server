"""시스템 폰트 후보 목록 계약.

후보가 전부 미스면 PIL 비트맵 폰트로 떨어져 크기 지정이 무시된다(48px 요청이
10px 로 렌더). macOS 경로가 없어 실제로 그렇게 되고 있었다.

목록이 세 벌로 나뉘어 있어 한 곳만 고치면 조용히 어긋난다. 여기서 함께 잠근다.
"""

import inspect

import pytest

from modes import bubble_layout, bubble_render, font_assets, postprocess

WINDOWS_PREFIX = "C:/Windows/Fonts/"
MACOS_PREFIX = "/System/Library/Fonts/"
LINUX_PREFIX = "/usr/share/fonts/"

LISTS = {
    "font_assets": font_assets.SYSTEM_FONT_CANDIDATES,
    "bubble_render": bubble_render._SYSTEM_FONT_CANDIDATES,
    "postprocess": postprocess._FONT_CANDIDATES,
}


def _group(path: str) -> str:
    if path.startswith(WINDOWS_PREFIX) or path.startswith("C:\\Windows"):
        return "windows"
    if path.startswith(MACOS_PREFIX):
        return "macos"
    if path.startswith(LINUX_PREFIX):
        return "linux"
    return "other"


@pytest.mark.parametrize("name", sorted(LISTS))
def test_every_platform_has_a_candidate(name):
    groups = {_group(p) for p in LISTS[name]}
    assert {"windows", "macos", "linux"} <= groups, (
        f"{name} 후보에 빠진 플랫폼이 있습니다: {groups}"
    )


@pytest.mark.parametrize("name", sorted(LISTS))
def test_windows_candidates_stay_first(name):
    """Windows 는 기존 선택이 그대로여야 한다 — 뒤에 덧붙이기만 한다."""
    order = [_group(p) for p in LISTS[name] if _group(p) != "other"]
    assert order == sorted(order, key=["windows", "macos", "linux"].index), (
        f"{name} 후보 순서가 Windows → macOS → Linux 가 아닙니다: {order}"
    )


def test_duplicated_lists_stay_in_sync():
    assert (
        font_assets.SYSTEM_FONT_CANDIDATES
        == bubble_render._SYSTEM_FONT_CANDIDATES
    )


def test_korean_first_list_has_a_macos_candidate():
    """bubble_layout 의 후보는 함수 안 튜플이라 소스로 확인한다.

    이 목록은 CJK 우선이라 한글 폰트만 넣는다(Arial 계열은 넣지 않는다).
    """
    source = inspect.getsource(bubble_layout.load_font)
    assert "/System/Library/Fonts/AppleSDGothicNeo.ttc" in source
    assert source.index("C:") < source.index(MACOS_PREFIX) < source.index(
        LINUX_PREFIX
    )
