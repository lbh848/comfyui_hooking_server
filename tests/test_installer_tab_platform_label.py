"""설치기 탭의 플랫폼 표기.

설치 게이트에 macOS 를 추가했는데 화면이 "Windows 전용" 이라고 표시하면, 되는
기능을 안 된다고 읽게 된다. 게이트와 화면이 어긋나지 않도록 함께 잠근다.
"""

from pathlib import Path

from comfy_installer.system_probe import SUPPORTED_INSTALL_PLATFORMS

HTML = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def test_badge_matches_the_supported_platforms():
    assert "Windows · macOS" in HTML
    assert '<span class="comfy-installer-badge">Windows 전용</span>' not in HTML


def test_every_supported_platform_is_named_somewhere():
    names = {"Windows": "Windows", "Darwin": "macOS"}
    for platform in SUPPORTED_INSTALL_PLATFORMS:
        assert names[platform] in HTML, f"{platform} 표기가 없습니다"


def test_probe_message_is_not_windows_only():
    assert "Windows, GPU, 드라이버" not in HTML
    assert "OS, GPU, 드라이버" in HTML
