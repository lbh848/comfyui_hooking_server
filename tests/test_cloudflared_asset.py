"""cloudflared 릴리스 자산 선택 회귀 테스트.

"Windows 가 아니면 리눅스" 로 갈라 macOS 에서 Linux ELF 를 받아 실행이 불가능했다.
아키텍처도 보지 않아 arm64 리눅스에 amd64 를 받았다.

자산 이름은 릴리스에 실제로 있는 것과 정확히 같아야 하므로 표로 잠근다.
"""

import pytest

import server

ASSETS = [
    ("Windows", "AMD64", "cloudflared-windows-amd64.exe", "cloudflared.exe"),
    ("Windows", "ARM64", "cloudflared-windows-amd64.exe", "cloudflared.exe"),
    ("Darwin", "arm64", "cloudflared-darwin-arm64.tgz", "cloudflared"),
    ("Darwin", "x86_64", "cloudflared-darwin-amd64.tgz", "cloudflared"),
    ("Linux", "x86_64", "cloudflared-linux-amd64", "cloudflared"),
    ("Linux", "aarch64", "cloudflared-linux-arm64", "cloudflared"),
    ("Linux", "armv7l", "cloudflared-linux-arm", "cloudflared"),
]


@pytest.mark.parametrize("system,machine,asset,filename", ASSETS)
def test_asset_matrix(system, machine, asset, filename):
    assert server._cloudflared_asset(system, machine) == (asset, filename)


def test_only_macos_assets_are_archives():
    """아카이브 여부로 추출 경로가 갈린다. macOS 자산만 tgz 다."""
    for system, _, asset, _ in ASSETS:
        assert asset.endswith(".tgz") is (system == "Darwin"), asset
