"""설치기 플랫폼 게이트 회귀 테스트.

게이트가 Windows 하드코딩이던 동안 macOS 설치는 1단계에서 막혔다. 게이트 아래
단계는 이미 플랫폼 중립이라(NVIDIA 부재는 cpu_fallback 이 처리하고 macOS arm64
torch 휠은 download.pytorch.org/whl/cpu 에 있다) 게이트만 풀면 통과한다.

지원 플랫폼의 판정을 문구로 단언하지 않는다. 문구를 보면 옛 게이트의 다른 문구도
"게이트에 안 걸림" 으로 읽혀 되돌려도 통과한다. 대신 Windows 와 결과가 같은지를
본다 — uv/git 이 없는 환경이면 둘 다 같은 이유로 실패하므로 오탐도 없다.
"""

import platform

import pytest

from comfy_installer.manifest import load_install_manifest
from comfy_installer.system_probe import (
    SUPPORTED_INSTALL_PLATFORMS,
    SystemProbeError,
    probe_system,
)


def _outcome(root, monkeypatch, system):
    """검사 결과를 플랫폼 이름과 무관하게 비교할 수 있는 형태로 돌려준다."""
    monkeypatch.setattr(platform, "system", lambda: system)
    try:
        probe_system(root, load_install_manifest(), require_disk=False)
    except SystemProbeError as exc:
        return str(exc).replace(system, "<플랫폼>")
    return None


@pytest.mark.parametrize(
    "system", sorted(SUPPORTED_INSTALL_PLATFORMS - {"Windows"})
)
def test_supported_platform_behaves_like_windows(tmp_path, monkeypatch, system):
    assert _outcome(tmp_path, monkeypatch, system) == _outcome(
        tmp_path, monkeypatch, "Windows"
    )


def test_unsupported_platform_is_rejected_with_its_name(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Plan9")
    with pytest.raises(SystemProbeError) as excinfo:
        probe_system(tmp_path, load_install_manifest(), require_disk=False)
    assert "Plan9" in str(excinfo.value)


def test_windows_stays_supported():
    assert "Windows" in SUPPORTED_INSTALL_PLATFORMS
