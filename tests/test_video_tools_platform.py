"""영상 도구의 플랫폼 분기 계약.

원래는 Windows 가 아니면 세 함수가 즉시 RuntimeError 를 던졌다. 영상 후처리가
기본 켜짐이라 그 OS 에서는 기능이 시작하자마자 죽는다.
"""

from pathlib import Path

import pytest

import ensure_video_tools as tools


def test_windows_keeps_project_local_tools(monkeypatch):
    monkeypatch.setattr(tools, "IS_WINDOWS", True)
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    local = Path("C:/x/ffmpeg.exe")
    assert tools._resolve_tool("ffmpeg", local) == local


def test_other_systems_use_path(monkeypatch):
    monkeypatch.setattr(tools, "IS_WINDOWS", False)
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/opt/bin/ffmpeg")
    assert tools._resolve_tool("ffmpeg", Path("/x/ffmpeg")) == Path("/opt/bin/ffmpeg")


def test_missing_from_path_falls_back_to_project_local(monkeypatch):
    """검증 단계에서 '누락' 으로 잡히도록 존재하지 않는 경로를 돌려준다."""
    monkeypatch.setattr(tools, "IS_WINDOWS", False)
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    local = Path("/x/ffmpeg")
    assert tools._resolve_tool("ffmpeg", local) == local


def test_realesrgan_assets_are_pinned():
    for platform_key, (archive, digest) in tools.REALESRGAN_PACKAGES.items():
        assert archive.endswith(".zip"), platform_key
        assert len(digest) == 64, platform_key


def test_unsupported_platform_names_the_supported_ones(monkeypatch):
    monkeypatch.setattr(tools, "_REALESRGAN_PACKAGE", None)
    with pytest.raises(RuntimeError) as excinfo:
        tools.ensure_realesrgan()
    for platform_key in tools.REALESRGAN_PACKAGES:
        assert platform_key in str(excinfo.value)


def test_system_validator_does_not_check_the_pinned_version():
    """시스템 ffmpeg 은 별도 검증기를 쓴다. 기존 핀 버전 계약은 그대로 둔다.

    Homebrew 빌드는 gyan.dev 핀 버전과 절대 일치하지 않는다. 기존 검증기를
    고치면 Windows 계약(테스트가 단언한다)이 깨지므로 검증기를 나눴다.
    """
    import inspect

    assert "FFMPEG_VERSION" in inspect.getsource(tools._ffmpeg_validation_error)
    assert "FFMPEG_VERSION" not in inspect.getsource(
        tools._system_ffmpeg_validation_error
    )
