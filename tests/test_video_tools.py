from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

import ensure_video_tools as tools


def test_run_en_prepares_video_tools_inside_uv_project_environment() -> None:
    source = (Path(__file__).resolve().parents[1] / "run_en.bat").read_text(
        encoding="utf-8"
    )

    uv_sync = source.index('"%UV_EXE%" sync')
    video_tools = source.index(
        '"%UV_EXE%" run --no-sync python ensure_video_tools.py'
    )
    server = source.index('"%UV_EXE%" run --no-sync python server.py')

    assert uv_sync < video_tools < server
    assert r'set "PATH=%CD%\.tools\ffmpeg\bin;%PATH%"' in source
    assert "Failed to prepare Real-ESRGAN or FFmpeg" in source


def test_video_runtime_resolves_project_local_ffmpeg() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "modes" / "video_postprocess.py"
    ).read_text(encoding="utf-8")

    assert "ensure_ffmpeg as _ensure_ffmpeg_sync" in source
    assert "ffmpeg = str(await ensure_ffmpeg())" in source
    assert 'shutil.which("ffmpeg")' not in source


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.exe", b"unsafe")

    with pytest.raises(RuntimeError, match="ZIP 경로 검증"):
        tools._extract_zip_safely(archive_path, tmp_path / "target", "TEST")

    assert not (tmp_path / "outside.exe").exists()


def test_ffmpeg_validation_requires_avif_and_animation_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"ffmpeg")
    ffprobe.write_bytes(b"ffprobe")
    monkeypatch.setattr(tools, "FFMPEG_EXE", ffmpeg)
    monkeypatch.setattr(tools, "FFPROBE_EXE", ffprobe)

    def complete_output(command: list[str], _label: str) -> str:
        if "-version" in command:
            return f"ffmpeg version {tools.FFMPEG_VERSION}"
        if "-encoders" in command:
            return "libaom-av1\nlibwebp_anim"
        if "-muxers" in command:
            return "avif"
        raise AssertionError(command)

    monkeypatch.setattr(tools, "_run_checked", complete_output)
    assert tools._ffmpeg_validation_error(smoke_test=False) is None

    monkeypatch.setattr(
        tools,
        "_run_checked",
        lambda command, _label: (
            f"ffmpeg version {tools.FFMPEG_VERSION}"
            if "-version" in command
            else "libwebp_anim"
        ),
    )
    assert "AV1 encoder" in str(
        tools._ffmpeg_validation_error(smoke_test=False)
    )


def test_ensure_video_tools_requires_both_native_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realesrgan = tmp_path / "realesrgan.exe"
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    monkeypatch.setattr(tools, "ensure_realesrgan", lambda: realesrgan)
    monkeypatch.setattr(tools, "ensure_ffmpeg", lambda: ffmpeg)
    monkeypatch.setattr(tools, "FFPROBE_EXE", ffprobe)

    assert tools.ensure_video_tools() == {
        "realesrgan": str(realesrgan),
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
    }
