"""Install and verify project-local native tools used by video post-processing."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.request
import uuid
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = PROJECT_ROOT / ".tools"

REALESRGAN_VERSION = "0.2.5.0"
REALESRGAN_ARCHIVE_NAME = "realesrgan-ncnn-vulkan-20220424-windows.zip"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    + REALESRGAN_ARCHIVE_NAME
)
REALESRGAN_SHA256 = (
    "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
)
REALESRGAN_DIR = TOOLS_ROOT / "realesrgan-ncnn-vulkan"
REALESRGAN_EXE = REALESRGAN_DIR / "realesrgan-ncnn-vulkan.exe"

FFMPEG_VERSION = "8.1.2"
FFMPEG_ARCHIVE_NAME = f"ffmpeg-{FFMPEG_VERSION}-essentials_build.zip"
FFMPEG_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/" + FFMPEG_ARCHIVE_NAME
)
FFMPEG_SHA256 = (
    "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
)
FFMPEG_DIR = TOOLS_ROOT / "ffmpeg"
FFMPEG_EXE = FFMPEG_DIR / "bin" / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "bin" / "ffprobe.exe"

_INSTALL_LOCK = threading.RLock()
_USER_AGENT = "comfyui-hooking-server-video-tools"
_REALESRGAN_VERIFIED = False
_FFMPEG_VERIFIED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    resolved_parent = parent.resolve()
    try:
        return os.path.commonpath([str(resolved), str(resolved_parent)]) == str(
            resolved_parent
        )
    except ValueError as exc:
        print(
            "[VIDEO_TOOLS][ERROR] 경로 공통 루트 검증 실패: "
            f"path={str(resolved)!r}, parent={str(resolved_parent)!r}, error={exc}"
        )
        traceback.print_exc()
        return False


def _remove_exact_path(path: Path, allowed_parent: Path) -> None:
    if not path.exists():
        return
    if not _is_within(path, allowed_parent) or path.resolve() == allowed_parent.resolve():
        print(
            "[VIDEO_TOOLS][ERROR] 안전하지 않은 삭제 경로 거부: "
            f"path={str(path)!r}, parent={str(allowed_parent)!r}"
        )
        raise RuntimeError("영상 도구 삭제 경로 안전 검증에 실패했습니다")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _download_verified(
    *,
    url: str,
    expected_sha256: str,
    archive_path: Path,
    label: str,
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        actual_hash = _sha256(archive_path)
        if actual_hash == expected_sha256:
            print(f"[VIDEO_TOOLS:{label}] 다운로드 캐시 검증 완료: {archive_path}")
            return archive_path
        print(
            f"[VIDEO_TOOLS:{label}] 캐시 무결성 불일치, 다시 다운로드: "
            f"path={archive_path}, expected={expected_sha256}, actual={actual_hash}"
        )
        _remove_exact_path(archive_path, TOOLS_ROOT)

    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    _remove_exact_path(part_path, TOOLS_ROOT)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    print(f"[VIDEO_TOOLS:{label}] 다운로드 시작: url={url}")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with part_path.open("xb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
        actual_hash = _sha256(part_path)
        if actual_hash != expected_sha256:
            print(
                f"[VIDEO_TOOLS:{label}][ERROR] 다운로드 무결성 검증 실패: "
                f"expected={expected_sha256}, actual={actual_hash}, path={part_path}"
            )
            raise RuntimeError(f"{label} 패키지 SHA-256 검증에 실패했습니다")
        os.replace(part_path, archive_path)
        print(
            f"[VIDEO_TOOLS:{label}] 다운로드 검증 완료: "
            f"bytes={archive_path.stat().st_size:,}, sha256={actual_hash}"
        )
        return archive_path
    except Exception as exc:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 다운로드 실패: "
            f"url={url}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        try:
            _remove_exact_path(part_path, TOOLS_ROOT)
        except Exception as cleanup_exc:
            print(
                f"[VIDEO_TOOLS:{label}][ERROR] 부분 다운로드 정리 실패: "
                f"path={part_path}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise


def _extract_zip_safely(archive_path: Path, target_dir: Path, label: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=False)
    target_root = target_dir.resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                destination = (target_root / entry.filename).resolve()
                if not _is_within(destination, target_root):
                    print(
                        f"[VIDEO_TOOLS:{label}][ERROR] 안전하지 않은 ZIP 항목 거부: "
                        f"entry={entry.filename!r}"
                    )
                    raise RuntimeError(f"{label} ZIP 경로 검증에 실패했습니다")
            archive.extractall(target_root)
    except Exception as exc:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 압축 해제 실패: "
            f"archive={archive_path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _replace_directory(staged_dir: Path, target_dir: Path, label: str) -> None:
    if not _is_within(target_dir, TOOLS_ROOT) or target_dir.resolve() == TOOLS_ROOT.resolve():
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 안전하지 않은 설치 대상 거부: "
            f"target={target_dir}, root={TOOLS_ROOT}"
        )
        raise RuntimeError(f"{label} 설치 대상 경로 검증에 실패했습니다")

    backup_dir = target_dir.with_name(
        f".{target_dir.name}.replace-{uuid.uuid4().hex[:8]}"
    )
    moved_existing = False
    try:
        if target_dir.exists():
            os.replace(target_dir, backup_dir)
            moved_existing = True
        os.replace(staged_dir, target_dir)
    except Exception as exc:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 설치 폴더 교체 실패: "
            f"target={target_dir}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if moved_existing and backup_dir.exists() and not target_dir.exists():
            try:
                os.replace(backup_dir, target_dir)
                print(f"[VIDEO_TOOLS:{label}] 기존 설치를 복구했습니다: {target_dir}")
            except Exception as restore_exc:
                print(
                    f"[VIDEO_TOOLS:{label}][ERROR] 기존 설치 복구 실패: "
                    f"backup={backup_dir}, error={restore_exc}"
                )
                traceback.print_exc()
        raise
    else:
        if backup_dir.exists():
            try:
                _remove_exact_path(backup_dir, TOOLS_ROOT)
            except Exception as cleanup_exc:
                print(
                    f"[VIDEO_TOOLS:{label}][ERROR] 교체 백업 정리 실패: "
                    f"path={backup_dir}, error={cleanup_exc}"
                )
                traceback.print_exc()


def _realesrgan_validation_error() -> str | None:
    required = [REALESRGAN_EXE]
    for scale in (2, 3, 4):
        required.extend(
            [
                REALESRGAN_DIR / "models" / f"realesr-animevideov3-x{scale}.param",
                REALESRGAN_DIR / "models" / f"realesr-animevideov3-x{scale}.bin",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return "필수 파일 누락: " + ", ".join(missing)
    return None


def ensure_realesrgan() -> Path:
    global _REALESRGAN_VERIFIED
    with _INSTALL_LOCK:
        if os.name != "nt":
            message = "Real-ESRGAN 자동 설치는 Windows에서만 지원합니다"
            print(f"[VIDEO_TOOLS:REALESRGAN][ERROR] {message}: os.name={os.name!r}")
            raise RuntimeError(message)

        if _REALESRGAN_VERIFIED and REALESRGAN_EXE.is_file():
            return REALESRGAN_EXE
        validation_error = _realesrgan_validation_error()
        if validation_error is None:
            _REALESRGAN_VERIFIED = True
            print(
                f"[VIDEO_TOOLS:REALESRGAN] 설치 검증 완료: "
                f"version={REALESRGAN_VERSION}, exe={REALESRGAN_EXE}"
            )
            return REALESRGAN_EXE
        print(f"[VIDEO_TOOLS:REALESRGAN] 설치 필요: {validation_error}")

        archive_path = _download_verified(
            url=REALESRGAN_URL,
            expected_sha256=REALESRGAN_SHA256,
            archive_path=TOOLS_ROOT / REALESRGAN_ARCHIVE_NAME,
            label="REALESRGAN",
        )
        TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="realesrgan_install_", dir=str(TOOLS_ROOT)
            ) as temp_name:
                extract_dir = Path(temp_name) / "extracted"
                _extract_zip_safely(archive_path, extract_dir, "REALESRGAN")
                staged_error = None
                staged_exe = extract_dir / "realesrgan-ncnn-vulkan.exe"
                if not staged_exe.is_file():
                    staged_error = f"실행 파일 누락: {staged_exe}"
                if staged_error:
                    print(f"[VIDEO_TOOLS:REALESRGAN][ERROR] {staged_error}")
                    raise RuntimeError("Real-ESRGAN 패키지 구성이 올바르지 않습니다")
                _replace_directory(extract_dir, REALESRGAN_DIR, "REALESRGAN")
        except Exception as exc:
            print(
                "[VIDEO_TOOLS:REALESRGAN][ERROR] 설치 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        validation_error = _realesrgan_validation_error()
        if validation_error is not None:
            print(
                "[VIDEO_TOOLS:REALESRGAN][ERROR] 최종 설치 검증 실패: "
                f"{validation_error}"
            )
            raise RuntimeError("Real-ESRGAN 최종 설치 검증에 실패했습니다")
        print(
            f"[VIDEO_TOOLS:REALESRGAN] 설치 완료: "
            f"version={REALESRGAN_VERSION}, exe={REALESRGAN_EXE}"
        )
        _REALESRGAN_VERIFIED = True
        return REALESRGAN_EXE


def _run_checked(command: list[str], label: str) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except Exception as exc:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 검증 명령 실행 실패: "
            f"command={command!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 검증 명령 실패: "
            f"returncode={completed.returncode}, command={command!r}, "
            f"output={output[-4000:]}"
        )
        raise RuntimeError(f"{label} 검증 명령이 실패했습니다")
    return output


def _ffmpeg_validation_error(*, smoke_test: bool = True) -> str | None:
    if not FFMPEG_EXE.is_file():
        return f"ffmpeg.exe 누락: {FFMPEG_EXE}"
    if not FFPROBE_EXE.is_file():
        return f"ffprobe.exe 누락: {FFPROBE_EXE}"
    try:
        version_output = _run_checked(
            [str(FFMPEG_EXE), "-hide_banner", "-version"], "FFMPEG"
        )
        encoder_output = _run_checked(
            [str(FFMPEG_EXE), "-hide_banner", "-encoders"], "FFMPEG"
        )
        muxer_output = _run_checked(
            [str(FFMPEG_EXE), "-hide_banner", "-muxers"], "FFMPEG"
        )
        required_tokens = {
            "version": (version_output.lower(), f"ffmpeg version {FFMPEG_VERSION}"),
            "AV1 encoder": (encoder_output, "libaom-av1"),
            "animated WebP encoder": (encoder_output, "libwebp_anim"),
            "AVIF muxer": (muxer_output.lower(), "avif"),
        }
        missing = [
            label
            for label, (haystack, needle) in required_tokens.items()
            if needle not in haystack
        ]
        if missing:
            return "필수 기능 누락: " + ", ".join(missing)

        if smoke_test:
            with tempfile.TemporaryDirectory(prefix="ffmpeg_avif_verify_") as temp_name:
                output_path = Path(temp_name) / "verify.avif"
                _run_checked(
                    [
                        str(FFMPEG_EXE),
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=black:s=16x16:r=2:d=1",
                        "-frames:v",
                        "2",
                        "-c:v",
                        "libaom-av1",
                        "-crf",
                        "50",
                        "-b:v",
                        "0",
                        "-loop",
                        "0",
                        "-f",
                        "avif",
                        str(output_path),
                    ],
                    "FFMPEG_AVIF",
                )
                if not output_path.is_file() or output_path.stat().st_size <= 0:
                    return f"AVIF 스모크 출력 없음: {output_path}"
    except Exception as exc:
        print(
            "[VIDEO_TOOLS:FFMPEG][ERROR] 기능 검증 중 예외: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return f"기능 검증 실패: {type(exc).__name__}: {exc}"
    return None


def ensure_ffmpeg() -> Path:
    global _FFMPEG_VERIFIED
    with _INSTALL_LOCK:
        if os.name != "nt":
            message = "FFmpeg 자동 설치는 Windows에서만 지원합니다"
            print(f"[VIDEO_TOOLS:FFMPEG][ERROR] {message}: os.name={os.name!r}")
            raise RuntimeError(message)

        if _FFMPEG_VERIFIED and FFMPEG_EXE.is_file():
            return FFMPEG_EXE
        validation_error = _ffmpeg_validation_error()
        if validation_error is None:
            _FFMPEG_VERIFIED = True
            print(
                f"[VIDEO_TOOLS:FFMPEG] 설치 및 AVIF 검증 완료: "
                f"version={FFMPEG_VERSION}, exe={FFMPEG_EXE}"
            )
            return FFMPEG_EXE
        print(f"[VIDEO_TOOLS:FFMPEG] 설치 필요: {validation_error}")

        archive_path = _download_verified(
            url=FFMPEG_URL,
            expected_sha256=FFMPEG_SHA256,
            archive_path=TOOLS_ROOT / FFMPEG_ARCHIVE_NAME,
            label="FFMPEG",
        )
        TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="ffmpeg_install_", dir=str(TOOLS_ROOT)
            ) as temp_name:
                extract_dir = Path(temp_name) / "extracted"
                _extract_zip_safely(archive_path, extract_dir, "FFMPEG")
                candidates = sorted(extract_dir.glob("*/bin/ffmpeg.exe"))
                if len(candidates) != 1:
                    print(
                        "[VIDEO_TOOLS:FFMPEG][ERROR] 압축 내 실행 파일 탐색 실패: "
                        f"candidates={[str(path) for path in candidates]!r}"
                    )
                    raise RuntimeError("FFmpeg 패키지 구성이 올바르지 않습니다")
                package_root = candidates[0].parent.parent
                _replace_directory(package_root, FFMPEG_DIR, "FFMPEG")
        except Exception as exc:
            print(
                "[VIDEO_TOOLS:FFMPEG][ERROR] 설치 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        validation_error = _ffmpeg_validation_error()
        if validation_error is not None:
            print(
                "[VIDEO_TOOLS:FFMPEG][ERROR] 최종 설치 검증 실패: "
                f"{validation_error}"
            )
            raise RuntimeError("FFmpeg 최종 설치 검증에 실패했습니다")
        print(
            f"[VIDEO_TOOLS:FFMPEG] 설치 완료: "
            f"version={FFMPEG_VERSION}, exe={FFMPEG_EXE}"
        )
        _FFMPEG_VERIFIED = True
        return FFMPEG_EXE


def ensure_video_tools() -> dict[str, str]:
    try:
        realesrgan = ensure_realesrgan()
        ffmpeg = ensure_ffmpeg()
        result = {
            "realesrgan": str(realesrgan),
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(FFPROBE_EXE),
        }
        print(
            "[VIDEO_TOOLS] 모든 영상 후처리 도구 준비 완료: "
            f"realesrgan={realesrgan}, ffmpeg={ffmpeg}"
        )
        return result
    except Exception as exc:
        print(
            "[VIDEO_TOOLS][ERROR] 영상 후처리 도구 준비 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        ensure_video_tools()
    except Exception:
        sys.exit(1)
    sys.exit(0)
