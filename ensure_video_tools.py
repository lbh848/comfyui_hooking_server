"""Install and verify project-local native tools used by video post-processing."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
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

# Anime4K의 mpv 호환 GLSL 셰이더를 FFmpeg libplacebo로 실행할 때만 설치한다.
# 기본 AVIF/WebP 인코딩은 위 essentials 빌드를 계속 사용한다.
FFMPEG_FULL_ARCHIVE_NAME = f"ffmpeg-{FFMPEG_VERSION}-full_build.7z"
FFMPEG_FULL_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/" + FFMPEG_FULL_ARCHIVE_NAME
)
FFMPEG_FULL_SHA256 = (
    "0fff188997a499b5382e0f66e845d4556c48c54f0113ebed4853d556dbdd7059"
)
FFMPEG_FULL_DIR = TOOLS_ROOT / "ffmpeg-full"
FFMPEG_FULL_EXE = FFMPEG_FULL_DIR / "bin" / "ffmpeg.exe"

ANIME4K_VERSION = "4.0.1"
ANIME4K_DIR = TOOLS_ROOT / f"anime4k-{ANIME4K_VERSION}"
ANIME4K_SHADER = ANIME4K_DIR / "Anime4K_Fast_M.glsl"
ANIME4K_SHADER_SHA256 = (
    "f44632616775fed96e46d9e393eda71332a9cfbcf1e3329d84c0a802a71f9dba"
)
ANIME4K_RESOURCES = (
    (
        "glsl/Restore/Anime4K_Clamp_Highlights.glsl",
        "a2a9bf7fbc1d75d09660ca2e701e4d7fb0cf5457b94da47e1825032fa2b3671a",
    ),
    (
        "glsl/Restore/Anime4K_Restore_CNN_M.glsl",
        "67ea3ed26539e8de3b7d307688535d2ff17e8d147e11dda0247da7770dbecf41",
    ),
    (
        "glsl/Upscale/Anime4K_Upscale_CNN_x2_M.glsl",
        "716e02098a68f0d648761f2b96b4dd139e1cb09b174bb369fca3aa34328fff7e",
    ),
    (
        "glsl/Upscale/Anime4K_AutoDownscalePre_x2.glsl",
        "8c58291740146bd766a4d73f132775a797fe80f7d07919b5d767e27a5dc85656",
    ),
    (
        "glsl/Upscale/Anime4K_AutoDownscalePre_x4.glsl",
        "5af62d8cd844916dc1126613e13bad3beab195787f93a71200b47c6ec78f2e41",
    ),
    (
        "glsl/Upscale/Anime4K_Upscale_CNN_x2_S.glsl",
        "4c53ec2e287908f7ee7bcB266b0170421626d663576468b7d7dafc62962649a4".lower(),
    ),
)
ANIME4K_LICENSE = (
    "LICENSE",
    "5bad448b737378e3d0c977ad0d0521fa37ad279a7e76ea9a31d9257eeb6953f5",
)

_INSTALL_LOCK = threading.RLock()
_USER_AGENT = "comfyui-hooking-server-video-tools"
_REALESRGAN_VERIFIED = False
_FFMPEG_VERIFIED = False
_FFMPEG_FULL_VERIFIED = False
_ANIME4K_VERIFIED = False


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


def _extract_7z_safely(archive_path: Path, target_dir: Path, label: str) -> None:
    """Windows 기본 bsdtar로 7z 항목을 검증한 뒤 정확한 대상에 푼다."""

    tar_executable = shutil.which("tar")
    if not tar_executable:
        print(f"[VIDEO_TOOLS:{label}][ERROR] 7z 압축 해제용 tar 실행 파일 없음")
        raise RuntimeError("영상 도구 7z 압축을 해제할 tar 실행 파일이 없습니다")
    try:
        listing = _run_checked(
            [tar_executable, "-tf", str(archive_path)],
            f"{label}_LIST",
            timeout=180,
        )
        entries = [line.strip() for line in listing.splitlines() if line.strip()]
        if not entries:
            print(f"[VIDEO_TOOLS:{label}][ERROR] 7z 압축 항목이 비어 있음: {archive_path}")
            raise RuntimeError(f"{label} 7z 패키지가 비어 있습니다")
        target_dir.mkdir(parents=True, exist_ok=False)
        target_root = target_dir.resolve()
        for entry in entries:
            normalized = entry.replace("\\", "/")
            if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
                print(f"[VIDEO_TOOLS:{label}][ERROR] 절대 경로 7z 항목 거부: {entry!r}")
                raise RuntimeError(f"{label} 7z 경로 검증에 실패했습니다")
            destination = (target_root / normalized).resolve()
            if not _is_within(destination, target_root):
                print(f"[VIDEO_TOOLS:{label}][ERROR] 안전하지 않은 7z 항목 거부: {entry!r}")
                raise RuntimeError(f"{label} 7z 경로 검증에 실패했습니다")
        _run_checked(
            [tar_executable, "-xf", str(archive_path), "-C", str(target_root)],
            f"{label}_EXTRACT",
            timeout=300,
        )
    except Exception as exc:
        print(
            f"[VIDEO_TOOLS:{label}][ERROR] 7z 압축 해제 실패: "
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


def _run_checked(command: list[str], label: str, *, timeout: int = 60) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
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


def _ffmpeg_full_validation_error() -> str | None:
    if not FFMPEG_FULL_EXE.is_file():
        return f"ffmpeg full 실행 파일 누락: {FFMPEG_FULL_EXE}"
    try:
        version_output = _run_checked(
            [str(FFMPEG_FULL_EXE), "-hide_banner", "-version"],
            "FFMPEG_FULL",
        )
        filter_output = _run_checked(
            [str(FFMPEG_FULL_EXE), "-hide_banner", "-filters"],
            "FFMPEG_FULL_FILTERS",
        )
        help_output = _run_checked(
            [str(FFMPEG_FULL_EXE), "-hide_banner", "-h", "filter=libplacebo"],
            "FFMPEG_FULL_LIBPLACEBO",
        )
        required_tokens = {
            "version": (version_output.lower(), f"ffmpeg version {FFMPEG_VERSION}"),
            "libplacebo filter": (filter_output.lower(), "libplacebo"),
            "custom shader": (help_output.lower(), "custom_shader_path"),
        }
        missing = [
            label
            for label, (haystack, needle) in required_tokens.items()
            if needle not in haystack
        ]
        if missing:
            return "Anime4K 필수 기능 누락: " + ", ".join(missing)
    except Exception as exc:
        print(
            "[VIDEO_TOOLS:FFMPEG_FULL][ERROR] 기능 검증 중 예외: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return f"기능 검증 실패: {type(exc).__name__}: {exc}"
    return None


def ensure_ffmpeg_full() -> Path:
    global _FFMPEG_FULL_VERIFIED
    with _INSTALL_LOCK:
        if os.name != "nt":
            message = "Anime4K용 FFmpeg 자동 설치는 Windows에서만 지원합니다"
            print(f"[VIDEO_TOOLS:FFMPEG_FULL][ERROR] {message}: os.name={os.name!r}")
            raise RuntimeError(message)
        if _FFMPEG_FULL_VERIFIED and FFMPEG_FULL_EXE.is_file():
            return FFMPEG_FULL_EXE
        validation_error = _ffmpeg_full_validation_error()
        if validation_error is None:
            _FFMPEG_FULL_VERIFIED = True
            print(
                "[VIDEO_TOOLS:FFMPEG_FULL] Anime4K 런타임 검증 완료: "
                f"version={FFMPEG_VERSION}, exe={FFMPEG_FULL_EXE}"
            )
            return FFMPEG_FULL_EXE
        print(f"[VIDEO_TOOLS:FFMPEG_FULL] 설치 필요: {validation_error}")

        archive_path = _download_verified(
            url=FFMPEG_FULL_URL,
            expected_sha256=FFMPEG_FULL_SHA256,
            archive_path=TOOLS_ROOT / FFMPEG_FULL_ARCHIVE_NAME,
            label="FFMPEG_FULL",
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="ffmpeg_full_install_", dir=str(TOOLS_ROOT)
            ) as temp_name:
                extract_dir = Path(temp_name) / "extracted"
                _extract_7z_safely(archive_path, extract_dir, "FFMPEG_FULL")
                candidates = sorted(extract_dir.glob("*/bin/ffmpeg.exe"))
                if len(candidates) != 1:
                    print(
                        "[VIDEO_TOOLS:FFMPEG_FULL][ERROR] 압축 내 실행 파일 탐색 실패: "
                        f"candidates={[str(path) for path in candidates]!r}"
                    )
                    raise RuntimeError("Anime4K용 FFmpeg 패키지 구성이 올바르지 않습니다")
                _replace_directory(
                    candidates[0].parent.parent,
                    FFMPEG_FULL_DIR,
                    "FFMPEG_FULL",
                )
        except Exception as exc:
            print(
                "[VIDEO_TOOLS:FFMPEG_FULL][ERROR] 설치 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        validation_error = _ffmpeg_full_validation_error()
        if validation_error is not None:
            print(
                "[VIDEO_TOOLS:FFMPEG_FULL][ERROR] 최종 설치 검증 실패: "
                f"{validation_error}"
            )
            raise RuntimeError("Anime4K용 FFmpeg 최종 설치 검증에 실패했습니다")
        _FFMPEG_FULL_VERIFIED = True
        print(
            "[VIDEO_TOOLS:FFMPEG_FULL] 설치 완료: "
            f"version={FFMPEG_VERSION}, exe={FFMPEG_FULL_EXE}"
        )
        return FFMPEG_FULL_EXE


def ensure_anime4k_shader() -> Path:
    global _ANIME4K_VERIFIED
    with _INSTALL_LOCK:
        if _ANIME4K_VERIFIED and ANIME4K_SHADER.is_file():
            return ANIME4K_SHADER
        if ANIME4K_SHADER.is_file() and _sha256(ANIME4K_SHADER) == ANIME4K_SHADER_SHA256:
            _ANIME4K_VERIFIED = True
            print(
                "[VIDEO_TOOLS:ANIME4K] Fast/M 셰이더 검증 완료: "
                f"version={ANIME4K_VERSION}, path={ANIME4K_SHADER}"
            )
            return ANIME4K_SHADER
        if ANIME4K_SHADER.exists():
            print(
                "[VIDEO_TOOLS:ANIME4K] 조합 셰이더 무결성 불일치, 다시 생성: "
                f"path={ANIME4K_SHADER}"
            )
            _remove_exact_path(ANIME4K_SHADER, TOOLS_ROOT)

        base_url = f"https://raw.githubusercontent.com/bloc97/Anime4K/v{ANIME4K_VERSION}/"
        source_dir = ANIME4K_DIR / "sources"
        source_paths: list[Path] = []
        for relative_path, expected_hash in ANIME4K_RESOURCES:
            source_paths.append(
                _download_verified(
                    url=base_url + relative_path,
                    expected_sha256=expected_hash,
                    archive_path=source_dir / Path(relative_path).name,
                    label="ANIME4K",
                )
            )
        license_relative, license_hash = ANIME4K_LICENSE
        _download_verified(
            url=base_url + license_relative,
            expected_sha256=license_hash,
            archive_path=ANIME4K_DIR / "LICENSE",
            label="ANIME4K_LICENSE",
        )

        shader_bytes = (
            b"\n".join(path.read_bytes().rstrip(b"\r\n") for path in source_paths)
            + b"\n"
        )
        actual_hash = hashlib.sha256(shader_bytes).hexdigest()
        if actual_hash != ANIME4K_SHADER_SHA256:
            print(
                "[VIDEO_TOOLS:ANIME4K][ERROR] Fast/M 조합 셰이더 해시 불일치: "
                f"expected={ANIME4K_SHADER_SHA256}, actual={actual_hash}"
            )
            raise RuntimeError("Anime4K Fast/M 조합 셰이더 검증에 실패했습니다")
        part_path = ANIME4K_SHADER.with_suffix(".glsl.part")
        _remove_exact_path(part_path, TOOLS_ROOT)
        try:
            ANIME4K_DIR.mkdir(parents=True, exist_ok=True)
            with part_path.open("xb") as handle:
                handle.write(shader_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(part_path, ANIME4K_SHADER)
        except Exception as exc:
            print(
                "[VIDEO_TOOLS:ANIME4K][ERROR] Fast/M 셰이더 저장 실패: "
                f"path={ANIME4K_SHADER}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            try:
                _remove_exact_path(part_path, TOOLS_ROOT)
            except Exception as cleanup_exc:
                print(
                    "[VIDEO_TOOLS:ANIME4K][ERROR] 부분 셰이더 정리 실패: "
                    f"path={part_path}, error={cleanup_exc}"
                )
                traceback.print_exc()
            raise
        _ANIME4K_VERIFIED = True
        print(
            "[VIDEO_TOOLS:ANIME4K] Fast/M 셰이더 설치 완료: "
            f"version={ANIME4K_VERSION}, bytes={len(shader_bytes):,}, path={ANIME4K_SHADER}"
        )
        return ANIME4K_SHADER


def ensure_anime4k() -> dict[str, str]:
    try:
        ffmpeg = ensure_ffmpeg_full()
        shader = ensure_anime4k_shader()
        return {"ffmpeg": str(ffmpeg), "shader": str(shader)}
    except Exception as exc:
        print(
            "[VIDEO_TOOLS:ANIME4K][ERROR] Anime4K 런타임 준비 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


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
