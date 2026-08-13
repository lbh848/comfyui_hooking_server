"""독립 영상 후처리 런타임: Real-ESRGAN 프레임 업스케일과 animated AVIF 저장."""

from __future__ import annotations

import asyncio
import bisect
import json
import os
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Awaitable, Callable

from PIL import Image

from ensure_video_tools import (
    REALESRGAN_DIR as _TOOL_DIR,
    ensure_anime4k as _ensure_anime4k_sync,
    ensure_ffmpeg as _ensure_ffmpeg_sync,
    ensure_realesrgan as _ensure_realesrgan_sync,
)

try:
    import pillow_avif  # noqa: F401 - animated AVIF 검증 지원 등록
except Exception:
    print("[VIDEO:POSTPROCESS] pillow-avif-plugin 로드 실패: AVIF 검증이 제한됩니다")
    traceback.print_exc()


ProgressCallback = Callable[[dict], Awaitable[None]]

VIDEO_UPSCALE_MODELS = frozenset(
    {"realesr-animevideov3", "anime4k-fast-m", "lanczos"}
)
VIDEO_OUTPUT_FORMATS = frozenset({"avif", "webp"})
VIDEO_REPROCESS_MIN_FPS = 1
VIDEO_REPROCESS_MAX_FPS = 60
VIDEO_REPROCESS_MIN_TARGET_MB = 0.1
VIDEO_REPROCESS_MAX_TARGET_MB = 2048.0

DEFAULT_VIDEO_POSTPROCESS_CONFIG = {
    "enabled": True,
    "scale": 2,
    "model": "realesr-animevideov3",
    "gpu_id": "auto",
    "tile_size": 0,
    "worker_count": 1,
}


def normalize_video_reprocess_fps(value: object) -> int:
    """Validate the output FPS used when reprocessing an existing animation."""

    try:
        if isinstance(value, bool):
            raise TypeError("bool은 FPS로 허용되지 않음")
        fps = int(value)
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("정수가 아닌 FPS는 허용되지 않음")
        if isinstance(value, str) and value.strip() != str(fps):
            raise ValueError("정수 문자열 형식이 아님")
        if not VIDEO_REPROCESS_MIN_FPS <= fps <= VIDEO_REPROCESS_MAX_FPS:
            raise ValueError("허용 범위를 벗어남")
        return fps
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO:REPROCESS] FPS 검증 실패: "
            f"value={value!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise ValueError(
            f"영상 FPS는 {VIDEO_REPROCESS_MIN_FPS}부터 "
            f"{VIDEO_REPROCESS_MAX_FPS}까지의 정수여야 합니다"
        ) from exc


def normalize_video_reprocess_target_bytes(value: object) -> int:
    """Validate an MB target and return its byte ceiling."""

    try:
        if isinstance(value, bool):
            raise TypeError("bool은 목표 용량으로 허용되지 않음")
        target_mb = float(value)
        if not VIDEO_REPROCESS_MIN_TARGET_MB <= target_mb <= VIDEO_REPROCESS_MAX_TARGET_MB:
            raise ValueError("허용 범위를 벗어남")
        return max(1, int(target_mb * 1024 * 1024))
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO:REPROCESS] 목표 용량 검증 실패: "
            f"value={value!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise ValueError(
            f"목표 용량은 {VIDEO_REPROCESS_MIN_TARGET_MB:g}MB부터 "
            f"{VIDEO_REPROCESS_MAX_TARGET_MB:g}MB까지 설정해야 합니다"
        ) from exc


class _WindowsVideoSubprocessLoop:
    """Windows 영상 도구 실행을 전담하는 Proactor 이벤트 루프."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            # Windows의 SelectorEventLoop는 asyncio subprocess를 지원하지 않는다.
            # 영상 도구만 별도 Proactor 루프에서 실행해 서버 루프와 격리한다.
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            print(
                "[VIDEO:SUBPROCESS] Windows Proactor 루프 시작 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            self._ready.set()
        finally:
            if loop is not None and not loop.is_closed():
                loop.close()

    def submit(self, coroutine):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._startup_error = None
                self._loop = None
                self._thread = threading.Thread(
                    target=self._run,
                    name="video-async-subprocess-loop",
                    daemon=True,
                )
                self._thread.start()
            if not self._ready.wait(timeout=10):
                coroutine.close()
                print("[VIDEO:SUBPROCESS] Windows Proactor 루프 시작 시간 초과")
                raise RuntimeError("영상 도구 실행 루프를 시작하지 못했습니다")
            if self._startup_error is not None or self._loop is None:
                coroutine.close()
                error = self._startup_error
                print(
                    "[VIDEO:SUBPROCESS] Windows Proactor 루프를 사용할 수 없음: "
                    f"error={type(error).__name__ if error else 'unknown'}: {error}"
                )
                if error is not None:
                    traceback.print_exception(type(error), error, error.__traceback__)
                raise RuntimeError("영상 도구 실행 루프가 준비되지 않았습니다") from error
            try:
                return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            except Exception as exc:
                coroutine.close()
                print(
                    "[VIDEO:SUBPROCESS] Windows Proactor 루프에 명령 제출 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise


_WINDOWS_VIDEO_SUBPROCESS_LOOP = (
    _WindowsVideoSubprocessLoop() if os.name == "nt" else None
)


def normalize_video_postprocess_config(raw: object) -> dict:
    """사용자 설정을 검증해 실행 가능한 영상 후처리 설정을 반환한다."""

    if raw is None:
        source = {}
    elif isinstance(raw, dict):
        source = raw
    else:
        message = (
            "video_postprocess 설정은 객체여야 합니다: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        print(f"[VIDEO:POSTPROCESS:CONFIG] {message}")
        raise ValueError(message)

    normalized = dict(DEFAULT_VIDEO_POSTPROCESS_CONFIG)

    enabled = source.get("enabled", normalized["enabled"])
    if not isinstance(enabled, bool):
        message = f"video_postprocess.enabled는 bool이어야 합니다: value={enabled!r}"
        print(f"[VIDEO:POSTPROCESS:CONFIG] {message}")
        raise ValueError(message)
    normalized["enabled"] = enabled

    raw_scale = source.get("scale", normalized["scale"])
    try:
        if isinstance(raw_scale, bool):
            raise TypeError("bool은 허용되지 않음")
        scale = int(raw_scale)
        if isinstance(raw_scale, float) and not raw_scale.is_integer():
            raise ValueError("정수가 아닌 실수는 허용되지 않음")
        if isinstance(raw_scale, str) and raw_scale.strip() != str(scale):
            raise ValueError("정수 문자열 형식이 아님")
        if scale not in (2, 3, 4):
            raise ValueError("허용 배율 2, 3, 4에 없음")
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO:POSTPROCESS:CONFIG] 업스케일 배율 검증 실패: "
            f"value={raw_scale!r}, error={exc}"
        )
        traceback.print_exc()
        raise ValueError("영상 업스케일 배율은 2, 3, 4 중 하나여야 합니다") from exc
    normalized["scale"] = scale

    model = str(source.get("model", normalized["model"]) or "").strip()
    if model not in VIDEO_UPSCALE_MODELS:
        message = f"지원하지 않는 영상 업스케일 모델입니다: {model!r}"
        print(f"[VIDEO:POSTPROCESS:CONFIG] {message}")
        raise ValueError(message)
    normalized["model"] = model

    raw_gpu_id = source.get("gpu_id", normalized["gpu_id"])
    if isinstance(raw_gpu_id, str) and raw_gpu_id.strip().lower() == "auto":
        normalized["gpu_id"] = "auto"
    else:
        try:
            if isinstance(raw_gpu_id, bool):
                raise TypeError("bool은 허용되지 않음")
            gpu_id = int(raw_gpu_id)
            if gpu_id < 0:
                raise ValueError("GPU id는 0 이상이어야 함")
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VIDEO:POSTPROCESS:CONFIG] GPU id 검증 실패: "
                f"value={raw_gpu_id!r}, error={exc}"
            )
            traceback.print_exc()
            raise ValueError("영상 후처리 GPU id는 auto 또는 0 이상의 정수여야 합니다") from exc
        normalized["gpu_id"] = gpu_id

    raw_tile = source.get("tile_size", normalized["tile_size"])
    try:
        if isinstance(raw_tile, bool):
            raise TypeError("bool은 허용되지 않음")
        tile_size = int(raw_tile)
        if tile_size != 0 and tile_size < 32:
            raise ValueError("타일 크기는 0 또는 32 이상이어야 함")
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO:POSTPROCESS:CONFIG] 타일 크기 검증 실패: "
            f"value={raw_tile!r}, error={exc}"
        )
        traceback.print_exc()
        raise ValueError("영상 후처리 타일 크기는 0 또는 32 이상의 정수여야 합니다") from exc
    normalized["tile_size"] = tile_size

    raw_workers = source.get("worker_count", normalized["worker_count"])
    if raw_workers != 1:
        print(
            "[VIDEO:POSTPROCESS:CONFIG] worker_count는 GPU 안정성을 위해 1로 고정합니다: "
            f"value={raw_workers!r}"
        )
    normalized["worker_count"] = 1
    return normalized


async def _notify(callback: ProgressCallback | None, **detail) -> None:
    if callback is not None:
        await callback(detail)


async def ensure_realesrgan() -> Path:
    return await asyncio.to_thread(_ensure_realesrgan_sync)


async def ensure_ffmpeg() -> Path:
    return await asyncio.to_thread(_ensure_ffmpeg_sync)


async def ensure_anime4k() -> dict[str, Path]:
    prepared = await asyncio.to_thread(_ensure_anime4k_sync)
    return {key: Path(value) for key, value in prepared.items()}


async def _run_command_on_subprocess_loop(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:
        print(
            f"[VIDEO:{label}] 프로세스 시작 실패: command={command!r}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise

    chunks: list[str] = []
    try:
        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                chunks.append(line)
                print(f"[VIDEO:{label}] {line}")
        return_code = await process.wait()
    except asyncio.CancelledError:
        print(f"[VIDEO:{label}] 작업 취소로 프로세스 종료: pid={process.pid}")
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except Exception as exc:
            print(
                f"[VIDEO:{label}] 정상 종료 실패, 강제 종료 시도: "
                f"pid={process.pid}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            process.kill()
            await process.wait()
        raise

    output = "\n".join(chunks)
    if return_code != 0:
        print(
            f"[VIDEO:{label}] 프로세스 실패: returncode={return_code}, "
            f"command={command!r}, output={output[-4000:]}"
        )
        raise RuntimeError(f"{label} 프로세스가 종료 코드 {return_code}로 실패했습니다")
    return output


async def _run_command(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
) -> str:
    if os.name != "nt":
        return await _run_command_on_subprocess_loop(
            command,
            label=label,
            cwd=cwd,
        )

    runner = _WINDOWS_VIDEO_SUBPROCESS_LOOP
    if runner is None:
        print(
            "[VIDEO:SUBPROCESS] Windows 전용 실행 루프가 초기화되지 않음: "
            f"command={command!r}"
        )
        raise RuntimeError("Windows 영상 도구 실행 루프가 초기화되지 않았습니다")

    concurrent_future = runner.submit(
        _run_command_on_subprocess_loop(command, label=label, cwd=cwd)
    )
    try:
        return await asyncio.wrap_future(concurrent_future)
    except asyncio.CancelledError:
        concurrent_future.cancel()
        raise


def _list_pngs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


async def _run_realesrgan(
    input_dir: Path,
    output_dir: Path,
    *,
    expected_frames: int,
    settings: dict,
    progress_callback: ProgressCallback | None,
) -> None:
    executable = await ensure_realesrgan()
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(executable),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-m",
        str(_TOOL_DIR / "models"),
        "-n",
        settings["model"],
        "-s",
        str(settings["scale"]),
        "-t",
        str(settings["tile_size"]),
        "-f",
        "png",
        "-v",
    ]
    if settings["gpu_id"] != "auto":
        command.extend(["-g", str(settings["gpu_id"])])

    task = asyncio.create_task(
        _run_command(command, label="REALESRGAN", cwd=_TOOL_DIR)
    )
    last_count = -1
    try:
        while not task.done():
            count = len(await asyncio.to_thread(_list_pngs, output_dir))
            if count != last_count:
                percentage = 25 + (min(count, expected_frames) / expected_frames * 50)
                await _notify(
                    progress_callback,
                    phase="video_upscaling",
                    percentage=percentage,
                    current=count,
                    total=expected_frames,
                )
                last_count = count
            await asyncio.sleep(0.35)
        await task
    except Exception:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise

    output_frames = await asyncio.to_thread(_list_pngs, output_dir)
    if len(output_frames) != expected_frames:
        print(
            "[VIDEO:REALESRGAN] 출력 프레임 수 불일치: "
            f"expected={expected_frames}, actual={len(output_frames)}, "
            f"output_dir={str(output_dir)!r}"
        )
        raise RuntimeError("Real-ESRGAN 출력 프레임 수가 원본과 일치하지 않습니다")


async def _run_upscale_command_with_progress(
    command: list[str],
    output_dir: Path,
    *,
    label: str,
    phase: str,
    expected_frames: int,
    progress_callback: ProgressCallback | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    task = asyncio.create_task(_run_command(command, label=label))
    last_count = -1
    try:
        while not task.done():
            count = len(await asyncio.to_thread(_list_pngs, output_dir))
            if count != last_count:
                await _notify(
                    progress_callback,
                    phase=phase,
                    percentage=25 + (min(count, expected_frames) / expected_frames * 50),
                    current=count,
                    total=expected_frames,
                )
                last_count = count
            await asyncio.sleep(0.35)
        await task
    except Exception:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise

    output_frames = await asyncio.to_thread(_list_pngs, output_dir)
    if len(output_frames) != expected_frames:
        print(
            f"[VIDEO:{label}] 출력 프레임 수 불일치: "
            f"expected={expected_frames}, actual={len(output_frames)}, "
            f"output_dir={str(output_dir)!r}"
        )
        raise RuntimeError(f"{label} 출력 프레임 수가 원본과 일치하지 않습니다")


async def _run_lanczos(
    input_dir: Path,
    output_dir: Path,
    *,
    expected_frames: int,
    scale: int,
    progress_callback: ProgressCallback | None,
) -> None:
    ffmpeg = str(await ensure_ffmpeg())
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "24",
        "-start_number",
        "0",
        "-i",
        str(input_dir / "frame_%08d.png"),
        "-vf",
        f"scale=iw*{scale}:ih*{scale}:flags=lanczos",
        "-frames:v",
        str(expected_frames),
        "-start_number",
        "0",
        str(output_dir / "frame_%08d.png"),
    ]
    await _run_upscale_command_with_progress(
        command,
        output_dir,
        label="LANCZOS",
        phase="video_upscaling_lanczos",
        expected_frames=expected_frames,
        progress_callback=progress_callback,
    )


def _ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:")


async def _run_anime4k(
    input_dir: Path,
    output_dir: Path,
    *,
    expected_frames: int,
    scale: int,
    progress_callback: ProgressCallback | None,
) -> None:
    tools = await ensure_anime4k()
    ffmpeg = str(tools["ffmpeg"])
    shader = _ffmpeg_filter_path(tools["shader"])
    video_filter = (
        "format=rgba,hwupload,"
        f"libplacebo=w=iw*{scale}:h=ih*{scale}:custom_shader_path='{shader}',"
        "hwdownload,format=rgba"
    )
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-init_hw_device",
        "vulkan=anime4k",
        "-filter_hw_device",
        "anime4k",
        "-framerate",
        "24",
        "-start_number",
        "0",
        "-i",
        str(input_dir / "frame_%08d.png"),
        "-vf",
        video_filter,
        "-frames:v",
        str(expected_frames),
        "-start_number",
        "0",
        str(output_dir / "frame_%08d.png"),
    ]
    await _run_upscale_command_with_progress(
        command,
        output_dir,
        label="ANIME4K",
        phase="video_upscaling_anime4k",
        expected_frames=expected_frames,
        progress_callback=progress_callback,
    )


def _avif_crf(quality: int) -> int:
    return max(0, min(63, round((100 - quality) * 0.8)))


def inspect_animation(
    path: str | os.PathLike[str],
    *,
    fallback_duration: float = 0.0,
) -> dict:
    """Return validated animation timing without retaining decoded frames."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        print(f"[VIDEO:REPROCESS] 원본 애니메이션 없음: path={str(source_path)!r}")
        raise FileNotFoundError("후처리할 영상 파일을 찾지 못했습니다")
    try:
        fallback_seconds = float(fallback_duration or 0.0)
        if fallback_seconds < 0:
            raise ValueError("fallback_duration은 0 이상이어야 함")
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO:REPROCESS] 원본 재생 시간 형식 오류: "
            f"value={fallback_duration!r}, path={str(source_path)!r}, error={exc}"
        )
        traceback.print_exc()
        raise ValueError("원본 영상 재생 시간이 올바르지 않습니다") from exc

    try:
        with Image.open(source_path) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            if not bool(getattr(image, "is_animated", False)) or frame_count < 2:
                print(
                    "[VIDEO:REPROCESS] animated 파일 아님: "
                    f"path={str(source_path)!r}, frames={frame_count}"
                )
                raise ValueError("영상 후처리는 animated AVIF/WebP 파일에만 사용할 수 있습니다")
            width, height = image.size
            raw_durations: list[float | None] = []
            for index in range(frame_count):
                image.seek(index)
                # Pillow WebP populates per-frame timestamp/duration only after load().
                image.load()
                raw_value = image.info.get("duration")
                try:
                    duration_ms = float(raw_value)
                    if duration_ms <= 0:
                        raise ValueError("프레임 duration은 0보다 커야 함")
                    raw_durations.append(duration_ms)
                except (TypeError, ValueError, OverflowError):
                    raw_durations.append(None)
    except ValueError:
        raise
    except Exception as exc:
        print(
            "[VIDEO:REPROCESS] 원본 애니메이션 검사 실패: "
            f"path={str(source_path)!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise RuntimeError("후처리할 영상 파일을 읽지 못했습니다") from exc

    known = [value for value in raw_durations if value is not None]
    fallback_frame_ms = (
        fallback_seconds * 1000.0 / frame_count
        if fallback_seconds > 0
        else (sum(known) / len(known) if known else 1000.0 / 24.0)
    )
    durations_ms = [
        float(value if value is not None else fallback_frame_ms)
        for value in raw_durations
    ]
    measured_seconds = sum(durations_ms) / 1000.0
    # 일부 디코더는 duration 메타데이터를 누락하거나 비정상적으로 짧게 보고한다.
    # 생성 당시 메타데이터가 있으면 그 길이를 진실 소스로 삼아 프레임 간격만 균등 보정한다.
    if fallback_seconds > 0 and (
        measured_seconds < 0.1
        or abs(measured_seconds - fallback_seconds) > max(0.25, fallback_seconds * 0.1)
    ):
        durations_ms = [fallback_seconds * 1000.0 / frame_count] * frame_count
        measured_seconds = fallback_seconds
        print(
            "[VIDEO:REPROCESS] 원본 duration 메타데이터 보정: "
            f"path={str(source_path)!r}, fallback={fallback_seconds:.3f}s, "
            f"frames={frame_count}"
        )

    result = {
        "path": str(source_path),
        "width": int(width),
        "height": int(height),
        "frame_count": frame_count,
        "duration": measured_seconds,
        "source_fps": frame_count / measured_seconds,
        "durations_ms": durations_ms,
        "size_bytes": source_path.stat().st_size,
    }
    print(
        "[VIDEO:REPROCESS] 원본 애니메이션 확인: "
        f"path={str(source_path)!r}, size={width}x{height}, frames={frame_count}, "
        f"duration={measured_seconds:.3f}s, bytes={result['size_bytes']:,}"
    )
    return result


def _extract_animation_frames(
    source_path: Path,
    output_dir: Path,
    *,
    fps: int,
    fallback_duration: float,
) -> dict:
    """Resample an animated AVIF/WebP onto a constant-FPS PNG sequence."""

    timing = inspect_animation(source_path, fallback_duration=fallback_duration)
    duration = float(timing["duration"])
    target_count = max(2, round(duration * fps))
    cumulative_ms: list[float] = []
    elapsed_ms = 0.0
    for frame_duration in timing["durations_ms"]:
        elapsed_ms += float(frame_duration)
        cumulative_ms.append(elapsed_ms)
    total_ms = cumulative_ms[-1]
    source_indices = [
        min(
            int(timing["frame_count"]) - 1,
            bisect.bisect_right(cumulative_ms, (index + 0.5) * total_ms / target_count),
        )
        for index in range(target_count)
    ]

    try:
        with Image.open(source_path) as image:
            loaded_index = -1
            loaded_frame: Image.Image | None = None
            for output_index, source_index in enumerate(source_indices):
                if source_index != loaded_index:
                    image.seek(source_index)
                    loaded_frame = image.convert("RGBA")
                    loaded_index = source_index
                if loaded_frame is None:
                    print(
                        "[VIDEO:REPROCESS] 프레임 추출 내부 상태 오류: "
                        f"source={str(source_path)!r}, output_index={output_index}"
                    )
                    raise RuntimeError("영상 프레임을 추출하지 못했습니다")
                loaded_frame.save(
                    output_dir / f"frame_{output_index:08d}.png",
                    format="PNG",
                )
    except Exception as exc:
        print(
            "[VIDEO:REPROCESS] animated 프레임 추출 실패: "
            f"path={str(source_path)!r}, fps={fps}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    print(
        "[VIDEO:REPROCESS] FPS 재샘플링 완료: "
        f"source_frames={timing['frame_count']}, output_frames={target_count}, "
        f"fps={fps}, duration={duration:.3f}s"
    )
    return {**timing, "output_frame_count": target_count}


def _verify_animation(path: Path, expected_frames: int) -> None:
    try:
        with Image.open(path) as image:
            animated = bool(getattr(image, "is_animated", False))
            frames = int(getattr(image, "n_frames", 1))
    except Exception as exc:
        print(
            "[VIDEO:ENCODE] 출력 애니메이션 열기 실패: "
            f"path={str(path)!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    if not animated or frames != expected_frames:
        print(
            "[VIDEO:ENCODE] 출력 애니메이션 검증 실패: "
            f"path={str(path)!r}, animated={animated}, "
            f"expected_frames={expected_frames}, actual_frames={frames}"
        )
        raise RuntimeError("영상 후처리 결과의 프레임 검증에 실패했습니다")


def _encode_command(
    ffmpeg: str,
    frame_pattern: Path,
    output_path: Path,
    *,
    fps: int,
    frame_count: int,
    quality: int,
    output_format: str,
    overlay_path: Path | None = None,
    canvas_height: int | None = None,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
    ]
    if overlay_path is not None:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-i",
                str(overlay_path),
                "-filter_complex",
                (
                    f"[0:v]pad=iw:{int(canvas_height or 1)}:0:0:color=black[base];"
                    "[base][1:v]overlay=0:0:shortest=1,format=yuv420p[v]"
                ),
                "-map",
                "[v]",
            ]
        )
    else:
        command.extend(["-vf", "format=yuv420p"])

    command.extend(["-frames:v", str(frame_count), "-an"])
    if output_format == "AVIF":
        command.extend(
            [
                "-c:v",
                "libaom-av1",
                "-crf",
                str(_avif_crf(quality)),
                "-b:v",
                "0",
                # 속도 우선 튜닝: cpu-used 6→8(≈2×), 2타일 열(tile-columns 1)로 프레임 내 병렬화,
                # threads 0(전체 코어) + row-mt 1. 애니메이션 업스케일 프레임에 품질 저하 미미.
                "-cpu-used",
                "8",
                "-row-mt",
                "1",
                "-tile-columns",
                "1",
                "-threads",
                "0",
                "-loop",
                "0",
                "-f",
                "avif",
            ]
        )
    else:
        command.extend(
            [
                "-c:v",
                "libwebp_anim",
                "-q:v",
                str(quality),
                "-loop",
                "0",
                "-f",
                "webp",
            ]
        )
    command.append(str(output_path))
    return command


async def _encode_pair(
    frames_dir: Path,
    job_dir: Path,
    *,
    fps: int,
    frame_count: int,
    quality: int,
    overlay_path: Path | None,
    canvas_height: int,
    progress_callback: ProgressCallback | None,
    output_format: str = "avif",
) -> tuple[Path, Path, str]:
    ffmpeg = str(await ensure_ffmpeg())
    frame_pattern = frames_dir / "frame_%08d.png"
    errors: list[str] = []
    requested_format = str(output_format or "avif").strip().lower()
    if requested_format not in VIDEO_OUTPUT_FORMATS:
        print(f"[VIDEO:ENCODE] 지원하지 않는 출력 형식: value={output_format!r}")
        raise ValueError("영상 출력 형식은 AVIF 또는 WebP여야 합니다")
    attempts = (
        (("AVIF", ".avif"), ("WEBP", ".webp"))
        if requested_format == "avif"
        else (("WEBP", ".webp"),)
    )
    for encoder_format, extension in attempts:
        raw_path = job_dir / f"result_raw{extension}"
        main_path = job_dir / f"result_main{extension}"
        for candidate in (raw_path, main_path):
            if candidate.exists():
                candidate.unlink()
        try:
            await _notify(
                progress_callback,
                phase="video_encoding_raw",
                percentage=80,
                format=encoder_format.lower(),
            )
            await _run_command(
                _encode_command(
                    ffmpeg,
                    frame_pattern,
                    raw_path,
                    fps=fps,
                    frame_count=frame_count,
                    quality=quality,
                    output_format=encoder_format,
                ),
                label="ENCODE_RAW",
            )
            await asyncio.to_thread(_verify_animation, raw_path, frame_count)

            await _notify(
                progress_callback,
                phase="video_encoding_composite",
                percentage=90,
                format=encoder_format.lower(),
            )
            if overlay_path is None:
                await asyncio.to_thread(shutil.copyfile, raw_path, main_path)
            else:
                await _run_command(
                    _encode_command(
                        ffmpeg,
                        frame_pattern,
                        main_path,
                        fps=fps,
                        frame_count=frame_count,
                        quality=quality,
                        output_format=encoder_format,
                        overlay_path=overlay_path,
                        canvas_height=canvas_height,
                    ),
                    label="ENCODE_COMPOSITE",
                )
            await asyncio.to_thread(_verify_animation, main_path, frame_count)
            print(
                f"[VIDEO:ENCODE] {encoder_format} 2종 저장 검증 완료: "
                f"frames={frame_count}, raw_bytes={raw_path.stat().st_size:,}, "
                f"main_bytes={main_path.stat().st_size:,}"
            )
            return main_path, raw_path, extension
        except Exception as exc:
            errors.append(f"{encoder_format}: {type(exc).__name__}: {exc}")
            print(
                f"[VIDEO:ENCODE] {encoder_format} 저장 실패, 다음 형식 시도: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            for candidate in (raw_path, main_path):
                try:
                    if candidate.is_file():
                        candidate.unlink()
                except OSError as cleanup_exc:
                    print(
                        "[VIDEO:ENCODE] 실패 출력 정리 실패: "
                        f"path={str(candidate)!r}, error={cleanup_exc}"
                    )
    raise RuntimeError("애니메이션 저장 실패: " + " / ".join(errors))


async def _encode_pair_to_target_size(
    frames_dir: Path,
    job_dir: Path,
    *,
    fps: int,
    frame_count: int,
    target_bytes: int,
    progress_callback: ProgressCallback | None,
    output_format: str,
) -> tuple[Path, Path, str, int, int]:
    """Find the highest encoder quality whose main output fits the byte ceiling."""

    if target_bytes <= 0:
        print(f"[VIDEO:REPROCESS] 목표 바이트 오류: value={target_bytes!r}")
        raise ValueError("영상 목표 용량이 올바르지 않습니다")

    attempts = 0

    async def encode(quality: int) -> tuple[Path, Path, str, int]:
        nonlocal attempts
        attempts += 1
        await _notify(
            progress_callback,
            phase="video_target_size_search",
            percentage=min(96, 78 + attempts * 2),
            quality=quality,
            attempt=attempts,
            target_bytes=target_bytes,
        )
        main_path, raw_path, extension = await _encode_pair(
            frames_dir,
            job_dir,
            fps=fps,
            frame_count=frame_count,
            quality=quality,
            overlay_path=None,
            canvas_height=1,
            progress_callback=None,
            output_format=output_format,
        )
        size_bytes = main_path.stat().st_size
        print(
            "[VIDEO:REPROCESS] 목표 용량 탐색: "
            f"attempt={attempts}, quality={quality}, bytes={size_bytes:,}, "
            f"target={target_bytes:,}, format={extension}"
        )
        return main_path, raw_path, extension, size_bytes

    high_main, high_raw, high_extension, high_size = await encode(100)
    if high_size <= target_bytes:
        return high_main, high_raw, high_extension, 100, high_size

    low_main, low_raw, low_extension, low_size = await encode(1)
    if low_size > target_bytes:
        minimum_mb = low_size / (1024 * 1024)
        target_mb = target_bytes / (1024 * 1024)
        print(
            "[VIDEO:REPROCESS] 목표 용량 달성 불가: "
            f"minimum={minimum_mb:.3f}MB, target={target_mb:.3f}MB, "
            f"fps={fps}, frames={frame_count}, format={low_extension}"
        )
        raise ValueError(
            f"선택한 FPS·해상도에서는 최소 약 {minimum_mb:.2f}MB가 필요합니다. "
            "목표 용량을 늘리거나 업스케일을 끄세요."
        )

    best_quality = 1
    best_size = low_size
    best_extension = low_extension
    best_main = job_dir / f"target_best_main{best_extension}"
    best_raw = job_dir / f"target_best_raw{best_extension}"
    await asyncio.to_thread(shutil.copyfile, low_main, best_main)
    await asyncio.to_thread(shutil.copyfile, low_raw, best_raw)

    search_low = 2
    search_high = 99
    while search_low <= search_high:
        quality = (search_low + search_high) // 2
        main_path, raw_path, extension, size_bytes = await encode(quality)
        if size_bytes <= target_bytes:
            for candidate in (best_main, best_raw):
                if candidate.is_file():
                    candidate.unlink()
            best_quality = quality
            best_size = size_bytes
            best_extension = extension
            best_main = job_dir / f"target_best_main{best_extension}"
            best_raw = job_dir / f"target_best_raw{best_extension}"
            await asyncio.to_thread(shutil.copyfile, main_path, best_main)
            await asyncio.to_thread(shutil.copyfile, raw_path, best_raw)
            search_low = quality + 1
        else:
            search_high = quality - 1

    final_main = job_dir / f"result_main{best_extension}"
    final_raw = job_dir / f"result_raw{best_extension}"
    for extension in (".avif", ".webp"):
        for prefix in ("result_main", "result_raw"):
            candidate = job_dir / f"{prefix}{extension}"
            if candidate.is_file():
                candidate.unlink()
    os.replace(best_main, final_main)
    os.replace(best_raw, final_raw)
    print(
        "[VIDEO:REPROCESS] 목표 용량 탐색 완료: "
        f"quality={best_quality}, bytes={best_size:,}, target={target_bytes:,}, "
        f"attempts={attempts}, format={best_extension}"
    )
    return final_main, final_raw, best_extension, best_quality, best_size


async def process_staged_video(
    job_dir: str | os.PathLike[str],
    *,
    settings: dict,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """스풀의 MP4를 프레임 업스케일하고 검증된 animated 파일 2종으로 만든다."""

    directory = Path(job_dir).resolve()
    manifest_path = directory / "job.json"
    if not manifest_path.is_file():
        print(
            "[VIDEO:POSTPROCESS] 스풀 파일 누락: "
            f"job_dir={str(directory)!r}, manifest={manifest_path.is_file()}"
        )
        raise FileNotFoundError("영상 후처리 스풀 파일이 완전하지 않습니다")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[VIDEO:POSTPROCESS] 스풀 manifest 로드 실패: "
            f"path={str(manifest_path)!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise

    job_kind = str(manifest.get("job_kind") or "h3_render").strip().lower()
    is_existing_animation = job_kind == "existing_animation"
    if is_existing_animation:
        input_filename = str(manifest.get("input_filename") or "").strip()
        if (
            not input_filename
            or os.path.basename(input_filename) != input_filename
            or Path(input_filename).suffix.lower() not in {".avif", ".webp"}
        ):
            print(
                "[VIDEO:REPROCESS] 스풀 입력 파일명 오류: "
                f"value={input_filename!r}, job_dir={str(directory)!r}"
            )
            raise ValueError("영상 후처리 스풀 입력 파일명이 올바르지 않습니다")
        input_path = directory / input_filename
    else:
        input_path = directory / "input.mp4"
    if not input_path.is_file():
        print(
            "[VIDEO:POSTPROCESS] 스풀 입력 파일 누락: "
            f"job_dir={str(directory)!r}, job_kind={job_kind!r}, "
            f"input={str(input_path)!r}"
        )
        raise FileNotFoundError("영상 후처리 스풀 입력 파일이 없습니다")

    fps = (
        normalize_video_reprocess_fps(manifest.get("fps"))
        if is_existing_animation
        else int(manifest.get("fps") or 24)
    )
    duration = float(manifest.get("duration") or 5.0)
    expected_frames = max(2, round(fps * duration))
    quality = max(1, min(100, int(manifest.get("quality") or 80)))
    output_format = str(manifest.get("output_format") or "avif").strip().lower()
    if output_format not in VIDEO_OUTPUT_FORMATS:
        print(
            "[VIDEO:POSTPROCESS] 스풀 출력 형식 오류: "
            f"value={manifest.get('output_format')!r}, job_dir={str(directory)!r}"
        )
        raise ValueError("영상 출력 형식은 AVIF 또는 WebP여야 합니다")
    effective = normalize_video_postprocess_config(settings)
    upscale_enabled = bool(manifest.get("upscale_enabled", effective["enabled"]))
    effective["enabled"] = upscale_enabled
    effective["scale"] = int(manifest.get("upscale_scale") or effective["scale"])
    if upscale_enabled:
        effective["model"] = str(
            manifest.get("upscale_model") or effective["model"]
        ).strip()
    effective = normalize_video_postprocess_config(effective)

    await _notify(
        progress_callback,
        phase="video_extracting_frames",
        percentage=5,
        current=0,
        total=expected_frames,
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="work_",
            dir=str(directory),
        ) as work_name:
            work_dir = Path(work_name)
            input_frames = work_dir / "input_frames"
            input_frames.mkdir()
            if is_existing_animation:
                timing = await asyncio.to_thread(
                    _extract_animation_frames,
                    input_path,
                    input_frames,
                    fps=fps,
                    fallback_duration=duration,
                )
                duration = float(timing["duration"])
                expected_frames = int(timing["output_frame_count"])
                manifest["duration"] = duration
                manifest["source_frame_count"] = int(timing["frame_count"])
                manifest["source_fps"] = float(timing["source_fps"])
            else:
                ffmpeg = str(await ensure_ffmpeg())
                await _run_command(
                    [
                        ffmpeg,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(input_path),
                        "-t",
                        str(duration),
                        "-vf",
                        f"fps={fps}",
                        "-vsync",
                        "0",
                        "-start_number",
                        "0",
                        str(input_frames / "frame_%08d.png"),
                    ],
                    label="DECODE",
                )
            extracted = await asyncio.to_thread(_list_pngs, input_frames)
            if len(extracted) != expected_frames:
                print(
                    "[VIDEO:DECODE] 프레임 수 불일치: "
                    f"expected={expected_frames}, actual={len(extracted)}, "
                    f"input={str(input_path)!r}, job_kind={job_kind!r}"
                )
                raise RuntimeError("영상에서 예상한 수의 프레임을 얻지 못했습니다")
            await _notify(
                progress_callback,
                phase="video_frames_extracted",
                percentage=20,
                current=len(extracted),
                total=expected_frames,
            )

            if effective["enabled"]:
                await _notify(
                    progress_callback,
                    phase="video_upscaler_preparing",
                    percentage=22,
                    current=0,
                    total=expected_frames,
                )
                output_frames = work_dir / "upscaled_frames"
                if effective["model"] == "realesr-animevideov3":
                    await _run_realesrgan(
                        input_frames,
                        output_frames,
                        expected_frames=expected_frames,
                        settings=effective,
                        progress_callback=progress_callback,
                    )
                elif effective["model"] == "anime4k-fast-m":
                    await _run_anime4k(
                        input_frames,
                        output_frames,
                        expected_frames=expected_frames,
                        scale=effective["scale"],
                        progress_callback=progress_callback,
                    )
                elif effective["model"] == "lanczos":
                    await _run_lanczos(
                        input_frames,
                        output_frames,
                        expected_frames=expected_frames,
                        scale=effective["scale"],
                        progress_callback=progress_callback,
                    )
                else:
                    print(
                        "[VIDEO:POSTPROCESS] 업스케일러 분기 누락: "
                        f"model={effective['model']!r}, settings={effective!r}"
                    )
                    raise RuntimeError("영상 업스케일러 실행 경로가 없습니다")
            else:
                output_frames = input_frames
                await _notify(
                    progress_callback,
                    phase="video_upscale_skipped",
                    percentage=75,
                    current=expected_frames,
                    total=expected_frames,
                )

            overlay_path = directory / "overlay.png"
            if not overlay_path.is_file():
                overlay_path = None
            output_frame_paths = await asyncio.to_thread(_list_pngs, output_frames)
            if not output_frame_paths:
                print(
                    "[VIDEO:POSTPROCESS] 인코딩할 프레임 없음: "
                    f"job_dir={str(directory)!r}, output_dir={str(output_frames)!r}"
                )
                raise RuntimeError("인코딩할 영상 프레임이 없습니다")
            try:
                with Image.open(output_frame_paths[0]) as output_frame:
                    manifest["output_width"], manifest["output_height"] = output_frame.size
                    manifest["raw_output_height"] = output_frame.height
            except Exception as exc:
                print(
                    "[VIDEO:POSTPROCESS] 출력 프레임 크기 확인 실패: "
                    f"path={str(output_frame_paths[0])!r}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise

            if is_existing_animation:
                target_bytes = int(manifest.get("target_size_bytes") or 0)
                main_path, raw_path, extension, quality, output_size_bytes = (
                    await _encode_pair_to_target_size(
                        output_frames,
                        directory,
                        fps=fps,
                        frame_count=expected_frames,
                        target_bytes=target_bytes,
                        progress_callback=progress_callback,
                        output_format=output_format,
                    )
                )
                manifest["quality"] = quality
                manifest["output_size_bytes"] = output_size_bytes
            else:
                main_path, raw_path, extension = await _encode_pair(
                    output_frames,
                    directory,
                    fps=fps,
                    frame_count=expected_frames,
                    quality=quality,
                    overlay_path=overlay_path,
                    canvas_height=int(manifest.get("output_height") or 1),
                    progress_callback=progress_callback,
                    output_format=output_format,
                )
                output_size_bytes = main_path.stat().st_size
    except Exception as exc:
        print(
            "[VIDEO:POSTPROCESS] 작업 실패: "
            f"job_dir={str(directory)!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise

    await _notify(
        progress_callback,
        phase="video_postprocess_validated",
        percentage=97,
        current=expected_frames,
        total=expected_frames,
    )
    return {
        "manifest": manifest,
        "main_path": str(main_path),
        "raw_path": str(raw_path),
        "extension": extension,
        "frame_count": expected_frames,
        "upscale_enabled": effective["enabled"],
        "upscale_scale": effective["scale"] if effective["enabled"] else 1,
        "upscale_model": effective["model"] if effective["enabled"] else "",
        "output_format_requested": output_format,
        "quality": quality,
        "output_size_bytes": output_size_bytes,
        "target_size_bytes": int(manifest.get("target_size_bytes") or 0),
    }
