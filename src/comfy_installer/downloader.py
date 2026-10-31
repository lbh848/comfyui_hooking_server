from __future__ import annotations

import hashlib
import os
import re
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Mapping

import httpx


class DownloadError(RuntimeError):
    """다운로드, 재개 또는 무결성 검증 실패."""


class DownloadCancelled(DownloadError):
    """사용자가 설치 중단을 요청함."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    sha256: str
    reused: bool


ProgressCallback = Callable[[dict], None]
ClientFactory = Callable[[], httpx.Client]


_CONTENT_RANGE_PATTERN = re.compile(
    r"^bytes\s+(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+|\*)$",
    re.IGNORECASE,
)


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    if not isinstance(value, str):
        return None
    match = _CONTENT_RANGE_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    raw_total = match.group("total")
    total = None if raw_total == "*" else int(raw_total)
    if start > end:
        return None
    if total is not None and (total <= 0 or end >= total):
        return None
    return start, end, total


def redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url
    redacted_query = urllib.parse.urlencode(
        [
            (key, "***" if key.casefold() in {"token", "api_key", "apikey"} else value)
            for key, value in urllib.parse.parse_qsl(
                parsed.query, keep_blank_values=True
            )
        ]
    )
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, redacted_query, parsed.fragment)
    )


def sha256_file(path: Path, cancel_event: Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("SHA-256 검증 중 중단 요청을 받았습니다.")
            digest.update(chunk)
    return digest.hexdigest()


class ResumableDownloader:
    def __init__(
        self,
        *,
        client_factory: ClientFactory | None = None,
        max_retries: int = 3,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self._client_factory = client_factory or (
            lambda: httpx.Client(
                timeout=httpx.Timeout(60.0, connect=30.0, read=60.0),
                follow_redirects=True,
                headers={"User-Agent": "comfyui-hooking-server-installer/1.0"},
            )
        )
        self.max_retries = max(1, int(max_retries))
        self.chunk_size = max(64 * 1024, int(chunk_size))

    @staticmethod
    def _emit(callback: ProgressCallback | None, payload: dict) -> None:
        if callback is not None:
            callback(payload)

    @staticmethod
    def _move_invalid(path: Path, reason: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        invalid = path.with_name(f"{path.name}.invalid_{stamp}")
        suffix = 1
        while invalid.exists():
            invalid = path.with_name(f"{path.name}.invalid_{stamp}_{suffix}")
            suffix += 1
        os.replace(path, invalid)
        print(
            "[COMFY_INSTALL][DOWNLOAD] 잘못된 기존 파일 보존: "
            f"path={path}, moved={invalid}, reason={reason}"
        )
        return invalid

    def _verify_existing(
        self,
        target: Path,
        expected_size: int,
        expected_sha256: str,
        cancel_event: Event,
        progress: ProgressCallback | None,
    ) -> DownloadResult | None:
        if not target.is_file():
            return None
        size = target.stat().st_size
        if size != expected_size:
            self._move_invalid(
                target,
                f"크기 불일치 expected={expected_size}, actual={size}",
            )
            return None
        self._emit(
            progress,
            {
                "event": "verify",
                "path": str(target),
                "downloaded": size,
                "total": expected_size,
            },
        )
        actual_hash = sha256_file(target, cancel_event)
        if actual_hash != expected_sha256:
            self._move_invalid(
                target,
                f"SHA-256 불일치 expected={expected_sha256}, actual={actual_hash}",
            )
            return None
        return DownloadResult(target, size, actual_hash, True)

    def download(
        self,
        *,
        url: str,
        target: str | os.PathLike[str],
        expected_size: int,
        expected_sha256: str,
        headers: Mapping[str, str] | None = None,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> DownloadResult:
        target_path = Path(target).resolve()
        cancel = cancel_event or Event()
        expected_hash = expected_sha256.lower()
        safe_url = redact_url(url)
        if expected_size <= 0:
            raise DownloadError(f"예상 파일 크기가 유효하지 않습니다: {expected_size}")
        if len(expected_hash) != 64:
            raise DownloadError("예상 SHA-256 형식이 유효하지 않습니다.")

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._verify_existing(
                target_path,
                expected_size,
                expected_hash,
                cancel,
                progress,
            )
            if existing is not None:
                self._emit(
                    progress,
                    {
                        "event": "reused",
                        "path": str(target_path),
                        "downloaded": expected_size,
                        "total": expected_size,
                    },
                )
                return existing

            part_path = target_path.with_name(f"{target_path.name}.part")
            if part_path.is_file() and part_path.stat().st_size > expected_size:
                self._move_invalid(part_path, "부분 파일이 예상 크기보다 큼")

            last_error: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                if cancel.is_set():
                    raise DownloadCancelled("다운로드 시작 전 중단 요청을 받았습니다.")
                offset = part_path.stat().st_size if part_path.is_file() else 0
                request_headers = dict(headers or {})
                if offset:
                    request_headers["Range"] = f"bytes={offset}-"
                self._emit(
                    progress,
                    {
                        "event": "attempt",
                        "attempt": attempt,
                        "path": str(target_path),
                        "url": safe_url,
                        "downloaded": offset,
                        "total": expected_size,
                    },
                )
                started = time.monotonic()
                last_reported = started
                downloaded_this_attempt = 0
                try:
                    with self._client_factory() as client:
                        with client.stream(
                            "GET", url, headers=request_headers
                        ) as response:
                            if response.status_code == 416 and offset == expected_size:
                                pass
                            else:
                                response.raise_for_status()
                                append = False
                                skip_prefix = 0
                                if response.status_code == 206:
                                    parsed_range = _parse_content_range(
                                        response.headers.get("Content-Range")
                                    )
                                    invalid_reason: str | None = None
                                    if parsed_range is None:
                                        invalid_reason = (
                                            "206 응답의 Content-Range가 없거나 "
                                            "유효하지 않음"
                                        )
                                    else:
                                        range_start, range_end, range_total = (
                                            parsed_range
                                        )
                                        if (
                                            range_total is not None
                                            and range_total != expected_size
                                        ):
                                            invalid_reason = (
                                                "Content-Range 전체 크기 불일치 "
                                                f"expected={expected_size}, "
                                                f"actual={range_total}"
                                            )
                                        elif range_start > offset:
                                            invalid_reason = (
                                                "Content-Range 시작점이 요청 offset보다 "
                                                f"뒤임 offset={offset}, "
                                                f"start={range_start}"
                                            )
                                        elif range_end < offset:
                                            invalid_reason = (
                                                "Content-Range가 요청 offset을 포함하지 "
                                                f"않음 offset={offset}, "
                                                f"range={range_start}-{range_end}"
                                            )
                                        else:
                                            append = offset > 0
                                            skip_prefix = offset - range_start
                                    if invalid_reason is not None:
                                        if part_path.is_file():
                                            self._move_invalid(
                                                part_path,
                                                invalid_reason,
                                            )
                                        raise DownloadError(invalid_reason)
                                if offset > 0 and response.status_code != 206:
                                    print(
                                        "[COMFY_INSTALL][DOWNLOAD] 서버가 Range를 "
                                        "무시하여 부분 파일을 처음부터 다시 받습니다: "
                                        f"url={safe_url}, status={response.status_code}"
                                    )
                                    offset = 0
                                mode = "ab" if append else "wb"
                                remaining_skip = skip_prefix
                                with part_path.open(mode) as stream:
                                    for chunk in response.iter_bytes(self.chunk_size):
                                        if cancel.is_set():
                                            raise DownloadCancelled(
                                                "다운로드 중 중단 요청을 받았습니다."
                                        )
                                        if not chunk:
                                            continue
                                        if remaining_skip:
                                            skipped = min(
                                                remaining_skip,
                                                len(chunk),
                                            )
                                            remaining_skip -= skipped
                                            chunk = chunk[skipped:]
                                            if not chunk:
                                                continue
                                        stream.write(chunk)
                                        downloaded_this_attempt += len(chunk)
                                        now = time.monotonic()
                                        if now - last_reported >= 0.25:
                                            current = offset + downloaded_this_attempt
                                            elapsed = max(now - started, 0.001)
                                            self._emit(
                                                progress,
                                                {
                                                    "event": "progress",
                                                    "attempt": attempt,
                                                    "path": str(target_path),
                                                    "url": safe_url,
                                                    "downloaded": current,
                                                    "total": expected_size,
                                                    "bytes_per_second": (
                                                        downloaded_this_attempt / elapsed
                                                    ),
                                                },
                                            )
                                            last_reported = now
                                    stream.flush()
                                    os.fsync(stream.fileno())
                                if remaining_skip:
                                    raise DownloadError(
                                        "Content-Range 응답이 요청 offset까지 "
                                        "도달하지 못했습니다: "
                                        f"remaining_skip={remaining_skip}"
                                    )

                    actual_size = part_path.stat().st_size
                    if actual_size != expected_size:
                        raise DownloadError(
                            "다운로드 크기 불일치: "
                            f"path={target_path}, expected={expected_size}, "
                            f"actual={actual_size}"
                        )
                    self._emit(
                        progress,
                        {
                            "event": "verify",
                            "path": str(target_path),
                            "downloaded": actual_size,
                            "total": expected_size,
                        },
                    )
                    actual_hash = sha256_file(part_path, cancel)
                    if actual_hash != expected_hash:
                        self._move_invalid(
                            part_path,
                            "다운로드 SHA-256 불일치: "
                            f"expected={expected_hash}, actual={actual_hash}",
                        )
                        raise DownloadError(
                            f"다운로드 SHA-256 검증 실패: {target_path.name}"
                        )
                    os.replace(part_path, target_path)
                    self._emit(
                        progress,
                        {
                            "event": "complete",
                            "path": str(target_path),
                            "downloaded": actual_size,
                            "total": expected_size,
                            "sha256": actual_hash,
                        },
                    )
                    return DownloadResult(
                        target_path, actual_size, actual_hash, False
                    )
                except DownloadCancelled:
                    raise
                except Exception as exc:
                    last_error = exc
                    print(
                        "[COMFY_INSTALL][DOWNLOAD] 다운로드 시도 실패: "
                        f"attempt={attempt}/{self.max_retries}, "
                        f"url={safe_url}, target={target_path}, error={exc}"
                    )
                    traceback.print_exc()
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** (attempt - 1), 5))

            raise DownloadError(
                f"다운로드 재시도 소진: {target_path.name}, error={last_error}"
            ) from last_error
        except DownloadCancelled:
            raise
        except DownloadError:
            raise
        except Exception as exc:
            print(
                "[COMFY_INSTALL][DOWNLOAD] 처리하지 못한 다운로드 오류: "
                f"url={safe_url}, target={target_path}, error={exc}"
            )
            traceback.print_exc()
            raise DownloadError(f"다운로드 실패: {target_path.name}: {exc}") from exc
