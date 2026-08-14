"""Vast 빌드 전 짧은 전송 측정값 해석과 전송 ETA 계산."""
from __future__ import annotations

import math
import re
import traceback
from typing import Any, Mapping, Sequence


SPEED_MARKER = "__SOYA_SPEED__"
_SPEED_LINE_RE = re.compile(
    rf"{SPEED_MARKER}:(\d+):([0-9]+(?:\.[0-9]+)?):(\d{{3}}):([0-9]+(?:\.[0-9]+)?)"
)


def empty_preflight_state() -> dict[str, Any]:
    return {
        "state": "waiting",
        "started_at": "",
        "completed_at": "",
        "elapsed_seconds": 0.0,
        "tests": [],
        "estimate": {
            "available": False,
            "remaining_seconds": None,
            "download_seconds": None,
            "upload_seconds": None,
            "download_bytes": 0,
            "upload_bytes": 0,
            "note": "실측 완료 후 계산합니다.",
        },
        "error": "",
    }


def parse_curl_speed_probe(
    stdout_text: str,
    *,
    exit_code: int,
    key: str,
    label: str,
    detail: str,
) -> dict[str, Any]:
    """curl ``--write-out`` 표식을 사용자에게 보여줄 측정 결과로 바꾼다.

    제한 시간에 걸렸더라도 1바이트 이상 실제로 받았다면 partial 측정값으로
    사용한다. HTTP 오류나 표식 누락은 호출자가 실패 경로를 로깅하도록 예외로
    올린다.
    """
    matches = list(_SPEED_LINE_RE.finditer(str(stdout_text or "")))
    if not matches:
        raise ValueError(
            f"{label} curl 측정 표식이 없습니다: exit={exit_code}, "
            f"stdout_tail={str(stdout_text or '')[-300:]!r}"
        )
    match = matches[-1]
    received_bytes = int(match.group(1))
    seconds = float(match.group(2))
    http_code = int(match.group(3))
    reported_bytes_per_second = float(match.group(4))
    if http_code not in {200, 206}:
        raise ValueError(
            f"{label} 속도 측정 HTTP 오류: http={http_code}, exit={exit_code}, "
            f"bytes={received_bytes}"
        )
    if received_bytes <= 0 or seconds <= 0:
        raise ValueError(
            f"{label} 속도 측정 데이터 없음: exit={exit_code}, "
            f"bytes={received_bytes}, seconds={seconds}"
        )
    measured_bytes_per_second = received_bytes / seconds
    bytes_per_second = (
        reported_bytes_per_second
        if reported_bytes_per_second > 0
        else measured_bytes_per_second
    )
    # 일부 curl/프록시는 write-out speed와 size/time 계산이 조금 다르다.
    # ETA에는 전송 바이트와 경과시간으로 직접 계산한 보수적인 쪽을 쓴다.
    bytes_per_second = min(bytes_per_second, measured_bytes_per_second)
    status = "done" if exit_code == 0 else "partial"
    return {
        "key": str(key),
        "label": str(label),
        "status": status,
        "bytes": received_bytes,
        "seconds": round(seconds, 3),
        "bytes_per_second": round(bytes_per_second, 3),
        "mbps": round(bytes_per_second * 8 / 1_000_000, 2),
        "mb_per_second": round(bytes_per_second / 1_000_000, 2),
        "http_code": http_code,
        "detail": str(detail),
    }


def speed_result(
    *,
    key: str,
    label: str,
    transferred_bytes: int,
    seconds: float,
    detail: str,
    status: str = "done",
) -> dict[str, Any]:
    if transferred_bytes <= 0:
        raise ValueError(f"{label} 측정 바이트가 0 이하입니다: {transferred_bytes}")
    if seconds <= 0:
        raise ValueError(f"{label} 측정 시간이 0 이하입니다: {seconds}")
    bytes_per_second = transferred_bytes / seconds
    return {
        "key": str(key),
        "label": str(label),
        "status": str(status),
        "bytes": int(transferred_bytes),
        "seconds": round(float(seconds), 3),
        "bytes_per_second": round(bytes_per_second, 3),
        "mbps": round(bytes_per_second * 8 / 1_000_000, 2),
        "mb_per_second": round(bytes_per_second / 1_000_000, 2),
        "detail": str(detail),
    }


def failed_result(*, key: str, label: str, detail: str) -> dict[str, Any]:
    return {
        "key": str(key),
        "label": str(label),
        "status": "error",
        "bytes": 0,
        "seconds": 0.0,
        "bytes_per_second": 0.0,
        "mbps": 0.0,
        "mb_per_second": 0.0,
        "detail": str(detail),
    }


def informational_result(
    *, key: str, label: str, seconds: float, detail: str
) -> dict[str, Any]:
    return {
        "key": str(key),
        "label": str(label),
        "status": "done",
        "bytes": 0,
        "seconds": round(max(0.0, float(seconds)), 3),
        "bytes_per_second": 0.0,
        "mbps": 0.0,
        "mb_per_second": 0.0,
        "detail": str(detail),
    }


def _successful_speed(test: Mapping[str, Any] | None) -> float:
    if not test or str(test.get("status") or "") not in {"done", "partial"}:
        return 0.0
    try:
        return max(0.0, float(test.get("bytes_per_second") or 0.0))
    except (TypeError, ValueError) as exc:
        print(
            "[VAST][PREFLIGHT][ERROR] 측정 속도 해석 실패: "
            f"test={dict(test)!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 0.0


def calculate_transfer_estimate(
    model_plan: Mapping[str, Any],
    lora_files: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """현재 병렬 빌드 구조를 반영해 남은 전송 시간을 계산한다.

    원격 모델 다운로드와 로컬 SFTP/노드 설치 브랜치는 병렬이므로 두 전송
    시간의 합이 아니라 큰 값을 반환한다. Custom Node 설치와 ComfyUI 기동
    시간은 환경 의존성이 커 별도라는 점을 note에 명시한다.
    """
    tests_by_key = {str(item.get("key") or ""): item for item in tests}
    cloudflare_speed = _successful_speed(tests_by_key.get("cloudflare"))
    huggingface_speed = _successful_speed(tests_by_key.get("huggingface"))
    upload_speed = _successful_speed(tests_by_key.get("upload"))
    generic_download_speeds = [
        value for value in (cloudflare_speed, huggingface_speed) if value > 0
    ]
    generic_download_speed = min(generic_download_speeds, default=0.0)

    download_bytes = 0
    upload_bytes = 0
    download_seconds = 0.0
    missing_download_speed = False
    for item in model_plan.get("models") or []:
        if not isinstance(item, Mapping):
            print(
                "[VAST][PREFLIGHT][ERROR] ETA 모델 항목 형식 이상(건너뜀): "
                f"type={type(item).__name__}, value={item!r}"
            )
            continue
        try:
            size_bytes = max(0, int(item.get("size_bytes") or 0))
        except (TypeError, ValueError) as exc:
            print(
                "[VAST][PREFLIGHT][ERROR] ETA 모델 크기 해석 실패: "
                f"key={item.get('key')!r}, size={item.get('size_bytes')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            size_bytes = 0
        source = item.get("source") or {}
        source_type = str(source.get("source_type") or "upload")
        if source_type in {"hf", "civitai", "url"}:
            download_bytes += size_bytes
            speed = (
                huggingface_speed
                if source_type == "hf" and huggingface_speed > 0
                else generic_download_speed
            )
            if size_bytes > 0 and speed <= 0:
                missing_download_speed = True
            elif speed > 0:
                download_seconds += size_bytes / speed
        else:
            upload_bytes += size_bytes

    for item in lora_files:
        if not isinstance(item, Mapping):
            print(
                "[VAST][PREFLIGHT][ERROR] ETA LoRA 항목 형식 이상(건너뜀): "
                f"type={type(item).__name__}, value={item!r}"
            )
            continue
        try:
            upload_bytes += max(
                0, int(item.get("size_bytes") or item.get("size") or 0)
            )
        except (TypeError, ValueError) as exc:
            print(
                "[VAST][PREFLIGHT][ERROR] ETA LoRA 크기 해석 실패: "
                f"name={item.get('name')!r}, "
                f"size={item.get('size_bytes') or item.get('size')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            continue

    missing_upload_speed = upload_bytes > 0 and upload_speed <= 0
    upload_seconds = upload_bytes / upload_speed if upload_speed > 0 else 0.0
    available = not missing_download_speed and not missing_upload_speed
    if available:
        remaining_seconds: int | None = math.ceil(
            max(download_seconds, upload_seconds)
        )
        note = (
            "원격 다운로드와 로컬 업로드의 병렬 전송 기준입니다. "
            "Custom Node 설치와 ComfyUI 기동 시간은 포함하지 않습니다."
        )
    else:
        remaining_seconds = None
        missing: list[str] = []
        if missing_download_speed:
            missing.append("다운로드")
        if missing_upload_speed:
            missing.append("업로드")
        note = f"{'·'.join(missing)} 실측 실패로 ETA를 계산할 수 없습니다."

    return {
        "available": available,
        "remaining_seconds": remaining_seconds,
        "download_seconds": (
            round(download_seconds, 1) if not missing_download_speed else None
        ),
        "upload_seconds": (
            round(upload_seconds, 1) if not missing_upload_speed else None
        ),
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "note": note,
    }
