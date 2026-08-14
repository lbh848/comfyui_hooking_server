"""Vast 준비 상태와 실제 파일 전송 기반 ETA 계산."""
from __future__ import annotations

import math
import traceback
from typing import Any, Mapping, Sequence


TRANSFER_KEYS = ("download", "upload")


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
            "download_completed_bytes": 0,
            "upload_completed_bytes": 0,
            "note": "실제 파일 전송이 시작되면 계산합니다.",
        },
        "error": "",
    }


def failed_result(*, key: str, label: str, detail: str) -> dict[str, Any]:
    return {
        "key": str(key),
        "label": str(label),
        "status": "error",
        "bytes": 0,
        "total_bytes": 0,
        "total_known": False,
        "seconds": 0.0,
        "bytes_per_second": 0.0,
        "mbps": 0.0,
        "mb_per_second": 0.0,
        "detail": str(detail),
    }


def informational_result(
    *,
    key: str,
    label: str,
    seconds: float,
    detail: str,
    status: str = "done",
) -> dict[str, Any]:
    return {
        "key": str(key),
        "label": str(label),
        "status": str(status),
        "bytes": 0,
        "total_bytes": 0,
        "total_known": True,
        "seconds": round(max(0.0, float(seconds)), 3),
        "bytes_per_second": 0.0,
        "mbps": 0.0,
        "mb_per_second": 0.0,
        "detail": str(detail),
    }


def actual_transfer_result(
    *,
    key: str,
    label: str,
    status: str,
    completed_bytes: int,
    total_bytes: int,
    total_known: bool,
    seconds: float,
    bytes_per_second: float,
    detail: str,
) -> dict[str, Any]:
    completed = max(0, int(completed_bytes))
    total = max(0, int(total_bytes))
    speed = max(0.0, float(bytes_per_second))
    return {
        "key": str(key),
        "label": str(label),
        "status": str(status),
        "bytes": completed,
        "total_bytes": total,
        "total_known": bool(total_known),
        "seconds": round(max(0.0, float(seconds)), 3),
        "bytes_per_second": round(speed, 3),
        "mbps": round(speed * 8 / 1_000_000, 2),
        "mb_per_second": round(speed / 1_000_000, 2),
        "detail": str(detail),
    }


def calculate_transfer_totals(
    model_plan: Mapping[str, Any],
    lora_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """계획에서 실제 다운로드·업로드 대상 크기와 크기 신뢰 여부를 구한다."""
    totals = {"download": 0, "upload": 0}
    known = {"download": True, "upload": True}

    def add_size(branch: str, value: Any, *, context: str) -> None:
        try:
            size = int(value or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VAST][TRANSFER][ERROR] 전송 대상 크기 해석 실패: "
                f"branch={branch}, context={context}, size={value!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            known[branch] = False
            return
        if size <= 0:
            print(
                "[VAST][TRANSFER][ERROR] 전송 대상 크기가 없어 ETA에서 확정할 수 없습니다: "
                f"branch={branch}, context={context}, size={size}"
            )
            known[branch] = False
            return
        totals[branch] += size

    for item in model_plan.get("models") or []:
        if not isinstance(item, Mapping):
            print(
                "[VAST][TRANSFER][ERROR] 모델 전송 항목 형식 이상: "
                f"type={type(item).__name__}, value={item!r}"
            )
            known["download"] = False
            known["upload"] = False
            continue
        source = item.get("source") or {}
        if not isinstance(source, Mapping):
            print(
                "[VAST][TRANSFER][ERROR] 모델 source 형식 이상: "
                f"key={item.get('key')!r}, type={type(source).__name__}, value={source!r}"
            )
            known["download"] = False
            known["upload"] = False
            continue
        source_type = str(source.get("source_type") or "upload")
        branch = "download" if source_type in {"hf", "civitai", "url"} else "upload"
        add_size(
            branch,
            item.get("size_bytes"),
            context=f"model={item.get('key') or item.get('filename') or '(unknown)'}",
        )

    for item in lora_files:
        if not isinstance(item, Mapping):
            print(
                "[VAST][TRANSFER][ERROR] LoRA 전송 항목 형식 이상: "
                f"type={type(item).__name__}, value={item!r}"
            )
            known["upload"] = False
            continue
        add_size(
            "upload",
            item.get("size_bytes") or item.get("size"),
            context=f"lora={item.get('name') or item.get('path') or '(unknown)'}",
        )

    return {
        "download_bytes": totals["download"],
        "upload_bytes": totals["upload"],
        "download_total_known": known["download"],
        "upload_total_known": known["upload"],
    }


def calculate_actual_transfer_estimate(
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """실제 누적 전송량과 최근 속도로 병렬 전송의 남은 시간을 계산한다."""
    tests_by_key = {str(item.get("key") or ""): item for item in tests}
    remaining_by_key: dict[str, float | None] = {}
    total_by_key: dict[str, int] = {}
    completed_by_key: dict[str, int] = {}
    waiting: list[str] = []
    unknown: list[str] = []
    failed: list[str] = []
    labels = {"download": "다운로드", "upload": "업로드"}

    for key in TRANSFER_KEYS:
        test = tests_by_key.get(key)
        if not test:
            total_by_key[key] = 0
            completed_by_key[key] = 0
            remaining_by_key[key] = None
            waiting.append(labels[key])
            continue
        status = str(test.get("status") or "waiting")
        try:
            total = max(0, int(test.get("total_bytes") or 0))
            completed = max(0, int(test.get("bytes") or 0))
            speed = max(0.0, float(test.get("bytes_per_second") or 0.0))
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VAST][TRANSFER][ERROR] 실제 전송 ETA 수치 해석 실패: "
                f"key={key}, test={dict(test)!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            total = 0
            completed = 0
            speed = 0.0
            status = "error"
        total_known = bool(test.get("total_known", True))
        total_by_key[key] = total
        completed_by_key[key] = completed

        if status == "error":
            failed.append(labels[key])
            remaining_by_key[key] = None
            continue
        if status in {"done", "skipped"} or (total_known and completed >= total):
            remaining_by_key[key] = 0.0
            continue
        if not total_known:
            unknown.append(labels[key])
            remaining_by_key[key] = None
            continue
        if total <= 0:
            remaining_by_key[key] = 0.0
            continue
        if speed <= 0:
            waiting.append(labels[key])
            remaining_by_key[key] = None
            continue
        remaining_by_key[key] = max(0, total - completed) / speed

    available = not failed and not unknown and not waiting
    if available:
        remaining_seconds: int | None = math.ceil(
            max(float(remaining_by_key[key] or 0.0) for key in TRANSFER_KEYS)
        )
        if remaining_seconds == 0:
            note = "실제 모델 파일 전송이 완료되었습니다."
        else:
            note = (
                "현재 실제 파일 전송 속도 기준입니다. 다운로드와 업로드가 병렬이므로 "
                "더 오래 남은 경로를 전체 ETA로 표시합니다."
            )
    else:
        remaining_seconds = None
        if failed:
            note = f"{'·'.join(failed)} 실패로 ETA를 계산할 수 없습니다."
        elif unknown:
            note = f"{'·'.join(unknown)} 대상 크기를 알 수 없어 ETA를 확정할 수 없습니다."
        else:
            note = f"{'·'.join(waiting)} 실제 전송 속도를 수집하고 있습니다."

    return {
        "available": available,
        "remaining_seconds": remaining_seconds,
        "download_seconds": (
            round(float(remaining_by_key["download"]), 1)
            if remaining_by_key.get("download") is not None
            else None
        ),
        "upload_seconds": (
            round(float(remaining_by_key["upload"]), 1)
            if remaining_by_key.get("upload") is not None
            else None
        ),
        "download_bytes": total_by_key.get("download", 0),
        "upload_bytes": total_by_key.get("upload", 0),
        "download_completed_bytes": completed_by_key.get("download", 0),
        "upload_completed_bytes": completed_by_key.get("upload", 0),
        "note": note,
    }
