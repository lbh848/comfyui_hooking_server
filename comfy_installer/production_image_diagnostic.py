from __future__ import annotations

import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import time
import traceback
import uuid
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from PIL import Image

from .e2e import ComfyE2ECancelled
from .runtime_state import git_head


ProductionCall = Callable[[dict[str, Any]], dict[str, Any]]
LogCallback = Callable[..., None]
ProgressCallback = Callable[[dict[str, Any]], None]

MEASUREMENT_VARIANTS = (
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, neutral expression, standing, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, happy, open mouth, waving, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking to the side, blonde hair, orange eyes, white shirt, pleated skirt, surprised, raised eyebrows, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, upper body, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, angry, frown, crossed arms, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking down, blonde hair, orange eyes, white shirt, pleated skirt, sad, closed mouth, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, portrait, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, embarrassed, blush, shy, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, facing to the side, blonde hair, orange eyes, white shirt, pleated skirt, thinking, hand on chin, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, full body, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, confident, hand on hip, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, upper body, looking away, blonde hair, orange eyes, white shirt, pleated skirt, worried, parted lips, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, laughing, closed eyes, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, portrait, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, sleepy, half-closed eyes, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, full body, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, dynamic pose, reaching out, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking back, blonde hair, orange eyes, white shirt, pleated skirt, startled, open mouth, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, upper body, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, serious, arms at sides, simple background",
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, gentle smile, head tilt, simple background",
)

WARMUP_VARIANT = (
    "newest, year2024, masterpiece, best quality, 1girl, solo, cowboy shot, "
    "looking at viewer, blonde hair, orange eyes, white shirt, pleated skirt, "
    "neutral expression, standing, simple background"
)

_BAD_LOG_TOKENS = (
    "runtimewarning: invalid value encountered in cast",
    "nan detected",
    "inf detected",
    "non-finite",
)


def _log(callback: LogCallback | None, message: str, level: str = "info") -> None:
    if callback is None:
        return
    try:
        callback(message, level)
    except TypeError:
        callback(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_git_head(path: Path) -> str | None:
    try:
        return git_head(path)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] Git HEAD 조회 실패: "
            f"path={path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return None


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] nvidia-smi 실패: "
                f"returncode={completed.returncode}, stderr={completed.stderr[:1000]}"
            )
            return {"available": False, "error": completed.stderr.strip()[:1000]}
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "index": parts[0] if len(parts) > 0 else None,
                    "name": parts[1] if len(parts) > 1 else None,
                    "driver": parts[2] if len(parts) > 2 else None,
                    "memory_used_mib": parts[3] if len(parts) > 3 else None,
                    "memory_total_mib": parts[4] if len(parts) > 4 else None,
                    "temperature_c": parts[5] if len(parts) > 5 else None,
                    "utilization_percent": parts[6] if len(parts) > 6 else None,
                }
            )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] "
            f"nvidia-smi 실행 예외: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _image_metrics(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
        numeric = array.astype(np.float32)
        gray = numeric.mean(axis=2)
        histogram = np.bincount(gray.astype(np.uint8).reshape(-1), minlength=256)
        probabilities = histogram[histogram > 0].astype(np.float64)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        horizontal = (
            float(np.abs(numeric[:, 1:] - numeric[:, :-1]).mean())
            if numeric.shape[1] > 1
            else 0.0
        )
        vertical = (
            float(np.abs(numeric[1:] - numeric[:-1]).mean())
            if numeric.shape[0] > 1
            else 0.0
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = {
            "decoded": True,
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "sha256": digest,
            "mean": round(float(numeric.mean()), 6),
            "std": round(float(numeric.std()), 6),
            "channel_mean": [round(float(value), 6) for value in numeric.mean(axis=(0, 1))],
            "channel_std": [round(float(value), 6) for value in numeric.std(axis=(0, 1))],
            "black_fraction": round(float((gray <= 3.0).mean()), 8),
            "near_black_fraction": round(float((gray <= 12.0).mean()), 8),
            "entropy": round(entropy, 6),
            "neighbor_difference": round((horizontal + vertical) / 2.0, 6),
            "file_bytes": path.stat().st_size,
        }
        reasons: list[str] = []
        if metrics["black_fraction"] >= 0.98:
            reasons.append("픽셀의 98% 이상이 검정")
        if metrics["mean"] <= 3.0 and metrics["std"] <= 3.0:
            reasons.append("평균·표준편차가 모두 거의 0")
        if (
            metrics["neighbor_difference"] >= 70.0
            and metrics["entropy"] >= 7.5
        ):
            reasons.append("고주파 컬러 노이즈 의심")
        metrics["pixel_abnormal"] = bool(reasons)
        metrics["pixel_abnormal_reasons"] = reasons
        return metrics
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 이미지 계측 실패: "
            f"path={path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return {
            "decoded": False,
            "pixel_abnormal": True,
            "pixel_abnormal_reasons": [f"이미지 디코드 실패: {type(exc).__name__}: {exc}"],
        }


def _log_findings(text: str) -> list[str]:
    folded = text.casefold()
    return [token for token in _BAD_LOG_TOKENS if token in folded]


def _wait_until_ready(
    production_call: ProductionCall,
    *,
    cancel_event: Event,
    timeout: float = 900.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if cancel_event.is_set():
            raise ComfyE2ECancelled("실제 프로그램 이미지 진단 준비 중 중단 요청을 받았습니다.")
        try:
            last = production_call({"action": "status"})
            if last.get("queue_idle") and last.get("runtime_ready"):
                return last
        except Exception as exc:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 준비 상태 조회 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            last = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(1.0)
    raise RuntimeError(f"관리 Comfy/작업 큐 준비 시간 초과: last={last}")


def _restart_managed_runtime(
    *,
    production_call: ProductionCall,
    pause_managed_comfy: Callable[[], Any],
    resume_managed_comfy: Callable[[Any], Any],
    cancel_event: Event,
    log: LogCallback | None,
    case_name: str,
) -> dict[str, Any]:
    before = production_call({"action": "status"})
    if not before.get("queue_idle"):
        raise RuntimeError(
            "이미지 진단 시작 시 작업 큐가 비어 있지 않습니다: "
            f"case={case_name}, status={before}"
        )
    if not before.get("runtime_running"):
        raise RuntimeError(
            "실제 프로그램이 관리하는 로컬 Comfy가 실행 중이 아닙니다: "
            f"case={case_name}, status={before}"
        )
    token = pause_managed_comfy()
    resumed = False
    try:
        resumed_result = resume_managed_comfy(token)
        resumed = True
        ready = _wait_until_ready(
            production_call,
            cancel_event=cancel_event,
        )
        ready_with_logs = production_call(
            {"action": "status", "include_logs": True}
        )
        if not ready_with_logs.get("runtime_ready"):
            raise RuntimeError(
                "관리 Comfy 시작 로그 수집 시 runtime_ready가 해제되었습니다: "
                f"case={case_name}, status={ready_with_logs}"
            )
        ready = ready_with_logs
        _log(
            log,
            f"[이미지 진단] {case_name}: 관리 Comfy 동일 설정 재시작 및 준비 완료",
        )
        return {
            "before": before,
            "pause": token,
            "resume": resumed_result,
            "ready": ready,
        }
    except Exception:
        if not resumed:
            try:
                resume_managed_comfy(token)
            except Exception as resume_exc:
                print(
                    "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] "
                    f"재시작 실패 후 원래 관리 Comfy 복구 실패: {resume_exc}"
                )
                traceback.print_exc()
        raise


def _copy_result_artifacts(
    *,
    result: Mapping[str, Any],
    report_dir: Path,
    case_name: str,
    run_label: str,
) -> tuple[Path, Path | None]:
    source = Path(str(result.get("local_path") or "")).resolve()
    if not source.is_file():
        raise RuntimeError(
            "실제 에셋 생성 결과 파일이 없습니다: "
            f"case={case_name}, run={run_label}, result={dict(result)}"
        )
    target_dir = report_dir / "images" / case_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{run_label}{source.suffix.lower()}"
    shutil.copy2(source, target)

    prompt_source_text = str(result.get("prompt_record_path") or "")
    prompt_target = None
    if prompt_source_text:
        prompt_source = Path(prompt_source_text).resolve()
        if prompt_source.is_file():
            prompt_target = target_dir / f"{run_label}_prompt.json"
            shutil.copy2(prompt_source, prompt_target)
        else:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 프롬프트 기록 없음: "
                f"path={prompt_source}, result={dict(result)}"
            )
    return target, prompt_target


def _case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(case.get("runs") or [])
    abnormal = [run for run in runs if run.get("abnormal")]
    return {
        "name": case.get("name"),
        "dcw_cwm_smc_enabled": case.get("dcw_cwm_smc_enabled"),
        "model_patcher_refresh": case.get("model_patcher_refresh"),
        "completed": len(runs),
        "abnormal": len(abnormal),
        "first_abnormal": abnormal[0].get("index") if abnormal else None,
        "first_abnormal_reasons": abnormal[0].get("abnormal_reasons") if abnormal else [],
    }


def _conclusions(cases: list[dict[str, Any]]) -> list[str]:
    summaries = {case["name"]: _case_summary(case) for case in cases}
    baseline = summaries.get("production_patch_on", {})
    refreshed = summaries.get("production_model_refresh", {})
    patch_off = summaries.get("production_patch_off", {})
    baseline_bad = int(baseline.get("abnormal") or 0) > 0
    refresh_bad = int(refreshed.get("abnormal") or 0) > 0
    off_bad = int(patch_off.get("abnormal") or 0) > 0
    if baseline_bad and not refresh_bad:
        return [
            "기존 실제 경로에서는 깨졌지만 sampler 직전 ModelPatcher Refresh 경로는 안정적이었습니다. 매 생성 새 ModelPatcher wrapper를 만드는 동작이 문제를 회피한다는 강한 증거입니다.",
            "Refresh는 모델 unload·VRAM 정리·LoRA/DCW 제거 없이 이전 진단 ModelProbe의 clone 경계만 재현합니다.",
        ]
    if baseline_bad and refresh_bad and not off_bad:
        return [
            "기존 경로와 ModelPatcher Refresh 경로는 모두 깨졌지만 DCW/CWM OFF는 안정적이었습니다. 단순 wrapper 재생성보다 DCW/CWM 연산 경로가 강한 원인 후보입니다.",
        ]
    if baseline_bad and refresh_bad and off_bad:
        return [
            "세 경로 모두 깨졌습니다. ModelPatcher wrapper 재생성과 DCW/CWM 비활성화로 회피되지 않으므로 그보다 앞선 모델 weight patch·상주 상태·conditioning 경로를 우선 확인해야 합니다.",
        ]
    if not baseline_bad and refresh_bad:
        return [
            "ModelPatcher Refresh 경로에서만 이상이 검출되었습니다. clone 경계가 해결책이라는 가설과 반대이므로 해당 케이스 로그와 제출 워크플로를 우선 확인해야 합니다.",
        ]
    if not baseline_bad and not refresh_bad and off_bad:
        return [
            "DCW/CWM OFF 경로에서만 이상이 검출되었습니다. DCW/CWM 또는 ModelPatcher warm 재사용이 원인이라는 가설과 반대 결과입니다.",
        ]
    return [
        "자동 검출 범위인 검정/고주파 컬러 노이즈/이미지 디코드 실패/비정상값 로그에서는 이상이 발견되지 않았습니다.",
        "프롬프트 무시·미완성 전조는 픽셀 통계만으로 단정하지 않습니다. images/의 같은 번호 기존/Refresh/DCW OFF 이미지와 실제 prompt JSON을 직접 대조해야 합니다.",
        "이 결과는 별도 진단 그래프가 아니라 사용자가 실제로 쓰는 관리 Comfy와 AssetMode 생성 경로에서 얻었습니다.",
    ]


def _write_report(
    path: Path,
    *,
    diagnostic_id: str,
    environment: Mapping[str, Any],
    cases: list[dict[str, Any]],
    conclusions: list[str],
    errors: list[str],
) -> None:
    lines = [
        "# 실제 프로그램 경로 이미지 깨짐 진단",
        "",
        f"- 진단 ID: `{diagnostic_id}`",
        "- 실행 경로: 프로그램 작업 큐 → AssetMode.generate → 설치된 전체 에셋 워크플로 → 관리 Comfy",
        "- 사용하지 않은 것: 독립 E2E Comfy, 진단 ModelProbe/계측 wrapper, sampler/VAE 축소 그래프",
        "- 사용자 선택 LoRA/캐릭터·얼굴·그림체 LoRA/Face ID/Style/Pose/Hires/Detailer: 모두 OFF",
        "- 팩 워크플로에 고정된 모델·LoRA 노드는 실제 배포 경로 보존을 위해 제거하지 않음",
        "- 비교: 동일한 warmup 1개+측정 15개 프롬프트와 동일 seed로 기존 경로 / sampler 직전 ModelPatcher Refresh / DCW/CWM OFF",
        "- Refresh는 model.clone() wrapper만 새로 만들며 모델 unload·VRAM 정리·weight patch 삭제를 하지 않음",
        "- 자동 판정 범위: 검정/고주파 컬러 노이즈/디코드 실패/NaN·Inf 로그. 프롬프트 무시·미완성은 이미지와 prompt JSON 직접 대조",
        "",
        "## 결론",
        "",
    ]
    lines.extend(f"- {value}" for value in conclusions)
    lines.extend(["", "## 환경", "", "```json", json.dumps(environment, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(
        [
            "## 케이스 결과",
            "",
            "| 케이스 | DCW/CWM/SMC | ModelPatcher Refresh | 완료 | 이상 | 최초 이상 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        summary = _case_summary(case)
        lines.append(
            f"| {summary['name']} | {summary['dcw_cwm_smc_enabled']} | "
            f"{summary['model_patcher_refresh']} | "
            f"{summary['completed']} | {summary['abnormal']} | "
            f"{summary['first_abnormal'] or '-'} |"
        )
    for case in cases:
        lines.extend(["", f"### {case['name']}", ""])
        lines.append("| # | phase | seed | 초 | 평균 | 표준편차 | 검정비율 | 인접차 | 로그 경고 | 판정 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
        for run in case.get("runs", []):
            metrics = run.get("image_metrics") or {}
            lines.append(
                f"| {run.get('index')} | {run.get('phase')} | {run.get('seed')} | "
                f"{run.get('duration_seconds')} | {metrics.get('mean', '-')} | "
                f"{metrics.get('std', '-')} | {metrics.get('black_fraction', '-')} | "
                f"{metrics.get('neighbor_difference', '-')} | "
                f"{', '.join(run.get('log_findings') or []) or '-'} | "
                f"{'; '.join(run.get('abnormal_reasons') or []) or '정상'} |"
            )
    if errors:
        lines.extend(["", "## 실행 오류", ""])
        lines.extend(f"- {value}" for value in errors)
    lines.extend(
        [
            "",
            "## 파일 안내",
            "",
            "- `images/`: 실제 프로그램이 저장한 결과 이미지와 대응 프롬프트",
            "- `logs/`: 케이스별 관리 Comfy 시작 로그와 실행별 원본 로그 조각",
            "- `workflows/`: 세 케이스 warmup에서 실제 Comfy에 제출한 최종 API 워크플로",
            "- `runs.json`: GPU 전후 상태, 이미지 수치, 유효 DCW 값, 큐 결과",
            "- `environment.json`: 실제 관리 Comfy 실행 명령·프로필·GPU 환경",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _archive_directory(source: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def run_production_image_diagnostic(
    *,
    project_root: Path,
    comfy_root: Path,
    cancel_event: Event,
    production_call: ProductionCall,
    pause_managed_comfy: Callable[[], Any],
    resume_managed_comfy: Callable[[Any], Any],
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    diagnostic_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    root = project_root / ".work" / "comfy-installer" / "image-diagnostics"
    report_dir = root / diagnostic_id
    archive_path = root / f"{diagnostic_id}.zip"
    report_dir.mkdir(parents=True, exist_ok=False)
    generated_character = f"image_diagnostic_{diagnostic_id.replace('-', '_')}"
    generated_root = (project_root / "asset" / generated_character).resolve()
    errors: list[str] = []
    cases: list[dict[str, Any]] = []
    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "project_head": _safe_git_head(project_root),
        "comfy_head": _safe_git_head(comfy_root),
        "gpu_at_start": _gpu_snapshot(),
        "production_status_at_start": {},
    }
    started = time.monotonic()
    try:
        initial = production_call({"action": "status"})
        environment["production_status_at_start"] = initial
        if not initial.get("queue_idle"):
            raise RuntimeError(f"진단 시작 전에 프로그램 작업 큐가 비어 있지 않습니다: {initial}")
        if not initial.get("runtime_running"):
            raise RuntimeError(f"관리 Comfy가 실행 중이 아닙니다: {initial}")
        if initial.get("execution_target") != "local":
            raise RuntimeError(
                "에셋 생성 대상이 로컬 관리 Comfy가 아닙니다: "
                f"target={initial.get('execution_target')!r}"
            )

        plans = (
            ("production_patch_on", True, False),
            ("production_model_refresh", True, True),
            ("production_patch_off", False, False),
        )
        total_runs = len(plans) * (1 + len(MEASUREMENT_VARIANTS))
        completed_runs = 0
        for case_name, patch_enabled, model_patcher_refresh in plans:
            restart = _restart_managed_runtime(
                production_call=production_call,
                pause_managed_comfy=pause_managed_comfy,
                resume_managed_comfy=resume_managed_comfy,
                cancel_event=cancel_event,
                log=log,
                case_name=case_name,
            )
            case: dict[str, Any] = {
                "name": case_name,
                "dcw_cwm_smc_enabled": patch_enabled,
                "model_patcher_refresh": model_patcher_refresh,
                "restart": restart,
                "runs": [],
            }
            cases.append(case)
            startup_log = str((restart.get("ready") or {}).get("comfy_log") or "")
            startup_log_path = report_dir / "logs" / case_name / "startup.log"
            startup_log_path.parent.mkdir(parents=True, exist_ok=True)
            startup_log_path.write_text(startup_log, encoding="utf-8")
            run_plan = [("warmup", 0, 910001, WARMUP_VARIANT)]
            run_plan.extend(
                ("measurement", index, 910101 + index * 7919, variant)
                for index, variant in enumerate(MEASUREMENT_VARIANTS, start=1)
            )
            for phase, index, seed, variant in run_plan:
                if cancel_event.is_set():
                    raise ComfyE2ECancelled("실제 프로그램 이미지 진단 중단 요청을 받았습니다.")
                completed_runs += 1
                if progress:
                    progress(
                        {
                            "event": "production_image_diagnostic_run",
                            "case": case_name,
                            "phase": phase,
                            "current": completed_runs,
                            "total": total_runs,
                            "run": index,
                        }
                    )
                run_label = "warmup" if phase == "warmup" else f"{index:02d}"
                _log(
                    log,
                    f"[이미지 진단] {case_name} {run_label}: 실제 프로그램 작업 큐 생성 시작",
                )
                before_gpu = _gpu_snapshot()
                run_started = time.monotonic()
                response = production_call(
                    {
                        "action": "generate",
                        "diagnostic_id": diagnostic_id,
                        "character": generated_character,
                        "case": case_name,
                        "phase": phase,
                        "index": index,
                        "seed": seed,
                        "variant": variant,
                        "dcw_cwm_smc_enabled": patch_enabled,
                        "model_patcher_refresh": model_patcher_refresh,
                    }
                )
                duration = round(time.monotonic() - run_started, 3)
                result = response.get("result")
                if not isinstance(result, dict) or not result.get("success"):
                    raise RuntimeError(
                        "실제 프로그램 작업 큐 이미지 생성 실패: "
                        f"case={case_name}, run={run_label}, response={response}"
                    )
                workflow_snapshot = result.pop("diagnostic_workflow", None)
                if phase == "warmup":
                    if not isinstance(workflow_snapshot, dict):
                        raise RuntimeError(
                            "warmup에서 실제 제출 워크플로를 수집하지 못했습니다: "
                            f"case={case_name}, type={type(workflow_snapshot).__name__}"
                        )
                    _write_json(
                        report_dir / "workflows" / f"{case_name}.json",
                        workflow_snapshot,
                    )
                image_path, prompt_path = _copy_result_artifacts(
                    result=result,
                    report_dir=report_dir,
                    case_name=case_name,
                    run_label=run_label,
                )
                metrics = _image_metrics(image_path)
                comfy_log = str(response.get("comfy_log") or "")
                log_path = report_dir / "logs" / case_name / f"{run_label}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(comfy_log, encoding="utf-8")
                findings = _log_findings(comfy_log)
                reasons = list(metrics.get("pixel_abnormal_reasons") or [])
                reasons.extend(f"Comfy 로그: {finding}" for finding in findings)
                run = {
                    "index": index,
                    "phase": phase,
                    "seed": seed,
                    "variant": variant,
                    "duration_seconds": duration,
                    "image": image_path.relative_to(report_dir).as_posix(),
                    "prompt_record": (
                        prompt_path.relative_to(report_dir).as_posix()
                        if prompt_path is not None
                        else None
                    ),
                    "image_metrics": metrics,
                    "log_findings": findings,
                    "abnormal": bool(reasons),
                    "abnormal_reasons": reasons,
                    "gpu_before": before_gpu,
                    "gpu_after": _gpu_snapshot(),
                    "queue_item_id": response.get("queue_item_id"),
                    "runtime": response.get("runtime"),
                    "diagnostic_model_patch": result.get("diagnostic_model_patch"),
                    "diagnostic_model_patcher_refresh": result.get(
                        "diagnostic_model_patcher_refresh"
                    ),
                }
                case["runs"].append(run)
                _write_json(report_dir / "runs.json", {"cases": cases, "errors": errors})
                if run["abnormal"]:
                    _log(
                        log,
                        f"[이미지 진단] {case_name} {run_label}: "
                        + "; ".join(reasons),
                        "warning",
                    )

        conclusions = _conclusions(cases)
        environment["gpu_at_end"] = _gpu_snapshot()
        environment["duration_seconds"] = round(time.monotonic() - started, 3)
        _write_json(report_dir / "environment.json", environment)
        _write_json(report_dir / "runs.json", {"cases": cases, "errors": errors})
        _write_report(
            report_dir / "REPORT.md",
            diagnostic_id=diagnostic_id,
            environment=environment,
            cases=cases,
            conclusions=conclusions,
            errors=errors,
        )
        _archive_directory(report_dir, archive_path)
        return {
            "operation": "image_diagnostic",
            "diagnostic_id": diagnostic_id,
            "archive_id": diagnostic_id,
            "archive_name": archive_path.name,
            "archive_path": str(archive_path),
            "incomplete": False,
            "conclusions": conclusions,
            "case_summaries": [_case_summary(case) for case in cases],
        }
    except ComfyE2ECancelled:
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(message)
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 실제 프로그램 진단 실패: "
            f"{message}"
        )
        traceback.print_exc()
        conclusions = [
            "실제 프로그램 경로 진단을 끝까지 수행하지 못했습니다. REPORT.md와 runs.json의 실패 위치 및 관리 Comfy 로그를 확인해야 합니다."
        ]
        environment["gpu_at_failure"] = _gpu_snapshot()
        environment["duration_seconds"] = round(time.monotonic() - started, 3)
        _write_json(report_dir / "environment.json", environment)
        _write_json(report_dir / "runs.json", {"cases": cases, "errors": errors})
        _write_report(
            report_dir / "REPORT.md",
            diagnostic_id=diagnostic_id,
            environment=environment,
            cases=cases,
            conclusions=conclusions,
            errors=errors,
        )
        _archive_directory(report_dir, archive_path)
        return {
            "operation": "image_diagnostic",
            "diagnostic_id": diagnostic_id,
            "archive_id": diagnostic_id,
            "archive_name": archive_path.name,
            "archive_path": str(archive_path),
            "incomplete": True,
            "conclusions": conclusions,
            "case_summaries": [_case_summary(case) for case in cases],
            "errors": errors,
        }
    finally:
        asset_root = (project_root / "asset").resolve()
        try:
            if generated_root.parent != asset_root:
                raise RuntimeError(
                    "진단 생성물 정리 대상이 asset 바로 아래가 아님: "
                    f"target={generated_root}, asset_root={asset_root}"
                )
            if generated_root.is_dir():
                shutil.rmtree(generated_root)
                print(
                    "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] "
                    f"ZIP 복사 후 임시 에셋 정리 완료: {generated_root}"
                )
        except Exception as cleanup_exc:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 임시 에셋 정리 실패: "
                f"target={generated_root}, error={type(cleanup_exc).__name__}: {cleanup_exc}"
            )
            traceback.print_exc()
