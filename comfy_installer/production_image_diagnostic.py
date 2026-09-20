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

_PROBE_LOG_PREFIX = "[Soya:ImageDiagnosticProbe] "
_EXPECTED_PROBE_STAGES = {
    "model_before_sampler",
    "conditioning_positive",
    "conditioning_negative",
    "latent_before_sampler",
    "latent_after_sampler",
}


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


def _custom_node_revisions(comfy_root: Path) -> list[dict[str, Any]]:
    custom_nodes = comfy_root / "custom_nodes"
    if not custom_nodes.is_dir():
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] custom_nodes 폴더 없음: "
            f"path={custom_nodes}"
        )
        return []
    revisions: list[dict[str, Any]] = []
    for path in sorted(custom_nodes.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir() or not (path / ".git").exists():
            continue
        entry: dict[str, Any] = {"name": path.name, "path": str(path)}
        try:
            head = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            status = subprocess.run(
                ["git", "-C", str(path), "status", "--short"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            entry["head"] = head.stdout.strip() if head.returncode == 0 else None
            entry["dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
            entry["status"] = status.stdout.splitlines()[:200]
            if head.returncode != 0 or status.returncode != 0:
                entry["error"] = (
                    f"head_rc={head.returncode}, status_rc={status.returncode}, "
                    f"head_stderr={head.stderr[:500]!r}, status_stderr={status.stderr[:500]!r}"
                )
                print(
                    "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] custom node revision 조회 일부 실패: "
                    f"entry={entry}"
                )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] custom node revision 조회 예외: "
                f"path={path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        revisions.append(entry)
    return revisions


def _python_packages(status: Mapping[str, Any]) -> dict[str, Any]:
    runtime = status.get("runtime") if isinstance(status, Mapping) else None
    command = runtime.get("command") if isinstance(runtime, Mapping) else None
    python_path = str(command[0]) if isinstance(command, list) and command else ""
    if not python_path or not Path(python_path).is_file():
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 관리 Comfy Python 경로 없음: "
            f"python_path={python_path!r}, status={status}"
        )
        return {"ok": False, "error": "managed Comfy Python executable not found"}
    try:
        completed = subprocess.run(
            [python_path, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] pip list 실패: "
                f"returncode={completed.returncode}, stderr={completed.stderr[:2000]}"
            )
            return {
                "ok": False,
                "python": python_path,
                "returncode": completed.returncode,
                "stderr": completed.stderr[:2000],
            }
        packages = json.loads(completed.stdout)
        return {"ok": True, "python": python_path, "packages": packages}
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] pip list 예외: "
            f"python={python_path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return {
            "ok": False,
            "python": python_path,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _workflow_artifact_references(workflow: Mapping[str, Any]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    direct_keys = {
        "unet_name": "unet",
        "vae_name": "vae",
        "ckpt_name": "checkpoint",
        "lora_name": "lora",
        "clip_name": "clip",
        "clip_name1": "clip",
        "clip_name2": "clip",
        "clip_name3": "clip",
        "clip_name4": "clip",
    }
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for key, kind in direct_keys.items():
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                references.append(
                    {"kind": kind, "name": value.strip(), "node_id": str(node_id)}
                )
        for key, value in inputs.items():
            if not str(key).startswith("lora_") or not isinstance(value, Mapping):
                continue
            if value.get("on") is not True:
                continue
            name = str(value.get("lora") or "").strip()
            if name:
                references.append(
                    {"kind": "lora", "name": name, "node_id": str(node_id)}
                )
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for reference in references:
        unique.setdefault((reference["kind"], reference["name"]), reference)
    return list(unique.values())


def _resolve_workflow_artifact(
    comfy_root: Path,
    *,
    kind: str,
    name: str,
) -> list[Path]:
    folders = {
        "unet": ("diffusion_models", "unet"),
        "vae": ("vae",),
        "checkpoint": ("checkpoints",),
        "lora": ("loras",),
        "clip": ("text_encoders", "clip"),
    }.get(kind, ())
    model_root = comfy_root / "models"
    direct: list[Path] = []
    normalized = Path(name.replace("\\", "/"))
    for folder in folders:
        folder_root = (model_root / folder).resolve()
        candidate = (folder_root / normalized).resolve()
        if not candidate.is_relative_to(folder_root):
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] "
                "모델 파일 경로 이탈 무시: "
                f"kind={kind}, name={name!r}, root={folder_root}, candidate={candidate}"
            )
            continue
        if candidate.is_file():
            direct.append(candidate)
    if direct:
        return list(dict.fromkeys(direct))
    matches: list[Path] = []
    for folder in folders:
        root = model_root / folder
        if not root.is_dir():
            continue
        try:
            matches.extend(path.resolve() for path in root.rglob(normalized.name) if path.is_file())
        except Exception as exc:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 모델 파일 탐색 실패: "
                f"root={root}, name={name!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
    return list(dict.fromkeys(matches))


def _workflow_artifacts(
    comfy_root: Path,
    workflow: Mapping[str, Any],
    hash_cache: dict[str, str],
) -> dict[str, Any]:
    canonical = json.dumps(
        workflow,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result: dict[str, Any] = {
        "workflow_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": [],
    }
    for reference in _workflow_artifact_references(workflow):
        matches = _resolve_workflow_artifact(
            comfy_root,
            kind=reference["kind"],
            name=reference["name"],
        )
        entry: dict[str, Any] = {**reference, "matches": []}
        if not matches:
            entry["error"] = "file not found"
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 워크플로 모델 파일 없음: "
                f"reference={reference}"
            )
        for match in matches:
            cache_key = str(match)
            try:
                if cache_key not in hash_cache:
                    hash_cache[cache_key] = _sha256_file(match)
                entry["matches"].append(
                    {
                        "path": str(match),
                        "size": match.stat().st_size,
                        "mtime_ns": match.stat().st_mtime_ns,
                        "sha256": hash_cache[cache_key],
                    }
                )
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] 모델 파일 해시 실패: "
                    f"path={match}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                entry["matches"].append(
                    {"path": str(match), "error": f"{type(exc).__name__}: {exc}"}
                )
        result["files"].append(entry)
    return result


def _probe_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        marker = line.find(_PROBE_LOG_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(_PROBE_LOG_PREFIX) :].strip()
        try:
            event = json.loads(payload)
            if isinstance(event, dict):
                events.append(event)
            else:
                raise TypeError(f"event is {type(event).__name__}, not dict")
        except Exception as exc:
            print(
                "[COMFY_INSTALL][PRODUCTION_IMAGE_DIAGNOSTIC] probe JSON 파싱 실패: "
                f"line={line_number}, payload={payload[:2000]!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            events.append(
                {
                    "event": "probe_parse_error",
                    "line": line_number,
                    "payload": payload[:2000],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return events


def _probe_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    model_before: dict[str, Any] | None = None
    model_after: dict[str, Any] | None = None
    stable_reuse: dict[str, Any] | None = None
    first_nonfinite_stage = None
    errors = []
    for event in events:
        event_name = str(event.get("event") or "")
        stage = str(event.get("stage") or "")
        if event_name == "model_state" and stage == "model_before_sampler":
            model_before = event.get("model") if isinstance(event.get("model"), dict) else {}
        if event_name == "tensor_state" and stage == "latent_after_sampler":
            model_after = (
                event.get("model")
                if isinstance(event.get("model"), dict)
                else {}
            )
        if event_name == "stable_model_reuse":
            stable_reuse = event
        if event_name in {"probe_error", "probe_parse_error"}:
            errors.append(event)
        stats = event.get("stats")
        if isinstance(stats, dict) and stage:
            stages[stage] = {
                "all_finite": stats.get("all_finite"),
                "total_nonfinite": stats.get("total_nonfinite"),
                "tensor_count": stats.get("tensor_count"),
            }
            if int(stats.get("total_nonfinite") or 0) > 0 and first_nonfinite_stage is None:
                first_nonfinite_stage = stage
    seen = set(stages)
    if model_before is not None:
        seen.add("model_before_sampler")
    missing = sorted(_EXPECTED_PROBE_STAGES - seen)
    if not any(stage.startswith("vae_output:") for stage in stages):
        missing.append("vae_output")
    return {
        "event_count": len(events),
        "stages": stages,
        "missing_stages": missing,
        "first_nonfinite_stage": first_nonfinite_stage,
        "probe_errors": errors,
        "model_before_sampler": model_before,
        "model_after_sampler": model_after,
        "stable_model_reuse": stable_reuse,
    }


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
            metrics["neighbor_difference"] >= 10.0
            and metrics["entropy"] >= 7.5
        ):
            reasons.append("블록형/고주파 컬러 노이즈 의심")
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
    first_nonfinite = next(
        (
            run
            for run in runs
            if (run.get("probe_summary") or {}).get("first_nonfinite_stage")
        ),
        None,
    )
    return {
        "name": case.get("name"),
        "dcw_cwm_smc_enabled": case.get("dcw_cwm_smc_enabled"),
        "model_patcher_refresh": case.get("model_patcher_refresh"),
        "stable_model_reuse": case.get("stable_model_reuse"),
        "completed": len(runs),
        "abnormal": len(abnormal),
        "first_abnormal": abnormal[0].get("index") if abnormal else None,
        "first_abnormal_reasons": abnormal[0].get("abnormal_reasons") if abnormal else [],
        "first_nonfinite_run": first_nonfinite.get("index") if first_nonfinite else None,
        "first_nonfinite_stage": (
            (first_nonfinite.get("probe_summary") or {}).get("first_nonfinite_stage")
            if first_nonfinite
            else None
        ),
        "model_reuse": _model_reuse_summary(case),
    }


def _model_reuse_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize warm ModelPatcher transitions without discarding raw events."""
    runs = list(case.get("runs") or [])
    measurement_runs = [run for run in runs if run.get("phase") == "measurement"]

    def models(key: str, selected_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        values = []
        for run in selected_runs:
            summary = run.get("probe_summary") or {}
            value = summary.get(key)
            if isinstance(value, dict) and value:
                values.append(value)
        return values

    def distinct(values: list[Any]) -> list[Any]:
        output = []
        seen = set()
        for value in values:
            if value is None:
                continue
            marker = json.dumps(value, ensure_ascii=True, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            output.append(value)
        return output

    before_models = models("model_before_sampler", measurement_runs)
    after_models = models("model_after_sampler", measurement_runs)
    post_alignment_failures = []
    for run in measurement_runs:
        after = (run.get("probe_summary") or {}).get("model_after_sampler") or {}
        if not after:
            continue
        if after.get("current_weight_patches_uuid") != after.get("patches_uuid"):
            post_alignment_failures.append(run.get("index"))

    transition_matches = 0
    transition_failures = []
    for previous, current in zip(runs, runs[1:]):
        previous_after = (
            (previous.get("probe_summary") or {}).get("model_after_sampler") or {}
        )
        current_before = (
            (current.get("probe_summary") or {}).get("model_before_sampler") or {}
        )
        expected = previous_after.get("patches_uuid")
        observed = current_before.get("current_weight_patches_uuid")
        if expected is None or observed is None:
            continue
        if expected == observed:
            transition_matches += 1
        else:
            transition_failures.append(
                {
                    "previous_run": previous.get("index"),
                    "current_run": current.get("index"),
                    "expected_previous_patch_uuid": expected,
                    "observed_current_weight_uuid": observed,
                }
            )

    stable_events = []
    for run in runs:
        stable = (run.get("probe_summary") or {}).get("stable_model_reuse")
        if isinstance(stable, dict) and stable:
            stable_events.append(stable)
    structure_mismatch_runs = [
        event.get("run_key")
        for event in stable_events
        if event.get("structure_matches") is not True
    ]

    def weight_hashes(values: list[dict[str, Any]]) -> list[str]:
        return distinct(
            [
                (value.get("patched_weight_samples") or {}).get("weights_sha256")
                for value in values
            ]
        )

    return {
        "measurement_runs_with_model_before": len(before_models),
        "measurement_runs_with_model_after": len(after_models),
        "resident_model_ids": distinct(
            [value.get("model_id") for value in before_models + after_models]
        ),
        "before_patcher_ids": distinct(
            [value.get("patcher_id") for value in before_models]
        ),
        "before_patch_uuids": distinct(
            [value.get("patches_uuid") for value in before_models]
        ),
        "patch_structure_hashes": distinct(
            [value.get("patch_structure_sha256") for value in before_models]
        ),
        "before_weight_state_hashes": weight_hashes(before_models),
        "after_weight_state_hashes": weight_hashes(after_models),
        "previous_patch_uuid_transition_matches": transition_matches,
        "previous_patch_uuid_transition_failures": transition_failures,
        "post_sampler_uuid_alignment_failures": post_alignment_failures,
        "stable_event_count": len(stable_events),
        "stable_cache_hits": sum(event.get("cache_hit") is True for event in stable_events),
        "stable_structure_mismatch_runs": structure_mismatch_runs,
        "stable_incoming_patcher_ids": distinct(
            [(event.get("incoming") or {}).get("patcher_id") for event in stable_events]
        ),
        "stable_chosen_patcher_ids": distinct(
            [(event.get("chosen") or {}).get("patcher_id") for event in stable_events]
        ),
        "stable_chosen_patch_uuids": distinct(
            [(event.get("chosen") or {}).get("patches_uuid") for event in stable_events]
        ),
    }


def _conclusions(cases: list[dict[str, Any]]) -> list[str]:
    summaries = {case["name"]: _case_summary(case) for case in cases}
    baseline = summaries.get("production_patch_on", {})
    refreshed = summaries.get("production_model_refresh", {})
    patch_off = summaries.get("production_patch_off", {})
    stable = summaries.get("production_stable_model_reuse", {})
    baseline_bad = int(baseline.get("abnormal") or 0) > 0
    refresh_bad = int(refreshed.get("abnormal") or 0) > 0
    off_bad = int(patch_off.get("abnormal") or 0) > 0
    stable_bad = int(stable.get("abnormal") or 0) > 0
    state_findings = []
    for case in cases:
        state = _model_reuse_summary(case)
        name = str(case.get("name") or "unknown")
        if len(state["after_weight_state_hashes"]) > 1:
            state_findings.append(
                f"{name}: 고정 LoRA 구성인데 sampler 후 resident weight 표본 상태가 "
                f"{len(state['after_weight_state_hashes'])}개로 변했습니다."
            )
        if state["previous_patch_uuid_transition_failures"]:
            state_findings.append(
                f"{name}: 이전 실행 patch UUID와 다음 실행 직전 resident weight UUID가 "
                f"{len(state['previous_patch_uuid_transition_failures'])}회 불일치했습니다."
            )
        if state["post_sampler_uuid_alignment_failures"]:
            state_findings.append(
                f"{name}: sampler 후 요청 patch UUID가 resident weight에 정렬되지 않은 실행이 "
                f"{len(state['post_sampler_uuid_alignment_failures'])}회 있습니다."
            )
        if state["stable_structure_mismatch_runs"]:
            state_findings.append(
                f"{name}: stable reuse 입력의 논리적 patch 구성이 도중에 바뀌었습니다."
            )
    if baseline_bad:
        recovered = []
        if not refresh_bad:
            recovered.append("sampler 직전 Refresh(B)")
        if not off_bad:
            recovered.append("DCW/CWM/SMC OFF(C)")
        if not stable_bad:
            recovered.append("LoRA 완료 ModelPatcher 고정 재사용(D)")
        conclusions = [
            "기존 실제 경로(A)에서 이미지 또는 텐서 이상이 재현되었습니다.",
        ]
        if recovered:
            conclusions.append(
                "A와 동일한 입력에서 안정화된 개입: " + ", ".join(recovered)
            )
        else:
            conclusions.append(
                "Refresh(B), DCW OFF(C), stable reuse(D) 어느 개입도 이상을 회피하지 못했습니다."
            )
        if not stable_bad and refresh_bad:
            conclusions.append(
                "D만 ModelPatcher 반복 재구성을 제거해 안정화되었습니다. 동일 LoRA 구성의 patch 완료 ModelPatcher 재사용을 실제 워크플로에 적용할 근거가 됩니다."
            )
        if not off_bad and refresh_bad:
            conclusions.append(
                "C는 안정적이고 B는 실패했습니다. 단순 wrapper clone보다 DCW/CWM/SMC 경계가 더 강한 원인 후보입니다."
            )
        if not refresh_bad:
            conclusions.append(
                "B가 안정적이므로 sampler 직전 model.clone() 경계가 회피책으로 유효합니다."
            )
        stages = {
            summary.get("first_nonfinite_stage")
            for summary in summaries.values()
            if summary.get("first_nonfinite_stage")
        }
        if stages:
            conclusions.append(
                "계측에서 최초로 확인된 비정상 텐서 단계: "
                + ", ".join(sorted(str(stage) for stage in stages))
            )
        return conclusions + state_findings

    intervention_failures = []
    if refresh_bad:
        intervention_failures.append("Refresh(B)")
    if off_bad:
        intervention_failures.append("DCW OFF(C)")
    if stable_bad:
        intervention_failures.append("stable reuse(D)")
    if intervention_failures:
        return [
            "기존 실제 경로(A)는 정상이지만 다음 개입 케이스에서만 이상이 검출되었습니다: "
            + ", ".join(intervention_failures),
            "해당 개입을 해결책으로 적용하면 안 됩니다. 케이스 로그와 tensor telemetry를 우선 확인해야 합니다.",
        ] + state_findings
    return [
        "A/B/C/D 모두 검정·컬러 노이즈·디코드 실패·NaN/Inf 계측에서 이상이 발견되지 않았습니다.",
        "프롬프트 무시·미완성 전조는 images/의 같은 번호 A/B/C/D 이미지와 실제 prompt JSON을 직접 대조해야 합니다.",
        "이 결과는 사용자가 실제로 쓰는 관리 Comfy와 AssetMode 생성 경로에서 얻었습니다.",
    ] + state_findings


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
        "- 사용하지 않은 것: 독립 E2E Comfy, sampler/VAE 대체 구현, 축소 그래프",
        "- 사용자 선택 LoRA/캐릭터·얼굴·그림체 LoRA/Face ID/Style/Pose/Hires/Detailer: 모두 OFF",
        "- 팩 워크플로에 고정된 모델·LoRA 노드는 실제 배포 경로 보존을 위해 제거하지 않음",
        "- 비교: 동일한 warmup 1개+측정 15개 프롬프트와 동일 seed로 A 실제 경로 / B sampler 직전 Refresh / C DCW OFF / D LoRA 완료 ModelPatcher 고정 재사용",
        "- Refresh는 model.clone() wrapper만 새로 만들며 모델 unload·VRAM 정리·weight patch 삭제를 하지 않음",
        "- D는 LoRA 적용 완료 지점의 첫 ModelPatcher를 같은 케이스 안에서 재사용하고 DCW와 sampler는 매 요청 실행",
        "- 모든 케이스에서 conditioning / sampler 입력·출력 latent / VAE 출력의 NaN·Inf와 ModelPatcher UUID·상주 UUID·patch·backup 상태를 기록",
        "- 자동 판정 범위: 검정/블록형·고주파 컬러 노이즈/디코드 실패/NaN·Inf/계측 누락. 프롬프트 무시·미완성은 이미지와 prompt JSON 직접 대조",
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
            "| 케이스 | DCW/CWM/SMC | Refresh | Stable reuse | 완료 | 이상 | 최초 이상 | 최초 non-finite 단계 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        summary = _case_summary(case)
        lines.append(
            f"| {summary['name']} | {summary['dcw_cwm_smc_enabled']} | "
            f"{summary['model_patcher_refresh']} | "
            f"{summary['stable_model_reuse']} | "
            f"{summary['completed']} | {summary['abnormal']} | "
            f"{summary['first_abnormal'] or '-'} | "
            f"{summary['first_nonfinite_stage'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## ModelPatcher 반복 상태",
            "",
            "고정된 모델·LoRA 구성에서 wrapper UUID가 매회 바뀌더라도 resident weight의 sampler 후 표본 상태와 UUID 전환은 일관되어야 합니다.",
            "",
            "| 케이스 | resident model | wrapper | patch UUID | patch 구조 | sampler 후 weight 상태 | 이전 patch→다음 warm weight | sampler 후 UUID 불일치 | D cache hit | D 선택 patcher |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        state = _model_reuse_summary(case)
        transitions = (
            f"{state['previous_patch_uuid_transition_matches']} 일치 / "
            f"{len(state['previous_patch_uuid_transition_failures'])} 불일치"
        )
        lines.append(
            f"| {case.get('name')} | {len(state['resident_model_ids'])} | "
            f"{len(state['before_patcher_ids'])} | "
            f"{len(state['before_patch_uuids'])} | "
            f"{len(state['patch_structure_hashes'])} | "
            f"{len(state['after_weight_state_hashes'])} | {transitions} | "
            f"{len(state['post_sampler_uuid_alignment_failures'])} | "
            f"{state['stable_cache_hits']} / {state['stable_event_count']} | "
            f"{len(state['stable_chosen_patcher_ids'])} |"
        )
        if state["previous_patch_uuid_transition_failures"]:
            lines.extend(
                [
                    "",
                    f"- `{case.get('name')}` UUID 전환 불일치: `"
                    + json.dumps(
                        state["previous_patch_uuid_transition_failures"],
                        ensure_ascii=False,
                    )
                    + "`",
                ]
            )
        if state["stable_structure_mismatch_runs"]:
            lines.extend(
                [
                    "",
                    f"- `{case.get('name')}` stable patch 구조 불일치 실행: "
                    + ", ".join(
                        f"`{value}`"
                        for value in state["stable_structure_mismatch_runs"]
                    ),
                ]
            )
    for case in cases:
        lines.extend(["", f"### {case['name']}", ""])
        lines.append("| # | phase | seed | 초 | 평균 | 표준편차 | 검정비율 | 인접차 | 최초 non-finite | probe 누락 | 로그 경고 | 판정 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for run in case.get("runs", []):
            metrics = run.get("image_metrics") or {}
            probe = run.get("probe_summary") or {}
            lines.append(
                f"| {run.get('index')} | {run.get('phase')} | {run.get('seed')} | "
                f"{run.get('duration_seconds')} | {metrics.get('mean', '-')} | "
                f"{metrics.get('std', '-')} | {metrics.get('black_fraction', '-')} | "
                f"{metrics.get('neighbor_difference', '-')} | "
                f"{probe.get('first_nonfinite_stage') or '-'} | "
                f"{', '.join(probe.get('missing_stages') or []) or '-'} | "
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
            "- `telemetry/`: 실행별 구조화된 ModelPatcher·conditioning·latent·VAE 계측 JSON",
            "- `workflows/`: 네 케이스 warmup에서 실제 Comfy에 제출한 최종 API 워크플로",
            "- `runs.json`: GPU 전후 상태, 이미지 수치, 유효 개입 값, 구조화 계측 요약",
            "- `environment.json`: 실행 명령·GPU·Python 패키지·custom-node Git revision·모델/LoRA SHA256",
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
    artifact_hash_cache: dict[str, str] = {}
    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "project_head": _safe_git_head(project_root),
        "comfy_head": _safe_git_head(comfy_root),
        "custom_node_revisions": _custom_node_revisions(comfy_root),
        "gpu_at_start": _gpu_snapshot(),
        "production_status_at_start": {},
        "python_packages": {},
        "workflow_artifacts": {},
    }
    started = time.monotonic()
    try:
        initial = production_call({"action": "status"})
        environment["production_status_at_start"] = initial
        environment["python_packages"] = _python_packages(initial)
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
            ("production_patch_on", True, False, False),
            ("production_model_refresh", True, True, False),
            ("production_patch_off", False, False, False),
            ("production_stable_model_reuse", True, False, True),
        )
        total_runs = len(plans) * (1 + len(MEASUREMENT_VARIANTS))
        completed_runs = 0
        for (
            case_name,
            patch_enabled,
            model_patcher_refresh,
            stable_model_reuse,
        ) in plans:
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
                "stable_model_reuse": stable_model_reuse,
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
                        "stable_model_reuse": stable_model_reuse,
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
                    environment["workflow_artifacts"][case_name] = (
                        _workflow_artifacts(
                            comfy_root,
                            workflow_snapshot,
                            artifact_hash_cache,
                        )
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
                probe_events = _probe_events(comfy_log)
                probe_summary = _probe_summary(probe_events)
                _write_json(
                    report_dir / "telemetry" / case_name / f"{run_label}.json",
                    {
                        "case": case_name,
                        "phase": phase,
                        "index": index,
                        "seed": seed,
                        "events": probe_events,
                        "summary": probe_summary,
                    },
                )
                reasons = list(metrics.get("pixel_abnormal_reasons") or [])
                reasons.extend(f"Comfy 로그: {finding}" for finding in findings)
                if probe_summary.get("first_nonfinite_stage"):
                    reasons.append(
                        "최초 non-finite 텐서: "
                        f"{probe_summary['first_nonfinite_stage']}"
                    )
                if probe_summary.get("probe_errors"):
                    reasons.append(
                        f"probe 오류 {len(probe_summary['probe_errors'])}개"
                    )
                if probe_summary.get("missing_stages"):
                    reasons.append(
                        "probe 단계 누락: "
                        + ", ".join(probe_summary["missing_stages"])
                    )
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
                    "probe_events": probe_events,
                    "probe_summary": probe_summary,
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
                    "diagnostic_stable_model_reuse": result.get(
                        "diagnostic_stable_model_reuse"
                    ),
                    "diagnostic_runtime_probes": result.get(
                        "diagnostic_runtime_probes"
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
