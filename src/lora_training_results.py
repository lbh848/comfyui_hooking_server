"""Read LoRA results in the application's flat layout and the native MOSS layout."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re


_FLAT_STEP = re.compile(r"(.+)-step(\d+)")
_MOSS_STEP = re.compile(r"(.+)_s(\d+)")


def _moss_step_name(path: Path) -> str | None:
    match = _MOSS_STEP.fullmatch(path.stem)
    if match and match[1] == path.parent.name:
        return f"{match[1]}-step{int(match[2]):08d}"
    return None


def _checkpoints(session_dir: Path) -> dict[str, Path]:
    """Use the same checkpoint names before and after the trainer moves the files."""
    if not session_dir.is_dir():
        return {}
    checkpoints = {}
    for run_dir in sorted(session_dir.iterdir()):
        if run_dir.is_symlink() or run_dir.is_junction() or not run_dir.is_dir():
            continue
        for path in sorted(run_dir.glob("*_s*.safetensors")):
            name = _moss_step_name(path)
            if name and path.is_file():
                checkpoints[name] = path
    for path in sorted(session_dir.glob("*.safetensors")):
        if path.is_file():
            checkpoints[path.stem] = path
    return checkpoints


def lora_session_summary(session_dir: str) -> dict:
    checkpoints = _checkpoints(Path(session_dir))
    return {
        "step_count": len(checkpoints),
        "has_final": any(not _FLAT_STEP.fullmatch(name) for name in checkpoints),
    }


def _read_metadata(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[LORA_TRAINED] JSON 읽기 실패: {path} - {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _moss_report(metadata: dict) -> bool:
    return isinstance(metadata.get("request"), dict) and type(metadata.get("steps")) is int


def _moss_step_losses(metadata: dict) -> dict[int, float]:
    losses = metadata.get("losses")
    if not isinstance(losses, list):
        return {}
    return {
        row["step"]: row["loss"] for row in losses
        if isinstance(row, dict) and type(row.get("step")) is int
        and type(row.get("loss")) in (int, float) and math.isfinite(row["loss"])
    }


def list_lora_checkpoints(session_dir: str) -> list[dict]:
    directory = Path(session_dir)
    inventory = _checkpoints(directory)
    metadata = {name: _read_metadata(directory / f"{name}.json") for name in inventory}
    # The final file can be deleted while intermediate files still use its report.
    run_names = {
        match[1] for name in inventory if (match := _FLAT_STEP.fullmatch(name))
    }
    for run_name in run_names - metadata.keys():
        metadata[run_name] = _read_metadata(directory / f"{run_name}.json")
    reports = {
        name: (data["steps"], _moss_step_losses(data))
        for name, data in metadata.items() if _moss_report(data)
    }
    results = []
    for name, path in sorted(inventory.items()):
        match = _FLAT_STEP.fullmatch(name)
        run_name = match[1] if match else name
        report = reports.get(run_name)
        step = int(match[2]) if match else report[0] if report else None
        info = metadata[name]
        previews = info.get("previews")
        json_file = f"{name}.json" if info and not _moss_report(info) else None
        toml_file = f"{name}.toml" if (directory / f"{name}.toml").is_file() else None
        results.append({
            "name": name,
            "safetensors": path.relative_to(directory).as_posix(),
            "previews": [preview for preview in previews if isinstance(preview, str)] if isinstance(previews, list) else [],
            "json_file": json_file,
            "toml_file": toml_file,
            "moss_report": f"{run_name}.json" if report else None,
            "avr_loss": info.get("avr_loss"),
            "loss": report[1].get(step) if report else None,
            "step": step,
            "is_final": match is None,
            "config_format": "toml" if toml_file else "json" if report else None,
        })
    return results


def read_lora_checkpoint_config(session_dir: str, step_name: str) -> dict:
    directory = Path(session_dir)
    step = next((row for row in list_lora_checkpoints(session_dir) if row["name"] == step_name), None)
    try:
        if step and step["toml_file"]:
            filename = step["toml_file"]
            return {"success": True, "content": (directory / filename).read_text(encoding="utf-8"),
                    "filename": filename, "format": "toml"}
        if step and step["moss_report"]:
            filename = step["moss_report"]
            metadata = _read_metadata(directory / filename)
            if _moss_report(metadata):
                return {"success": True, "content": json.dumps(metadata["request"], ensure_ascii=False, indent=2),
                        "filename": filename, "format": "json"}
    except (OSError, ValueError) as exc:
        return {"success": False, "error": str(exc)}
    return {"success": False, "error": "저장된 학습 설정이 없습니다"}


def _delete_lora_checkpoints(session_dir: str, steps: list[dict], keep: list[dict]) -> dict:
    directory = Path(session_dir).resolve()
    deleted = []
    errors = []
    retained = list(keep)
    removed = []

    def delete(filename: str) -> bool:
        path = directory / filename
        try:
            if not path.resolve().is_relative_to(directory):
                raise ValueError("세션 폴더 밖의 파일입니다")
            path.unlink()
            deleted.append(filename)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
            return False
        return True

    for step in steps:
        if delete(step["safetensors"]):
            removed.append(step)
        else:
            retained.append(step)
    kept_files = {filename for step in retained for filename in _checkpoint_sidecars(step)}
    sidecars = dict.fromkeys(filename for step in removed for filename in _checkpoint_sidecars(step))
    for filename in sidecars:
        if filename not in kept_files:
            delete(filename)
    return {"success": True, "deleted": deleted, "errors": errors}


def _checkpoint_sidecars(step: dict) -> list[str]:
    files = list(step["previews"])
    files.extend(step[key] for key in ("toml_file", "json_file", "moss_report") if step[key])
    return files


def delete_lora_checkpoint(session_dir: str, step_name: str) -> dict:
    steps = list_lora_checkpoints(session_dir)
    step = next((row for row in steps if row["name"] == step_name), None)
    if not step:
        return {"success": False, "error": "LoRA 파일을 찾을 수 없습니다"}
    result = _delete_lora_checkpoints(session_dir, [step], [row for row in steps if row is not step])
    return {**result, "safetensors": step["safetensors"]}


def delete_other_lora_checkpoints(session_dir: str, representative: str) -> dict:
    steps = list_lora_checkpoints(session_dir)
    moved_step_name = _moss_step_name(Path(representative))
    keep = next((step for step in steps if step["safetensors"] == representative or step["name"] == moved_step_name), None)
    if representative and keep is None:
        return {"deleted_steps": [], "kept": None, "errors": ["세션 대표 LoRA 파일을 찾을 수 없습니다"]}
    remove = [step for step in steps if step is not keep]
    result = _delete_lora_checkpoints(session_dir, remove, [keep] if keep else [])
    deleted = set(result["deleted"])
    return {
        "deleted_steps": [step["safetensors"] for step in remove if step["safetensors"] in deleted],
        "kept": keep["safetensors"] if keep else None,
        "errors": result["errors"],
    }
