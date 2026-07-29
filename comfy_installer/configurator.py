from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class ConfigUpdateError(RuntimeError):
    """설치 완료 후 프로젝트 설정 적용 또는 복원 실패."""


@dataclass(frozen=True)
class ConfigUpdateResult:
    config_path: Path
    backup_path: Path
    before_sha256: str
    after_sha256: str
    updated_keys: tuple[str, ...]


def backup_current_config(
    *,
    config_path: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    reason: str,
) -> dict:
    config_file = Path(config_path).resolve()
    backup_root = Path(backup_dir).resolve()
    safe_reason = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(reason)
    ).strip("_") or "operation"
    try:
        if not config_file.is_file():
            raise ConfigUpdateError(f"백업할 config.json이 없습니다: {config_file}")
        before_hash = _sha256_file(config_file)
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_root / f"config_before_{safe_reason}_{stamp}.json"
        shutil.copy2(config_file, backup_path)
        if _sha256_file(backup_path) != before_hash:
            raise ConfigUpdateError(
                f"config.json 백업 SHA-256 검증에 실패했습니다: {backup_path}"
            )
        print(
            "[COMFY_INSTALL][CONFIG] 설정 백업 완료: "
            f"reason={safe_reason}, path={backup_path}"
        )
        return {
            "config_path": str(config_file),
            "backup_path": str(backup_path),
            "sha256": before_hash,
            "reason": safe_reason,
        }
    except ConfigUpdateError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][CONFIG] 설정 백업 실패: {exc}")
        traceback.print_exc()
        raise ConfigUpdateError(f"설정 백업 실패: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[COMFY_INSTALL][CONFIG] {label} JSON 읽기 실패: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise ConfigUpdateError(f"{label} JSON을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigUpdateError(f"{label} 최상위 값이 JSON 객체가 아닙니다: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2) + os.linesep
    ).encode("utf-8")
    part_path = path.with_name(f"{path.name}.part")
    try:
        with part_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part_path, path)
    except Exception:
        if part_path.exists():
            print(
                "[COMFY_INSTALL][CONFIG] 원자적 설정 쓰기 실패 후 part 파일 보존: "
                f"{part_path}"
            )
        raise


def _copy_file_atomic(source: Path, destination: Path) -> None:
    part_path = destination.with_name(f"{destination.name}.part")
    try:
        with source.open("rb") as input_stream, part_path.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(part_path, destination)
    except Exception:
        if part_path.exists():
            print(
                "[COMFY_INSTALL][CONFIG] 원자적 설정 복사 실패 후 part 파일 보존: "
                f"{part_path}"
            )
        raise


def _set_dotted(config: dict, dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise ConfigUpdateError(f"설정 키 형식이 잘못되었습니다: {dotted_key!r}")
    cursor = config
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ConfigUpdateError(
                f"중첩 설정 대상이 객체가 아닙니다: key={dotted_key!r}, part={part!r}"
            )
        cursor = existing
    cursor[parts[-1]] = value


def apply_installed_config(
    *,
    config_path: str | os.PathLike[str],
    requirements_dir: str | os.PathLike[str],
    comfy_root: str | os.PathLike[str],
    workflow_bindings: Mapping[str, str],
    required_bindings: Iterable[str],
    comfy_port: int = 8188,
) -> ConfigUpdateResult:
    config_file = Path(config_path).resolve()
    backup_root = Path(requirements_dir).resolve()
    install_root = Path(comfy_root).resolve()
    try:
        if not config_file.is_file():
            raise ConfigUpdateError(f"수정할 config.json이 없습니다: {config_file}")
        if not install_root.is_dir():
            raise ConfigUpdateError(
                f"설치 완료된 ComfyUI 폴더가 없습니다: {install_root}"
            )
        required = tuple(str(key) for key in required_bindings)
        missing = [key for key in required if not workflow_bindings.get(key)]
        if missing:
            raise ConfigUpdateError(
                "워크플로우 바인딩이 누락되었습니다: " + ", ".join(missing)
            )

        normalized_bindings: dict[str, str] = {}
        workflow_root = (install_root / "user" / "default" / "workflows").resolve()
        for key in required:
            workflow_path = Path(workflow_bindings[key]).resolve()
            try:
                workflow_path.relative_to(workflow_root)
            except ValueError as exc:
                raise ConfigUpdateError(
                    "워크플로우 경로가 설치 폴더 밖을 가리킵니다: "
                    f"key={key!r}, path={workflow_path}"
                ) from exc
            if not workflow_path.is_file():
                raise ConfigUpdateError(
                    f"설치된 워크플로우 파일이 없습니다: key={key!r}, "
                    f"path={workflow_path}"
                )
            normalized_bindings[key] = str(workflow_path)

        before_hash = _sha256_file(config_file)
        config = _read_json_object(config_file, "현재 설정")
        updated = copy.deepcopy(config)
        for key, value in normalized_bindings.items():
            _set_dotted(updated, key, value)

        lora_base = install_root / "models" / "loras" / "SOYA_CHAR_LORA"
        direct_updates = {
            "comfyui_port": int(comfy_port),
            "comfy_input_dir": str(install_root / "input"),
            "lora_load_path": str(lora_base),
            "bot_lora_load_path": str(lora_base / "SOYA_BOT_LORA"),
            "instance_lora_load_path": str(lora_base / "SOYA_INSTANCE_LORA"),
            "style_lora_load_path": str(lora_base / "SOYA_STYLE_LORA"),
            "outfit_mode_enabled": False,
            "outfit_workflow_source_path": "",
        }
        updated.update(direct_updates)

        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_root / f"config_before_comfy_install_{stamp}.json"
        shutil.copy2(config_file, backup_path)
        if _sha256_file(backup_path) != before_hash:
            raise ConfigUpdateError(
                f"config.json 백업 SHA-256 검증에 실패했습니다: {backup_path}"
            )
        print(f"[COMFY_INSTALL][CONFIG] 설정 백업 완료: {backup_path}")

        _write_json_atomic(config_file, updated)
        reloaded = _read_json_object(config_file, "갱신 설정")
        if reloaded != updated:
            raise ConfigUpdateError("config.json 저장 후 재검증 값이 일치하지 않습니다.")
        after_hash = _sha256_file(config_file)
        all_updated_keys = tuple(
            sorted(set(normalized_bindings) | set(direct_updates))
        )
        print(
            "[COMFY_INSTALL][CONFIG] 설치 경로 설정 적용 완료: "
            f"backup={backup_path}, updated={len(all_updated_keys)}"
        )
        return ConfigUpdateResult(
            config_path=config_file,
            backup_path=backup_path,
            before_sha256=before_hash,
            after_sha256=after_hash,
            updated_keys=all_updated_keys,
        )
    except ConfigUpdateError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][CONFIG] 설정 적용 실패: {exc}")
        traceback.print_exc()
        raise ConfigUpdateError(f"설정 적용 실패: {exc}") from exc


def restore_config_backup(
    *,
    config_path: str | os.PathLike[str],
    requirements_dir: str | os.PathLike[str],
    backup_path: str | os.PathLike[str],
) -> dict:
    config_file = Path(config_path).resolve()
    backup_root = Path(requirements_dir).resolve()
    source = Path(backup_path).resolve()
    try:
        try:
            source.relative_to(backup_root)
        except ValueError as exc:
            raise ConfigUpdateError(
                f"설정 백업 폴더 밖의 파일은 복원할 수 없습니다: {source}"
            ) from exc
        if not source.is_file() or not source.name.startswith(
            "config_before_comfy_install_"
        ):
            raise ConfigUpdateError(f"유효한 Comfy 설치 설정 백업이 아닙니다: {source}")
        backup_config = _read_json_object(source, "설정 백업")

        if config_file.is_file():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safety_backup = (
                backup_root / f"config_before_comfy_restore_{stamp}.json"
            )
            shutil.copy2(config_file, safety_backup)
            print(
                "[COMFY_INSTALL][CONFIG] 복원 직전 현재 설정 백업 완료: "
                f"{safety_backup}"
            )
        else:
            safety_backup = None
            print(
                "[COMFY_INSTALL][CONFIG] 복원 대상 config.json이 없어 "
                "복원 직전 백업을 건너뜁니다."
            )

        _copy_file_atomic(source, config_file)
        restored_hash = _sha256_file(config_file)
        expected_hash = _sha256_file(source)
        if restored_hash != expected_hash:
            raise ConfigUpdateError("복원한 config.json SHA-256이 백업과 다릅니다.")
        print(
            "[COMFY_INSTALL][CONFIG] 원래 설정 복원 완료: "
            f"source={source}, target={config_file}"
        )
        return {
            "config_path": str(config_file),
            "source_backup": str(source),
            "safety_backup": str(safety_backup) if safety_backup else None,
            "sha256": restored_hash,
        }
    except ConfigUpdateError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][CONFIG] 설정 복원 실패: {exc}")
        traceback.print_exc()
        raise ConfigUpdateError(f"설정 복원 실패: {exc}") from exc
