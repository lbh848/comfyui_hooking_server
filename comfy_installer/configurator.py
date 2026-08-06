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
from typing import Any, Iterable, Mapping


class ConfigUpdateError(RuntimeError):
    """설치 완료 후 프로젝트 설정 적용 또는 복원 실패."""


@dataclass(frozen=True)
class ConfigUpdateResult:
    config_path: Path
    backup_path: Path
    before_sha256: str
    after_sha256: str
    updated_keys: tuple[str, ...]


@dataclass(frozen=True)
class ConfigRetargetResult:
    config_path: Path
    backup_path: Path
    before_sha256: str
    after_sha256: str
    updated_paths: tuple[str, ...]
    missing_targets: tuple[tuple[str, str], ...]
    already_retargeted: bool


_EMBEDDED_DIRECT_PATHS: dict[str, tuple[str, ...]] = {
    "$.comfy_input_dir": ("input",),
    "$.lora_load_path": ("models", "loras", "SOYA_CHAR_LORA"),
    "$.bot_lora_load_path": (
        "models",
        "loras",
        "SOYA_CHAR_LORA",
        "SOYA_BOT_LORA",
    ),
    "$.instance_lora_load_path": (
        "models",
        "loras",
        "SOYA_CHAR_LORA",
        "SOYA_INSTANCE_LORA",
    ),
    "$.style_lora_load_path": (
        "models",
        "loras",
        "SOYA_CHAR_LORA",
        "SOYA_STYLE_LORA",
    ),
}
_WORKFLOW_RELATIVE_ROOT = ("user", "default", "workflows")


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
        if existing is None or (
            isinstance(existing, str) and not existing.strip()
        ):
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ConfigUpdateError(
                f"중첩 설정 대상이 객체가 아닙니다: key={dotted_key!r}, part={part!r}"
            )
        cursor = existing
    cursor[parts[-1]] = value


def _normalize_embedded_workflow_bindings(
    workflow_bindings: Mapping[str, str],
    *,
    comfy_root: Path,
    label: str,
) -> dict[str, str]:
    workflow_root = (comfy_root / "user" / "default" / "workflows").resolve()
    normalized: dict[str, str] = {}
    for raw_key, raw_value in workflow_bindings.items():
        key = str(raw_key).strip()
        if not key:
            print(
                f"[COMFY_INSTALL][CONFIG] {label} 바인딩 키가 비어 있습니다: "
                f"key={raw_key!r}, value={raw_value!r}"
            )
            raise ConfigUpdateError(f"{label} 바인딩 키가 비어 있습니다.")
        try:
            workflow_path = Path(raw_value).resolve()
        except (OSError, RuntimeError, TypeError) as exc:
            print(
                f"[COMFY_INSTALL][CONFIG] {label} 경로 해석 실패: "
                f"key={key!r}, value={raw_value!r}, error={exc}"
            )
            traceback.print_exc()
            raise ConfigUpdateError(
                f"{label} 경로를 해석할 수 없습니다: key={key!r}"
            ) from exc
        try:
            workflow_path.relative_to(workflow_root)
        except ValueError as exc:
            print(
                f"[COMFY_INSTALL][CONFIG] {label} 경로가 내장 Comfy "
                f"워크플로우 폴더 밖입니다: key={key!r}, path={workflow_path}"
            )
            raise ConfigUpdateError(
                f"{label} 경로가 설치 폴더 밖을 가리킵니다: "
                f"key={key!r}, path={workflow_path}"
            ) from exc
        if not workflow_path.is_file():
            print(
                f"[COMFY_INSTALL][CONFIG] {label} 파일이 없습니다: "
                f"key={key!r}, path={workflow_path}"
            )
            raise ConfigUpdateError(
                f"{label} 파일이 없습니다: key={key!r}, path={workflow_path}"
            )
        normalized[key] = str(workflow_path)
    return normalized


def _normalize_embedded_workflow_base_dir(
    value: str | os.PathLike[str],
    *,
    comfy_root: Path,
) -> str:
    workflow_root = (comfy_root / "user" / "default" / "workflows").resolve()
    try:
        base_dir = Path(value).resolve()
        base_dir.relative_to(workflow_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            "[COMFY_INSTALL][CONFIG] 내장 워크플로우 베이스 경로 검증 실패: "
            f"value={value!r}, workflow_root={workflow_root}, error={exc}"
        )
        traceback.print_exc()
        raise ConfigUpdateError(
            f"내장 워크플로우 베이스 경로가 잘못되었습니다: {value!r}"
        ) from exc
    if not base_dir.is_dir():
        print(
            "[COMFY_INSTALL][CONFIG] 내장 워크플로우 베이스 폴더가 "
            f"없습니다: {base_dir}"
        )
        raise ConfigUpdateError(
            f"내장 워크플로우 베이스 폴더가 없습니다: {base_dir}"
        )
    return str(base_dir)


def _apply_workflow_bindings(
    config: dict,
    workflow_bindings: Mapping[str, str],
) -> list[str]:
    changed: list[str] = []
    for key, value in workflow_bindings.items():
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current != value:
            changed.append(key)
        _set_dotted(config, key, value)
        print(
            "[COMFY_INSTALL][CONFIG] 내장 워크플로우 경로 강제 설정: "
            f"key={key}, path={value}"
        )
    return changed


def _json_child_path(parent: str, key: str) -> str:
    if key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def retarget_legacy_workflow_paths(
    *,
    config_path: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    legacy_user_root: str | os.PathLike[str],
    user_root: str | os.PathLike[str],
    path_map: Mapping[str, str] | None = None,
) -> dict:
    """백업을 만든 뒤 레거시 사용자 워크플로우 경로를 ASCII 경로로 바꾼다."""
    config_file = Path(config_path).resolve()
    legacy_root = Path(legacy_user_root).resolve()
    target_root = Path(user_root).resolve()
    normalized_map: dict[str, Path] = {}
    try:
        if not config_file.is_file():
            print(
                "[COMFY_INSTALL][CONFIG][WORKFLOW_ASCII] config.json이 없어 "
                f"경로 전환을 건너뜁니다: {config_file}"
            )
            return {
                "updated": False,
                "config_path": str(config_file),
                "backup_path": None,
                "updated_paths": [],
            }

        for source, destination in (path_map or {}).items():
            source_path = Path(source).resolve()
            destination_path = Path(destination).resolve()
            normalized_map[os.path.normcase(str(source_path))] = destination_path

        config = _read_json_object(config_file, "현재 설정")
        updated_paths: list[str] = []

        def retarget(value: Any, json_path: str) -> Any:
            if isinstance(value, dict):
                return {
                    key: retarget(
                        child,
                        _json_child_path(json_path, str(key)),
                    )
                    for key, child in value.items()
                }
            if isinstance(value, list):
                return [
                    retarget(child, f"{json_path}[{index}]")
                    for index, child in enumerate(value)
                ]
            if not isinstance(value, str) or not value.strip():
                return value

            try:
                candidate = Path(value.strip())
                if not candidate.is_absolute():
                    return value
                resolved = candidate.resolve()
                relative = resolved.relative_to(legacy_root)
            except ValueError:
                return value
            except (OSError, RuntimeError) as exc:
                print(
                    "[COMFY_INSTALL][CONFIG][WORKFLOW_ASCII] 레거시 경로 "
                    "해석 실패; 원래 값을 유지합니다: "
                    f"setting={json_path}, value={value!r}, error={exc}"
                )
                traceback.print_exc()
                return value

            mapped = normalized_map.get(os.path.normcase(str(resolved)))
            target = (
                mapped
                if mapped is not None
                else (target_root / relative).resolve()
            )
            if not target.exists():
                print(
                    "[COMFY_INSTALL][CONFIG][WORKFLOW_ASCII] 복사된 대상이 없어 "
                    "설정 경로를 유지합니다: "
                    f"setting={json_path}, source={resolved}, target={target}"
                )
                return value
            updated_paths.append(json_path)
            return str(target)

        updated = retarget(config, "$")
        if not updated_paths:
            return {
                "updated": False,
                "config_path": str(config_file),
                "backup_path": None,
                "updated_paths": [],
            }

        backup = backup_current_config(
            config_path=config_file,
            backup_dir=backup_dir,
            reason="workflow_ascii_migration",
        )
        if _sha256_file(config_file) != backup["sha256"]:
            raise ConfigUpdateError(
                "워크플로우 ASCII 경로 전환 전 config.json이 변경되었습니다."
            )

        _write_json_atomic(config_file, updated)
        reloaded = _read_json_object(
            config_file,
            "워크플로우 ASCII 경로 전환 설정",
        )
        if reloaded != updated:
            raise ConfigUpdateError(
                "워크플로우 ASCII 경로 전환 후 config.json 재검증이 "
                "일치하지 않습니다."
            )
        print(
            "[COMFY_INSTALL][CONFIG][WORKFLOW_ASCII] 설정 경로 전환 완료: "
            f"backup={backup['backup_path']}, updated={len(updated_paths)}"
        )
        return {
            "updated": True,
            "config_path": str(config_file),
            "backup_path": str(backup["backup_path"]),
            "updated_paths": updated_paths,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][CONFIG][WORKFLOW_ASCII] 설정 경로 전환 실패: "
            f"{exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ConfigUpdateError):
            raise
        raise ConfigUpdateError(
            f"워크플로우 ASCII 설정 경로 전환 실패: {exc}"
        ) from exc


def _is_workflow_source_key(key: str) -> bool:
    return (
        key == "comfy_workflow_source_path"
        or key.endswith("_workflow_source_path")
        or key.endswith("_workflow_source_paths")
    )


def _relative_from_anchor(
    candidate: Path, anchor_parts: tuple[str, ...]
) -> Path | None:
    parts = candidate.parts
    folded = tuple(part.casefold() for part in parts)
    anchor = tuple(part.casefold() for part in anchor_parts)
    limit = len(parts) - len(anchor) + 1
    for index in range(max(0, limit)):
        if folded[index : index + len(anchor)] == anchor:
            return Path(*parts[index:])
    return None


def _record_retarget(
    *,
    json_path: str,
    target: Path,
    updated_paths: list[str],
    missing_targets: list[tuple[str, str]],
) -> str:
    updated_paths.append(json_path)
    if not target.exists():
        missing_targets.append((json_path, str(target)))
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 대상 경로 없음; "
            f"설정 경로는 전환합니다: setting={json_path}, target={target}"
        )
    return str(target)


def _retarget_descendant_paths(
    value: Any,
    *,
    json_path: str,
    old_root: Path | None,
    new_root: Path,
    updated_paths: list[str],
    missing_targets: list[tuple[str, str]],
    workflow_source: bool = False,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _retarget_descendant_paths(
                child,
                json_path=_json_child_path(json_path, str(key)),
                old_root=old_root,
                new_root=new_root,
                updated_paths=updated_paths,
                missing_targets=missing_targets,
                workflow_source=(
                    workflow_source or _is_workflow_source_key(str(key))
                ),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _retarget_descendant_paths(
                child,
                json_path=f"{json_path}[{index}]",
                old_root=old_root,
                new_root=new_root,
                updated_paths=updated_paths,
                missing_targets=missing_targets,
                workflow_source=workflow_source,
            )
            for index, child in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    direct_parts = _EMBEDDED_DIRECT_PATHS.get(json_path)
    if direct_parts is not None:
        try:
            target = (new_root.joinpath(*direct_parts)).resolve()
            candidate = Path(value.strip()).resolve() if value.strip() else None
        except (OSError, RuntimeError) as exc:
            print(
                "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 직접 경로 해석 실패; "
                f"원래 값을 유지합니다: setting={json_path}, error={exc}"
            )
            traceback.print_exc()
            return value
        if candidate == target:
            return value
        return _record_retarget(
            json_path=json_path,
            target=target,
            updated_paths=updated_paths,
            missing_targets=missing_targets,
        )

    if not value.strip():
        return value

    try:
        candidate = Path(value.strip())
        if not candidate.is_absolute():
            return value
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 설정 경로 해석 실패; "
            f"원래 값을 유지합니다: setting={json_path}, error={exc}"
        )
        traceback.print_exc()
        return value

    try:
        resolved.relative_to(new_root)
        return value
    except ValueError:
        pass

    if old_root is not None:
        try:
            relative = resolved.relative_to(old_root)
            target = (new_root / relative).resolve()
            return _record_retarget(
                json_path=json_path,
                target=target,
                updated_paths=updated_paths,
                missing_targets=missing_targets,
            )
        except ValueError:
            pass
        except (OSError, RuntimeError) as exc:
            print(
                "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 이전 Comfy 경로 "
                f"변환 실패: setting={json_path}, error={exc}"
            )
            traceback.print_exc()
            return value

    if not workflow_source:
        return value

    try:
        # mode_workflow는 현재 프로젝트가 직접 관리하는 워크플로우이므로
        # ComfyUI 외부에 있어도 의도된 경로로 유지한다.
        resolved.relative_to((new_root.parent / "mode_workflow").resolve())
        return value
    except ValueError:
        pass

    try:
        relative = _relative_from_anchor(resolved, _WORKFLOW_RELATIVE_ROOT)
        if relative is None:
            relative = Path(*_WORKFLOW_RELATIVE_ROOT, resolved.name)
        target = (new_root / relative).resolve()
    except (OSError, RuntimeError) as exc:
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 워크플로우 대상 경로 "
            "해석 실패; "
            f"원래 값을 유지합니다: setting={json_path}, error={exc}"
        )
        traceback.print_exc()
        return value
    return _record_retarget(
        json_path=json_path,
        target=target,
        updated_paths=updated_paths,
        missing_targets=missing_targets,
    )


def _collect_descendant_paths(
    value: Any,
    *,
    json_path: str,
    root: Path,
    matches: list[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_descendant_paths(
                child,
                json_path=_json_child_path(json_path, str(key)),
                root=root,
                matches=matches,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _collect_descendant_paths(
                child,
                json_path=f"{json_path}[{index}]",
                root=root,
                matches=matches,
            )
        return
    if not isinstance(value, str) or not value.strip():
        return

    try:
        candidate = Path(value.strip())
        if not candidate.is_absolute():
            return
        candidate.resolve().relative_to(root)
    except ValueError:
        return
    except (OSError, RuntimeError) as exc:
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 내장 경로 확인 실패; "
            f"setting={json_path}, value={value!r}, error={exc}"
        )
        traceback.print_exc()
        return
    matches.append(json_path)


def retarget_config_to_embedded_comfy(
    *,
    config_path: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    backup_path: str | os.PathLike[str],
    old_comfy_root: str | os.PathLike[str] | None,
    new_comfy_root: str | os.PathLike[str],
    default_workflow_bindings: Mapping[str, str] | None = None,
    workflow_base_dir: str | os.PathLike[str] | None = None,
) -> ConfigRetargetResult:
    config_file = Path(config_path).resolve()
    backup_root = Path(backup_dir).resolve()
    backup_file = Path(backup_path).resolve()
    old_root = (
        Path(old_comfy_root).resolve()
        if old_comfy_root is not None
        else None
    )
    new_root = Path(new_comfy_root).resolve()
    try:
        if not config_file.is_file():
            raise ConfigUpdateError(f"수정할 config.json이 없습니다: {config_file}")
        if old_root is not None and not old_root.is_dir():
            raise ConfigUpdateError(f"기존 ComfyUI 폴더가 없습니다: {old_root}")
        if old_root is not None and old_root == new_root:
            raise ConfigUpdateError("기존 ComfyUI와 내장 ComfyUI 경로가 같습니다.")
        if not new_root.is_dir() or not (new_root / ".git").is_dir():
            raise ConfigUpdateError(
                "경로를 전환할 내장 ComfyUI가 설치되지 않았습니다: "
                f"{new_root}"
            )

        try:
            backup_file.relative_to(backup_root)
        except ValueError as exc:
            raise ConfigUpdateError(
                f"설치기 백업 폴더 밖의 설정 백업은 사용할 수 없습니다: {backup_file}"
            ) from exc
        expected_prefix = (
            "config_before_comfy_v4_migrate_"
            if old_root is not None
            else "config_before_comfy_embedded_retarget_"
        )
        if not backup_file.is_file() or not backup_file.name.startswith(
            expected_prefix
        ):
            raise ConfigUpdateError(
                f"유효한 내장 Comfy 전환 설정 백업이 아닙니다: {backup_file}"
            )

        before_hash = _sha256_file(config_file)
        backup_hash = _sha256_file(backup_file)
        if backup_hash != before_hash:
            raise ConfigUpdateError(
                "내장 Comfy 전환 설정 백업과 현재 config.json이 다릅니다. "
                "동시 설정 변경 가능성이 있어 덮어쓰지 않습니다."
            )

        config = _read_json_object(config_file, "현재 설정")
        updated_paths: list[str] = []
        missing_targets: list[tuple[str, str]] = []
        seeded = copy.deepcopy(config)
        if workflow_base_dir is not None:
            normalized_base_dir = _normalize_embedded_workflow_base_dir(
                workflow_base_dir,
                comfy_root=new_root,
            )
            if seeded.get("workflow_base_dir") != normalized_base_dir:
                updated_paths.append("$.workflow_base_dir")
            seeded["workflow_base_dir"] = normalized_base_dir
            print(
                "[COMFY_INSTALL][CONFIG] 내장 워크플로우 베이스 경로 강제 설정: "
                f"path={normalized_base_dir}"
            )
        if default_workflow_bindings:
            normalized_defaults = _normalize_embedded_workflow_bindings(
                default_workflow_bindings,
                comfy_root=new_root,
                label="기본 워크플로우",
            )
            defaulted_keys = _apply_workflow_bindings(
                seeded,
                normalized_defaults,
            )
            updated_paths.extend(f"$.{key}" for key in defaulted_keys)
        disabled_outfit_paths: list[str] = []
        for key, value in (
            ("outfit_mode_enabled", False),
            ("outfit_workflow_source_path", ""),
        ):
            if key not in seeded or seeded.get(key) == value:
                continue
            seeded[key] = value
            json_path = f"$.{key}"
            updated_paths.append(json_path)
            disabled_outfit_paths.append(json_path)
        if disabled_outfit_paths:
            print(
                "[COMFY_INSTALL][CONFIG] 내장 배포 워크플로우가 없는 복장 추출 "
                "설정을 비활성화합니다: "
                f"updated={disabled_outfit_paths}"
            )
        updated = _retarget_descendant_paths(
            seeded,
            json_path="$",
            old_root=old_root,
            new_root=new_root,
            updated_paths=updated_paths,
            missing_targets=missing_targets,
        )
        if not updated_paths:
            embedded_paths: list[str] = []
            _collect_descendant_paths(
                config,
                json_path="$",
                root=new_root,
                matches=embedded_paths,
            )
            if embedded_paths:
                print(
                    "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 이미 내장 Comfy로 "
                    "전환된 설정 확인: "
                    f"new_root={new_root}, matched={len(embedded_paths)}"
                )
                return ConfigRetargetResult(
                    config_path=config_file,
                    backup_path=backup_file,
                    before_sha256=before_hash,
                    after_sha256=before_hash,
                    updated_paths=(),
                    missing_targets=(),
                    already_retargeted=True,
                )
            print(
                "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 전환할 설정 경로 없음: "
                f"old_root={old_root}, new_root={new_root}"
            )
            raise ConfigUpdateError(
                "config.json에서 내장 Comfy로 전환할 설정 경로를 "
                f"찾지 못했습니다: old_root={old_root}"
            )

        _write_json_atomic(config_file, updated)
        reloaded = _read_json_object(config_file, "내장 Comfy 갱신 설정")
        if reloaded != updated:
            raise ConfigUpdateError(
                "내장 Comfy config.json 저장 후 재검증 값이 일치하지 않습니다."
            )
        after_hash = _sha256_file(config_file)
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 내장 Comfy 경로 전환 완료: "
            f"backup={backup_file}, updated={len(updated_paths)}, "
            f"missing={len(missing_targets)}"
        )
        return ConfigRetargetResult(
            config_path=config_file,
            backup_path=backup_file,
            before_sha256=before_hash,
            after_sha256=after_hash,
            updated_paths=tuple(updated_paths),
            missing_targets=tuple(missing_targets),
            already_retargeted=False,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][CONFIG][EMBEDDED_RETARGET] 설정 경로 전환 실패: "
            f"{exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ConfigUpdateError):
            raise
        raise ConfigUpdateError(f"내장 Comfy 설정 경로 전환 실패: {exc}") from exc


def apply_installed_config(
    *,
    config_path: str | os.PathLike[str],
    requirements_dir: str | os.PathLike[str],
    comfy_root: str | os.PathLike[str],
    workflow_bindings: Mapping[str, str],
    required_bindings: Iterable[str],
    default_workflow_bindings: Mapping[str, str] | None = None,
    workflow_base_dir: str | os.PathLike[str] | None = None,
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

        normalized_bindings = _normalize_embedded_workflow_bindings(
            {key: workflow_bindings[key] for key in required},
            comfy_root=install_root,
            label="설치된 워크플로우",
        )
        normalized_defaults = _normalize_embedded_workflow_bindings(
            default_workflow_bindings or {},
            comfy_root=install_root,
            label="기본 워크플로우",
        )
        normalized_base_dir = (
            _normalize_embedded_workflow_base_dir(
                workflow_base_dir,
                comfy_root=install_root,
            )
            if workflow_base_dir is not None
            else None
        )

        before_hash = _sha256_file(config_file)
        config = _read_json_object(config_file, "현재 설정")
        updated = copy.deepcopy(config)
        for key, value in normalized_bindings.items():
            _set_dotted(updated, key, value)
        defaulted_keys = _apply_workflow_bindings(
            updated,
            normalized_defaults,
        )

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
        if normalized_base_dir is not None:
            direct_updates["workflow_base_dir"] = normalized_base_dir
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
            sorted(
                set(normalized_bindings)
                | set(defaulted_keys)
                | set(direct_updates)
            )
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
