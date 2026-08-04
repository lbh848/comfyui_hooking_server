from __future__ import annotations

import datetime
import json
import os
import shutil
import traceback
from pathlib import Path


class CredentialStoreError(RuntimeError):
    """설치기 키 파일 읽기 또는 저장 실패."""


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _write_json_atomic(path: Path, value: dict, *, label: str) -> None:
    part = path.with_name(f"{path.name}.part")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2) + os.linesep
    ).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)
    except Exception as exc:
        print(
            f"[COMFY_INSTALL][KEY] {label} 원자적 저장 실패: "
            f"path={path}, part={part}, error={exc}"
        )
        traceback.print_exc()
        raise CredentialStoreError(f"{label} 저장 실패: {exc}") from exc


def _backup_json(path: Path, backup_root: Path, *, filename_prefix: str) -> Path | None:
    if not path.is_file():
        print(f"[COMFY_INSTALL][KEY] 백업 생략, 기존 파일 없음: {path}")
        return None
    try:
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_path = backup_root / f"{filename_prefix}_{_timestamp()}.json"
        shutil.copy2(path, backup_path)
        print(f"[COMFY_INSTALL][KEY] 기존 설정 백업 완료: {backup_path}")
        return backup_path
    except Exception as exc:
        print(
            "[COMFY_INSTALL][KEY] 기존 설정 백업 실패: "
            f"path={path}, backup_root={backup_root}, error={exc}"
        )
        traceback.print_exc()
        raise CredentialStoreError(f"기존 설정 백업 실패: {exc}") from exc


def load_civitai_key(project_root: str | os.PathLike[str]) -> str:
    key_path = Path(project_root).resolve() / "key" / "civitai_key.json"
    if not key_path.is_file():
        print(f"[COMFY_INSTALL][KEY] Civitai 키 파일 없음: {key_path}")
        return ""
    try:
        value = json.loads(key_path.read_text(encoding="utf-8"))
        key = value.get("api_key", "") if isinstance(value, dict) else ""
        if not isinstance(key, str):
            raise CredentialStoreError(
                f"저장된 Civitai api_key가 문자열이 아닙니다: {key_path}"
            )
        print(
            "[COMFY_INSTALL][KEY] Civitai 키 로드 완료: "
            f"{'set' if key else 'empty'}"
        )
        return key
    except CredentialStoreError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][KEY] Civitai 키 로드 실패: "
            f"path={key_path}, error={exc}"
        )
        traceback.print_exc()
        raise CredentialStoreError(f"Civitai 키를 읽을 수 없습니다: {exc}") from exc


def save_civitai_key(
    project_root: str | os.PathLike[str],
    api_key: str,
) -> dict:
    if not isinstance(api_key, str):
        print(
            "[COMFY_INSTALL][KEY] Civitai 키 저장 실패: "
            f"api_key_type={type(api_key).__name__}"
        )
        raise CredentialStoreError("Civitai api_key는 문자열이어야 합니다.")
    key = api_key.strip()
    root = Path(project_root).resolve()
    key_path = root / "key" / "civitai_key.json"
    try:
        backup_path = _backup_json(
            key_path,
            key_path.parent / "backups",
            filename_prefix="civitai_key_before_save",
        )
        _write_json_atomic(
            key_path,
            {"api_key": key},
            label="Civitai 키",
        )
        print(
            "[COMFY_INSTALL][KEY] Civitai 키 저장 완료: "
            f"path={key_path}, state={'set' if key else 'empty'}"
        )
        return {
            "api_key": key,
            "path": str(key_path),
            "backup_path": str(backup_path) if backup_path else None,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][KEY] Civitai 키 저장 실패: "
            f"path={key_path}, error={exc}"
        )
        traceback.print_exc()
        raise CredentialStoreError(f"Civitai 키 저장 실패: {exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict:
    if not path.is_file():
        print(f"[COMFY_INSTALL][KEY] {label} 파일 없음, 새로 생성 예정: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            print(
                f"[COMFY_INSTALL][KEY] {label} 읽기 실패: "
                f"JSON 루트가 객체가 아님, path={path}"
            )
            raise CredentialStoreError(f"{label} JSON 루트가 객체가 아닙니다: {path}")
        return value
    except CredentialStoreError:
        raise
    except Exception as exc:
        print(
            f"[COMFY_INSTALL][KEY] {label} 읽기 실패: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise CredentialStoreError(f"{label}을 읽을 수 없습니다: {exc}") from exc


def resolve_lora_manager_settings_path(
    comfy_root: str | os.PathLike[str],
) -> Path:
    root = Path(comfy_root).resolve()
    node_root = root / "custom_nodes" / "comfyui-lora-manager"
    if not node_root.is_dir():
        print(f"[COMFY_INSTALL][KEY] LoRA Manager 폴더 없음: {node_root}")
        raise CredentialStoreError(
            "설치된 ComfyUI-Lora-Manager를 찾을 수 없습니다. 먼저 ComfyUI를 설치하거나 업데이트하세요."
        )

    portable_path = node_root / "settings.json"
    if portable_path.is_file():
        portable_settings = _load_json_object(
            portable_path,
            label="LoRA Manager 휴대용 설정",
        )
        if portable_settings.get("use_portable_settings") is True:
            print(
                "[COMFY_INSTALL][KEY] LoRA Manager 휴대용 설정 사용: "
                f"{portable_path}"
            )
            return portable_path

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app_data:
        print("[COMFY_INSTALL][KEY] LOCALAPPDATA가 없어 LoRA Manager 설정 경로를 찾지 못했습니다.")
        raise CredentialStoreError(
            "LOCALAPPDATA가 없어 LoRA Manager 설정 경로를 찾을 수 없습니다."
        )
    settings_path = (
        Path(local_app_data).resolve()
        / "ComfyUI-LoRA-Manager"
        / "settings.json"
    )
    print(f"[COMFY_INSTALL][KEY] LoRA Manager 사용자 설정 경로: {settings_path}")
    return settings_path


def save_lora_manager_civitai_key(
    comfy_root: str | os.PathLike[str],
    api_key: str,
) -> dict:
    if not isinstance(api_key, str):
        print(
            "[COMFY_INSTALL][KEY] LoRA Manager Civitai 키 교체 실패: "
            f"api_key_type={type(api_key).__name__}"
        )
        raise CredentialStoreError("LoRA Manager Civitai api_key는 문자열이어야 합니다.")
    key = api_key.strip()
    if not key:
        print("[COMFY_INSTALL][KEY] LoRA Manager Civitai 키 교체 실패: 빈 키")
        raise CredentialStoreError("교체할 Civitai API 키가 비어 있습니다.")

    settings_path = resolve_lora_manager_settings_path(comfy_root)
    settings = _load_json_object(settings_path, label="LoRA Manager 설정")
    backup_path = _backup_json(
        settings_path,
        settings_path.parent / "backups",
        filename_prefix="settings_before_civitai_key_replace",
    )
    settings["civitai_api_key"] = key
    _write_json_atomic(
        settings_path,
        settings,
        label="LoRA Manager 설정",
    )
    print(
        "[COMFY_INSTALL][KEY] LoRA Manager Civitai 키 교체 완료: "
        f"path={settings_path}, state=set"
    )
    return {
        "path": str(settings_path),
        "backup_path": str(backup_path) if backup_path else None,
    }
