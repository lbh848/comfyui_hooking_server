from __future__ import annotations

import datetime
import json
import os
import shutil
import traceback
from pathlib import Path


class CredentialStoreError(RuntimeError):
    """설치기 키 파일 읽기 또는 저장 실패."""


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
    requirements_dir: str | os.PathLike[str],
    api_key: str,
) -> dict:
    if not isinstance(api_key, str):
        raise CredentialStoreError("Civitai api_key는 문자열이어야 합니다.")
    key = api_key.strip()
    root = Path(project_root).resolve()
    key_path = root / "key" / "civitai_key.json"
    backup_path: Path | None = None
    part = key_path.with_name(f"{key_path.name}.part")
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.is_file():
            backup_root = Path(requirements_dir).resolve()
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = backup_root / f"civitai_key_before_save_{stamp}.json"
            shutil.copy2(key_path, backup_path)
            print(
                "[COMFY_INSTALL][KEY] 기존 Civitai 키 백업 완료: "
                f"{backup_path}"
            )
        payload = (
            json.dumps({"api_key": key}, ensure_ascii=False, indent=2)
            + os.linesep
        ).encode("utf-8")
        with part.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, key_path)
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

