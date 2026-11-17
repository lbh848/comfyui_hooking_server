"""Shared category roots for trained LoRA storage and remote artifacts."""
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from app_paths import SOURCE_DIR, data_root


MANAGED_LORA_ROOT = "SOYA_CHAR_LORA"
ASSET_LORA_FOLDER = "SOYA_ASSET_LORA"
LORA_CATEGORY_CONFIG_KEYS = {
    ASSET_LORA_FOLDER: "lora_load_path",
    "SOYA_BOT_LORA": "bot_lora_load_path",
    "SOYA_INSTANCE_LORA": "instance_lora_load_path",
    "SOYA_STYLE_LORA": "style_lora_load_path",
}


_STORAGE_NAME_ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_STORAGE_NAME_ALLOWED_TEXT = "영문 대소문자, 숫자, 밑줄(_), 하이픈(-), 마침표(.)"
_WINDOWS_RESERVED_FILE_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_lora_storage_name(value, label: str = "이름") -> str:
    """Use the same unchanged directory names for bots, assets and LoRAs."""
    if not isinstance(value, str):
        return f"{label}은(는) 문자열이어야 합니다."
    if not value:
        return f"{label}이(가) 비어있습니다."
    if not _STORAGE_NAME_ALLOWED_PATTERN.fullmatch(value):
        invalid = list(dict.fromkeys(
            char for char in value if not _STORAGE_NAME_ALLOWED_PATTERN.fullmatch(char)
        ))
        invalid_text = ", ".join("공백" if char == " " else repr(char) for char in invalid)
        return (
            f"{label}에 사용할 수 없는 문자가 있습니다: {invalid_text}. "
            f"사용 가능: {_STORAGE_NAME_ALLOWED_TEXT}."
        )
    if value in {".", ".."} or value.endswith("."):
        return f"{label}은(는) 마침표로만 구성하거나 마침표로 끝낼 수 없습니다."
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_FILE_STEMS:
        return f"{label}에 Windows 예약 이름 {value!r}을(를) 사용할 수 없습니다."
    return ""


# 인스턴스 LoRA는 더 이상 사용하지 않는다.
DEFAULT_LORA_CATEGORY_FOLDERS = (
    ASSET_LORA_FOLDER,
    "SOYA_BOT_LORA",
    "SOYA_STYLE_LORA",
)


def lora_training_session_names(directory: str) -> list[str]:
    """List training sessions, excluding the trainer's old progress directory."""
    with os.scandir(directory) as entries:
        return sorted((entry.name for entry in entries if entry.name.casefold() != "logs" and entry.is_dir()),
                      reverse=True)


def default_lora_load_paths() -> dict[str, str]:
    """Portable category roots relative to the install's data directory."""
    root = f"comfy/models/loras/{MANAGED_LORA_ROOT}"
    return {
        LORA_CATEGORY_CONFIG_KEYS[folder]: f"{root}/{folder}"
        for folder in DEFAULT_LORA_CATEGORY_FOLDERS
    }


def resolve_lora_load_path(value: Any, project_root: str | Path = SOURCE_DIR) -> str:
    """Resolve a configured LoRA path against the install's data directory."""
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    return str(path if path.is_absolute() else data_root(project_root) / path)


def resolve_lora_load_paths(
    config: Mapping[str, Any],
    project_root: str | Path = SOURCE_DIR,
) -> dict[str, str]:
    defaults = default_lora_load_paths()
    return {
        key: resolve_lora_load_path(config.get(key) or defaults.get(key, ""), project_root)
        for key in LORA_CATEGORY_CONFIG_KEYS.values()
    }


def portable_lora_load_path(value: Any, project_root: str | Path = SOURCE_DIR) -> str:
    """Store LoRA paths relative to the data folder when they live inside it."""
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return PurePosixPath(text.replace("\\", "/")).as_posix()
    try:
        return path.resolve().relative_to(data_root(project_root).resolve()).as_posix()
    except ValueError:
        return text


def portable_lora_load_paths(
    config: Mapping[str, Any],
    project_root: str | Path = SOURCE_DIR,
) -> dict[str, str]:
    return {
        key: portable_lora_load_path(config.get(key) or "", project_root)
        for key in LORA_CATEGORY_CONFIG_KEYS.values()
    }


def managed_lora_root(config: Mapping[str, Any]) -> str:
    asset_root = str(config.get("lora_load_path") or "").strip()
    return str(Path(asset_root).parent) if asset_root else ""


def bot_lora_load_path(config: Mapping[str, Any]) -> str:
    configured = str(config.get("bot_lora_load_path") or "").strip()
    root = managed_lora_root(config)
    return configured or (str(Path(root) / "SOYA_BOT_LORA") if root else "")


def lora_artifact_destination(config: Mapping[str, Any], relative_path: str) -> Path:
    """Resolve an artifact relative to the shared root into its category folder."""
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or any(":" in part for part in relative.parts):
        raise ValueError(f"안전하지 않은 LoRA 결과 상대 경로입니다: {relative_path}")
    key = LORA_CATEGORY_CONFIG_KEYS.get(relative.parts[0])
    configured = str(config.get(key) or "").strip() if key else ""
    root_raw = configured or managed_lora_root(config)
    if not root_raw:
        raise ValueError("LoRA 결과를 저장할 로컬 경로가 비어 있습니다.")
    root = Path(root_raw).resolve()
    parts = relative.parts[1:] if configured else relative.parts
    target = root.joinpath(*parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"로컬 LoRA 폴더 밖에는 결과를 저장할 수 없습니다: {target}")
    return target
