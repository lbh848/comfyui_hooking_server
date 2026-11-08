"""Distribution preset data and the shared expression/composition category."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from functools import lru_cache

from app_paths import SOURCE_DIR


def resolve_preset_tags(tags: dict, category: str, name: str) -> list[str]:
    """Read only an explicitly selected preset; an empty selection adds nothing."""
    if not name:
        return []
    presets = tags.get(category, {})
    if not isinstance(presets, dict) or name not in presets:
        raise ValueError(f"선택한 프리셋을 찾을 수 없습니다: {category} / {name}")
    values = presets[name]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"프리셋 태그 형식을 확인하세요: {category} / {name}")
    return list(values)


def preset_catalog_only(tags: dict) -> dict:
    """Keep named catalogs and character references, never implicit tag sets."""
    return {key: value for key, value in tags.items()
            if key in {"appearances", "outfits", "expressions", "characters"} or key.endswith("_presets")}


def normalize_preset_name(name: str) -> str:
    name = re.sub(r"^SDstudio-80종\s*공유용2[/\\_-]", "SD-80_", name.strip())
    name = re.sub(r"^SD-80종", "SD-80", name)
    return name.replace("/", "_").replace("\\", "_")


def merge_expression_presets(tags: dict) -> dict:
    """Join same-name expression and composition fragments without losing order."""
    result = copy.deepcopy(tags)
    composition = result.pop("composition_presets", {})
    if not isinstance(composition, dict):
        raise ValueError("구도 프리셋 형식을 확인하세요.")
    expressions = result.setdefault("expressions", {})
    if not isinstance(expressions, dict):
        raise ValueError("표정·구도 프리셋 형식을 확인하세요.")
    for name, value in composition.items():
        if name not in expressions:
            expressions[name] = value
        else:
            if not isinstance(expressions[name], list) or not isinstance(value, list):
                raise ValueError("표정·구도 태그 형식을 확인하세요.")
            expressions[name] = list(dict.fromkeys([*expressions[name], *value]))
    return result


def load_builtin_presets(project_root: Path = SOURCE_DIR, *, hidden: bool = False) -> dict:
    path = Path(project_root) / "presets" / "presets.json"
    if not path.is_file():
        return {}
    stat = path.stat()
    document = _read_builtin_presets(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    return copy.deepcopy(document.get("hidden" if hidden else "presets", {}))


@lru_cache(maxsize=8)
def _read_builtin_presets(path: str, modified: int, size: int) -> dict:
    del modified, size
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("presets"), dict):
        raise ValueError("배포 프리셋 파일 형식을 확인하세요.")
    if not isinstance(document.get("hidden", {}), dict):
        raise ValueError("배포 프리셋 파일 형식을 확인하세요.")
    return document
