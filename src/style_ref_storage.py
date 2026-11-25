"""Persistent storage, validation, and JSON embedding for ANIMA STYLE_REF images."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import math
from pathlib import Path
import re

from app_paths import data_root

STYLE_REF_KEY = "STYLE_REF"
DEFAULT_STYLE_REF_MODEL = "snapref_v2s_pre1.safetensors"
MAX_STYLE_REF_IMAGES = 8
STYLE_REF_IMAGE_ID_RE = re.compile(r"^[a-f0-9]{24}\.png$")
_ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
_STYLE_REF_FIELDS = {"enabled", "strength", "fidelity", "megapixels", "apply_to_refiners", "images"}
_STYLE_REF_IMAGE_FIELDS = {"id", "name", "weight", "enabled"}


def default_style_ref() -> dict:
    return {
        "enabled": False,
        "strength": 1.0,
        "fidelity": 1.0,
        "megapixels": 1.0,
        "apply_to_refiners": True,
        "images": [],
    }


def normalize_style_ref(value) -> dict:
    base = default_style_ref()
    if not isinstance(value, dict):
        return base
    merged = {**base, **copy.deepcopy(value)}
    if isinstance(merged.get("images"), list):
        normalized_images = []
        for item in merged["images"]:
            if not isinstance(item, dict):
                normalized_images.append(item)
                continue
            entry = copy.deepcopy(item)
            entry.setdefault("enabled", True)
            entry.setdefault("weight", 1.0)
            entry.setdefault("name", entry.get("id", ""))
            normalized_images.append(entry)
        merged["images"] = normalized_images
    return merged


def validate_style_ref(value) -> None:
    if not isinstance(value, dict) or set(value) != _STYLE_REF_FIELDS:
        raise ValueError("STYLE_REF 설정 항목을 확인하세요.")
    if not isinstance(value["enabled"], bool):
        raise ValueError("STYLE_REF.enabled: 켜기 또는 끄기를 선택하세요.")
    if not isinstance(value["apply_to_refiners"], bool):
        raise ValueError("STYLE_REF.apply_to_refiners: 켜기 또는 끄기를 선택하세요.")
    for key, low, high in (("strength", 0.0, 2.0), ("fidelity", 0.0, 1.0), ("megapixels", 0.25, 2.0)):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or not low <= number <= high):
            raise ValueError(f"STYLE_REF.{key}: {low:g}~{high:g} 사이의 숫자를 입력하세요.")
    images = value["images"]
    if not isinstance(images, list):
        raise ValueError("STYLE_REF.images에는 배열이 필요합니다.")
    if len(images) > MAX_STYLE_REF_IMAGES:
        raise ValueError(f"STYLE_REF 이미지는 최대 {MAX_STYLE_REF_IMAGES}장까지 등록할 수 있습니다.")
    for item in images:
        if not isinstance(item, dict) or set(item) - _STYLE_REF_IMAGE_FIELDS:
            raise ValueError("STYLE_REF 이미지 항목을 확인하세요.")
        image_id = item.get("id")
        if not isinstance(image_id, str) or not STYLE_REF_IMAGE_ID_RE.fullmatch(image_id):
            raise ValueError("STYLE_REF 이미지 식별자를 확인하세요.")
        name = item.get("name", "")
        if not isinstance(name, str) or len(name) > 200:
            raise ValueError("STYLE_REF 이미지 이름을 확인하세요.")
        weight = item.get("weight", 1.0)
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or not 0.0 <= weight <= 2.0):
            raise ValueError("STYLE_REF 이미지 가중치는 0~2 사이의 숫자를 입력하세요.")
        if not isinstance(item.get("enabled", True), bool):
            raise ValueError("STYLE_REF 이미지 사용 여부는 켜기 또는 끄기여야 합니다.")


def enabled_style_ref(value) -> dict:
    """Copy STYLE_REF keeping only active images when enabled."""
    normalized = normalize_style_ref(value)
    if not normalized["enabled"] or normalized["strength"] <= 0:
        normalized["images"] = []
        return normalized
    normalized["images"] = [
        item for item in normalized["images"]
        if isinstance(item, dict) and item.get("enabled", True) is not False and float(item.get("weight", 1.0)) > 0
    ]
    return normalized


def style_ref_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else data_root() / "style_ref_images"


def style_ref_image_path(image_id: str, root: str | Path | None = None) -> Path:
    if not isinstance(image_id, str) or not STYLE_REF_IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError("STYLE_REF 이미지 식별자를 확인하세요.")
    return style_ref_dir(root) / image_id


def save_style_ref_image(image_bytes: bytes, filename: str = "style_ref.png",
                         root: str | Path | None = None) -> dict:
    from PIL import Image, ImageOps

    if not image_bytes:
        raise ValueError("비어 있는 이미지 파일입니다.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            if opened.format not in _ALLOWED_IMAGE_FORMATS:
                raise ValueError("PNG, JPEG, WebP 이미지 파일을 업로드하세요.")
            rgba = ImageOps.exif_transpose(opened).convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, "white")
            canvas.alpha_composite(rgba)
            rgb = canvas.convert("RGB")
            width, height = rgb.size
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("이미지 파일을 읽을 수 없습니다.") from exc

    png_bytes = buffer.getvalue()
    image_id = f"{hashlib.sha256(png_bytes).hexdigest()[:24]}.png"
    target = style_ref_image_path(image_id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(".png.tmp")
        temporary.write_bytes(png_bytes)
        temporary.replace(target)
    clean_name = Path(str(filename or "style_ref.png").replace("\\", "/")).name.strip()[:200] or image_id
    return {
        "id": image_id,
        "name": clean_name,
        "weight": 1.0,
        "enabled": True,
        "width": width,
        "height": height,
    }


def _safe_display_segment(value: str) -> str:
    return value.replace("/", " ").replace("\\", " ").strip()


def _validated_filename(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("가져올 이미지 파일 이름을 확인하세요.")
    filename = value.strip()
    if filename in {".", ".."} or "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise ValueError("허용되지 않는 이미지 파일 이름입니다.")
    return filename


def resolve_style_ref_source_path(
    item: dict,
    *,
    asset_dir: str | Path | None = None,
    bot_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve an asset or bot image descriptor inside its storage root."""
    if not isinstance(item, dict):
        raise ValueError("가져올 이미지 정보를 확인하세요.")
    source_type = item.get("type")
    filename = _validated_filename(item.get("filename"))

    if source_type == "asset":
        from modes.asset_mode import ASSET_DIR, AssetMode

        character = item.get("character")
        outfit = item.get("outfit")
        expression = item.get("expression")
        if not all(isinstance(part, str) and part.strip() for part in (character, outfit, expression)):
            raise ValueError("에셋 이미지의 캐릭터·복장·표정 정보를 확인하세요.")
        parts = [AssetMode._safe_dirname(part) for part in (character, outfit, expression)]
        if any(part in {".", ".."} for part in parts):
            raise ValueError("에셋 이미지 경로가 올바르지 않습니다.")
        base = Path(asset_dir if asset_dir is not None else ASSET_DIR).resolve()
        target = base.joinpath(*parts, filename).resolve()
        if not target.is_relative_to(base):
            raise ValueError("허용되지 않는 에셋 이미지 경로입니다.")
        if not target.is_file():
            raise ValueError(f"에셋 이미지를 찾을 수 없습니다: {filename}")
        label = f"{_safe_display_segment(character)} · {filename}"
        return target, label

    if source_type == "bot":
        from modes.bot_lora_mode import BOT_DIR, _safe_dirname as bot_safe_dirname

        bot_name = item.get("bot")
        character = item.get("character")
        if not all(isinstance(part, str) and part.strip() for part in (bot_name, character)):
            raise ValueError("봇 이미지의 봇·캐릭터 정보를 확인하세요.")
        parts = [bot_safe_dirname(bot_name), bot_safe_dirname(character)]
        if any(part in {".", ".."} for part in parts):
            raise ValueError("봇 이미지 경로가 올바르지 않습니다.")
        base = Path(bot_dir if bot_dir is not None else BOT_DIR).resolve()
        target = base.joinpath(*parts, filename).resolve()
        if not target.is_relative_to(base):
            raise ValueError("허용되지 않는 봇 이미지 경로입니다.")
        if not target.is_file():
            raise ValueError(f"봇 이미지를 찾을 수 없습니다: {filename}")
        label = f"{_safe_display_segment(bot_name)} · {_safe_display_segment(character)} · {filename}"
        return target, label

    raise ValueError("지원하지 않는 이미지 출처입니다. 봇 또는 에셋에서 선택하세요.")


def import_style_ref_images(
    items: list,
    *,
    root: str | Path | None = None,
    asset_dir: str | Path | None = None,
    bot_dir: str | Path | None = None,
) -> list[dict]:
    if not isinstance(items, list) or not items:
        raise ValueError("가져올 이미지를 선택하세요.")
    if len(items) > MAX_STYLE_REF_IMAGES:
        raise ValueError(f"STYLE_REF 이미지는 한 번에 최대 {MAX_STYLE_REF_IMAGES}장까지 가져올 수 있습니다.")
    resolved = [resolve_style_ref_source_path(item, asset_dir=asset_dir, bot_dir=bot_dir) for item in items]
    imported = []
    for path, label in resolved:
        try:
            imported.append(save_style_ref_image(path.read_bytes(), label, root=root))
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
    return imported


def embed_style_ref_images(payload: dict, root: str | Path | None = None) -> None:
    """Embed active STYLE_REF images into payload['assets'] for self-contained execution."""
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return
    style_ref = fields.get(STYLE_REF_KEY)
    if not isinstance(style_ref, dict):
        return
    if "references" in style_ref:
        return
    if not style_ref.get("enabled") or float(style_ref.get("strength", 1.0)) <= 0:
        style_ref["references"] = []
        return
    assets = payload.setdefault("assets", {})
    references = []
    for item in style_ref.get("images") or []:
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            continue
        image_id = item.get("id", "")
        path = style_ref_image_path(image_id, root)
        if not path.is_file():
            label = item.get("name") or image_id
            raise ValueError(f"STYLE_REF 이미지 파일을 찾을 수 없습니다: {label}. 이미지를 다시 업로드하세요.")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assets.setdefault(digest, {
            "encoding": "base64",
            "name": str(item.get("name") or path.name),
            "media_type": "image/png",
            "sha256": digest,
            "data": base64.b64encode(data).decode("ascii"),
        })
        references.append({
            "asset": digest,
            "weight": weight,
            "name": str(item.get("name") or path.name),
        })
    style_ref["references"] = references
