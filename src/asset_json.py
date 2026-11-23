"""Asset requests use the same per-node JSON contract as illustrations."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
from modes.prompt_path_normalizer import _CONTROL_BLOCK_HEADER_RE

from illustration_json import illustration_fields, preprocess_reference_images, REFERENCE_PREPROCESS_LABELS

CHARACTER_UPLOAD_OUTFIT = "참고복장"
CHARACTER_UPLOAD_EXPRESSION = "참고표정"


REMOVED_FIELDS = {
    "POSE_ACTIVATE", "POSE_DATA", "HRF_CONTROL_NET", "FACE_ID_ACTIVATE",
    "FACE_ID_STR", "FACE_ID_DIR", "STYLE_ACTIVATE", "STYLE_STR", "STYLE_DIR",
    "CACHE_PATH", "FACE_EMBEDDINGS",
}


def asset_document(positive: str, negative: str = "") -> dict:
    if not positive.lstrip().startswith("{"):
        matches = list(_CONTROL_BLOCK_HEADER_RE.finditer(positive))
        positive = "".join(positive[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(positive)]
                           for index, match in enumerate(matches) if match.group("name") not in REMOVED_FIELDS)
    payload = illustration_fields(positive, negative)
    fields = payload["fields"]
    # The old asset ANIMA block already includes quality and artist tags.
    if "ANIMA" in fields:
        fields["ANIMA_CONTENT"] = fields.pop("ANIMA")
        fields["ANIMA_QUALITY"] = ""
    for key in REMOVED_FIELDS:
        fields.pop(key, None)
    fields.setdefault("CHAR_LIST", ["asset_mode"])
    fields.setdefault("MULTI_CHAR", {"enable": False})
    fields.setdefault("CHAR_FACE_TAG_INFORM", {"list": []})
    payload.setdefault("assets", {})
    return payload


def prepare_asset_prompt(positive: str, negative: str, *, snapref_enabled=False,
                         snapref_dir="", snapref_strength=0.0, snapref_mode="char",
                         snapref_megapixels=1.0, reference_job: dict | None = None) -> str:
    payload = asset_document(positive, negative)
    fields = payload["fields"]
    if fields.get("SEED", -1) == -1:
        fields["SEED"] = random.randint(0, 2**32 - 1)
    settings = fields.setdefault("SNAPREF", {
        "strength": snapref_strength, "mode": snapref_mode, "megapixels": snapref_megapixels,
    })
    settings["enabled"] = bool(snapref_enabled)
    if snapref_enabled and reference_job is not None and not settings.get("references"):
        prepare_asset_references(payload, {**reference_job, "snapref_enabled": True})
    if snapref_enabled and settings.get("references"):
        settings.pop("dir", None)
    elif snapref_enabled and snapref_dir:
        settings["dir"] = snapref_dir
        settings.pop("references", None)
    elif not snapref_enabled:
        settings.pop("dir", None)
        settings["references"] = []
    elif not settings.get("references") and not (settings.get("strength_scale") == "linear" and settings.get("strength") == 0):
        raise ValueError("Ref 이미지가 켜져 있지만 참조 이미지가 없습니다.")
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def prepare_asset_mask_document(document: dict) -> dict:
    """Turn submitted normalized boxes into the sampler's aligned RGB/JSON inputs."""
    from modes.multi_char_mask import layout_fingerprint, render_region_mask, validate_multi_char_layout

    fields = document["fields"]
    multi = fields.get("MULTI_CHAR", {})
    if not isinstance(multi, dict) or not multi.get("enable") or "layout" not in multi:
        return document
    names = multi.get("char_name_list")
    if not isinstance(names, list) or not 1 <= len(names) <= 3 or multi.get("char_num") != len(names):
        raise ValueError("마스크 기반 생성의 캐릭터 수는 1~3명이어야 합니다.")
    for key in ("char_inform", "char_trigger_list"):
        if not isinstance(multi.get(key), list) or len(multi[key]) != len(names):
            raise ValueError(f"영역별 프롬프트 개수를 확인하세요: {key}")
    if any(not isinstance(group, list) or any(not isinstance(value, str) for value in group) for group in multi["char_trigger_list"]):
        raise ValueError("영역별 LoRA 트리거 형식을 확인하세요.")
    placements = multi.get("char_composition_list")
    if "char_composition_list" in multi and (not isinstance(placements, list) or len(placements) != len(names)
            or any(not isinstance(value, str) for value in placements)):
        raise ValueError("마스크별 자연어를 확인하세요.")
    if any(not isinstance(value, str) or not (value.strip() or placements is not None and placements[index].strip())
           for index, value in enumerate(multi["char_inform"])):
        raise ValueError("각 영역의 캐릭터 프롬프트 또는 자연어를 입력하세요.")
    if not isinstance(multi.get("composition_prompt"), str) or placements is None and not multi["composition_prompt"].strip():
        raise ValueError("공통 자연어를 확인하세요.")
    if not isinstance(multi.get("background_prompt", ""), str):
        raise ValueError("공통 배경은 문자열이어야 합니다.")
    shared = multi.get("shared_tag")
    if not isinstance(shared, dict) or any(not isinstance(shared.get(key), list) or any(not isinstance(part, str) for part in shared[key]) for key in ("before_char", "after_char")):
        raise ValueError("공통 그림체 프롬프트 형식을 확인하세요.")
    background_triggers = multi.get("background_trigger_list", [])
    if not isinstance(background_triggers, list) or any(not isinstance(value, str) for value in background_triggers):
        raise ValueError("공통 LoRA 트리거 형식을 확인하세요.")
    # The model preset may add style LoRAs after the browser assembled its tags.
    style_triggers = [item.get("trigger", "").strip() for item in fields.get("STYLE_LORA_DATA", {}).get("list", [])
                      if item.get("enabled", True) is not False
                      and item.get("BASE", "anima") == "anima" and item.get("trigger")]
    multi["background_trigger_list"] = list(dict.fromkeys([*background_triggers, *style_triggers]))
    groups = [list(dict.fromkeys([*group, *style_triggers])) for group in multi["char_trigger_list"]]
    if len(names) == 1 and not any(part.strip() for part in [*shared["before_char"], *shared["after_char"], *multi["background_trigger_list"], multi.get("background_prompt", "")]):
        raise ValueError("1인 위치 제어에 사용할 공통 배경 또는 그림체 태그를 입력하세요.")
    dimensions = [fields.get("IMG_W"), fields.get("IMG_H")]
    if any(isinstance(value, bool) or not isinstance(value, int) or not 64 <= value <= 8192 for value in dimensions):
        raise ValueError("이미지 너비와 높이는 64~8192 사이의 정수여야 합니다.")
    source = multi["layout"]
    if not isinstance(source, dict) or not isinstance(source.get("regions"), list) or any(not isinstance(region, dict) for region in source["regions"]):
        raise ValueError("사각형 영역 좌표를 확인하세요.")
    # Render at a bounded resolution; the sampler resizes masks to its latent size.
    scale = 1024 / max(dimensions)
    layout = validate_multi_char_layout({
        "regions": [{key: region.get(key) for key in ("name", "x", "y", "width", "height")} for region in source["regions"]],
        "mask_width": round(dimensions[0] * scale), "mask_height": round(dimensions[1] * scale),
    }, names)
    order = layout["character_order"]
    positions = [names.index(name) for name in order]
    if placements is not None:
        multi["char_composition_list"] = [placements[index].strip() for index in positions]
    multi.update(char_name_list=order, char_inform=[multi["char_inform"][index] for index in positions],
                 char_trigger_list=[groups[index] for index in positions], layout=layout,
                 mask_render_version=layout["mask_render_version"], mask_fingerprint=layout_fingerprint(layout))
    fields["CHAR_LIST"] = order
    mask = render_region_mask(layout)
    if any(mask.getchannel(channel).getbbox() is None for channel in "RGB"[:len(names)]):
        raise ValueError("겹침 처리 후 비어 있는 영역이 있습니다. 사각형의 위치나 크기를 조절하세요.")
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    fields["REGION_MASK"] = _embed_asset(document, buffer.getvalue(), "region_mask.png")
    return document


def _embed_asset(document: dict, data: bytes, name: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    document.setdefault("assets", {})[digest] = {"encoding": "base64", "name": name, "media_type": "image/png",
                                               "sha256": digest, "data": base64.b64encode(data).decode("ascii")}
    return digest


def _embed_reference_images(document: dict, references: list, names: list) -> None:
    """Use the illustration asset/face_asset/role contract for each asset character."""
    from PIL import Image, ImageOps
    settings = document["fields"]["SNAPREF"]
    mode = settings.get("secondary_reference", "none")

    def embed(images, label):
        if not isinstance(images, list) or len(images) > 1 or any(not isinstance(image, dict) for image in images):
            raise ValueError(f"{label}는 한 장만 선택하세요.")
        if not images:
            return None
        path = Path(images[0].get("local_path") or "")
        if not path.is_file():
            raise ValueError(f"{label} 파일을 찾을 수 없습니다. 이미지를 다시 선택하세요.")
        with Image.open(path) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            image = Image.new("RGBA", rgba.size, "white")
            image.alpha_composite(rgba)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        return _embed_asset(document, buffer.getvalue(), path.stem + ".png")

    if not isinstance(references, list) or any(not isinstance(ref, dict) for ref in references):
        raise ValueError("참조 이미지 목록을 확인하세요.")
    embedded = []
    seen = set()
    for reference in references:
        name = reference.get("CHAR")
        if name not in names or name in seen:
            raise ValueError("영역의 참조 이미지와 캐릭터 연결을 확인하세요.")
        seen.add(name)
        face = embed(reference.get("face_images", []), "얼굴 이미지") if mode in {"face_only", "face", "stage"} else None
        primary = face if mode == "face_only" and face else embed(reference.get("images", []), "대표 이미지")
        if not primary:
            raise ValueError("Ref 이미지가 켜져 있지만 얼굴 이미지와 대표 이미지가 모두 없습니다." if mode == "face_only" else "Ref 이미지가 켜져 있지만 대표 이미지가 없습니다.")
        embedded.append({"CHAR": name, "asset": primary, **({"face_asset": face} if face else {})})
        if mode == "representative":
            secondary = embed(reference.get("secondary_images", []), "두 번째 대표 이미지")
            if secondary:
                embedded.append({"CHAR": name, "asset": secondary, "role": "representative"})
    settings["references"] = embedded
    settings.pop("dir", None)


def prepare_asset_references(document: dict, job: dict) -> dict:
    """Validate and snapshot selected images before preview or queue submission."""
    settings = document["fields"].setdefault("SNAPREF", {})
    if not isinstance(settings, dict):
        raise ValueError("참조 이미지 설정을 확인하세요.")
    choices = {"preprocess": REFERENCE_PREPROCESS_LABELS, "mode": {"char", "cs"},
               "secondary_reference": {"none", "face_only", "face", "representative", "stage"},
               "strength_scale": {"linear", "log"}}
    defaults = {"preprocess": "none", "mode": "char", "secondary_reference": "none", "strength_scale": "log"}
    for key, options in choices.items():
        if not isinstance(settings.get(key, defaults[key]), str) or settings.get(key, defaults[key]) not in options:
            raise ValueError(f"참조 이미지 설정을 확인하세요: {key}")
    linear = settings.get("strength_scale") == "linear"
    for key, default, low, high in (("strength", 1 if linear else 0, 0 if linear else -3, 2 if linear else 3),
                                   ("fidelity", 1, 0, 1), ("megapixels", 1, .1, 4)):
        value = settings.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"참조 이미지 설정 범위를 확인하세요: {key}")
    enabled = bool(job.get("snapref_enabled", settings.get("enabled", False)))
    settings["enabled"] = enabled
    if not enabled or linear and settings.get("strength", 1) == 0:
        settings["references"] = []
        settings.pop("dir", None)
        return document
    multi = document["fields"].get("MULTI_CHAR", {})
    names = multi.get("char_name_list", []) if multi.get("enable") else document["fields"].get("CHAR_LIST", ["asset_mode"])
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("참조 이미지를 연결할 캐릭터를 확인하세요.")
    references = job.get("regional_references", []) if multi.get("enable") else [{
        "CHAR": names[0], "images": job.get("snapref_images", []),
        "face_images": job.get("snapref_face_images", []), "secondary_images": job.get("snapref_secondary_images", []),
    }]
    _embed_reference_images(document, references, names)
    if not settings["references"]:
        raise ValueError("Ref 이미지가 켜져 있지만 참조 이미지가 없습니다.")
    preprocess_reference_images(document)
    return document


def embed_asset_mask_references(positive: str, references: list) -> str:
    """Compatibility for regional jobs queued before image snapshots were added."""
    document = json.loads(positive)
    _embed_reference_images(document, references, document["fields"].get("MULTI_CHAR", {}).get("char_name_list", []))
    document["fields"]["SNAPREF"]["enabled"] = bool(references)
    preprocess_reference_images(document)
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def merge_tagged_character_prompt(document: dict, tags: list[str], background_tags: str = "") -> dict:
    """Insert Anima Tagger character tags into a prepared style/model prompt document."""
    merged = copy.deepcopy(document)
    fields = merged.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("프롬프트 JSON의 fields를 확인하세요.")
    clean_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
    character_prompt = ", ".join(clean_tags)
    background = background_tags.strip() if isinstance(background_tags, str) else ""

    def inject_character_tags(existing: str) -> str:
        text = existing.strip() if isinstance(existing, str) else ""
        if not character_prompt:
            return text
        prefix = text
        suffix = ""
        if background:
            if text == background:
                prefix, suffix = "", background
            elif text.endswith(", " + background):
                prefix, suffix = text[:-(len(background) + 2)].strip(), background
        return ", ".join(part for part in (prefix, character_prompt, suffix) if part)

    for key in ("ANIMA_CONTENT", "SDXL_CONTENT"):
        if key in fields:
            fields[key] = inject_character_tags(fields[key])

    multi = fields.get("MULTI_CHAR")
    if isinstance(multi, dict) and multi.get("mask_area_limit_enabled"):
        multi["char_inform"] = [character_prompt or "character"]
        multi["char_composition_list"] = [""]

    return merged
