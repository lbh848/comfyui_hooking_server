"""Shared post-processing settings and image rendering for asset and background workspaces."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import threading
import traceback
from typing import Any

from aiohttp import web
import cv2
import numpy as np
from PIL import Image

from configuration import settings_lock
from json_store import json_file_signature, write_json_atomic


ASSET_WORKSPACE = "asset"
BG_ASSET_WORKSPACE = "bg_asset"
VALID_WORKSPACES = (ASSET_WORKSPACE, BG_ASSET_WORKSPACE)

ALPHA_FOREGROUND_THRESHOLD = 96
MIN_OUTLINE_COMPONENT_PIXELS = 16
MIN_TRANSPARENT_RATIO = 0.001

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_ASSET_POSTPROCESS_SETTINGS: dict[str, dict[str, Any]] = {
    "character_outline": {
        "enabled": False,
        "color": "#ffffff",
        "width": 6,
        "feather": 1.0,
        "opacity": 100,
    },
}

DEFAULT_BG_ASSET_POSTPROCESS_SETTINGS: dict[str, dict[str, Any]] = {
    "blur": {
        "enabled": False,
        "radius": 6.0,
    },
}

_PREVIEW_CACHE_MAX_ENTRIES = 64
_PREVIEW_CACHE_MAX_BYTES = 32 * 1024 * 1024
_preview_cache_lock = threading.Lock()
_preview_cache: OrderedDict[tuple[Any, ...], tuple[bytes, str, bool]] = OrderedDict()
_preview_cache_bytes = 0


def default_postprocess_settings(workspace: str) -> dict[str, dict[str, Any]]:
    if workspace == ASSET_WORKSPACE:
        return deepcopy(DEFAULT_ASSET_POSTPROCESS_SETTINGS)
    if workspace == BG_ASSET_WORKSPACE:
        return deepcopy(DEFAULT_BG_ASSET_POSTPROCESS_SETTINGS)
    raise ValueError(f"지원하지 않는 후처리 워크스페이스입니다: {workspace}")


def _validate_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} 설정은 켜기/끄기 값이어야 합니다.")
    return value


def _validate_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}는 {minimum}~{maximum} 사이의 정수여야 합니다.")
    rounded = int(round(value))
    if abs(float(value) - rounded) > 1e-6 or not minimum <= rounded <= maximum:
        raise ValueError(f"{label}는 {minimum}~{maximum} 사이의 정수여야 합니다.")
    return rounded


def _validate_float(value: Any, label: str, minimum: float, maximum: float, *, decimals: int = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}는 {minimum}~{maximum} 사이의 숫자여야 합니다.")
    number = round(float(value), decimals)
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}는 {minimum}~{maximum} 사이의 숫자여야 합니다.")
    return number


def _validate_hex_color(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX_COLOR_RE.fullmatch(value.strip()):
        raise ValueError(f"{label}는 #RRGGBB 형식의 색상 코드여야 합니다.")
    return value.strip().lower()


def _parse_hex_rgb(color: str) -> tuple[int, int, int]:
    clean = _validate_hex_color(color, "테두리 색상")
    return int(clean[1:3], 16), int(clean[3:5], 16), int(clean[5:7], 16)


def normalize_postprocess_settings(
    workspace: str,
    payload: Any = None,
    *,
    base: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate and normalize post-processing settings for a workspace."""
    defaults = default_postprocess_settings(workspace)
    merged = deepcopy(defaults)
    if isinstance(base, dict):
        for feature_key, default_feature in defaults.items():
            base_feature = base.get(feature_key)
            if isinstance(base_feature, dict):
                merged[feature_key] = {**default_feature, **{k: v for k, v in base_feature.items() if k in default_feature}}

    if payload is not None:
        if not isinstance(payload, dict):
            if strict:
                raise ValueError("후처리 설정 형식을 확인하세요.")
            payload = {}
        unknown_features = set(payload) - set(defaults)
        if strict and unknown_features:
            raise ValueError("지원하지 않는 후처리 항목이 포함되어 있습니다.")
        for feature_key, default_feature in defaults.items():
            if feature_key not in payload:
                continue
            incoming = payload[feature_key]
            if not isinstance(incoming, dict):
                if strict:
                    raise ValueError(f"{feature_key} 설정 형식을 확인하세요.")
                continue
            unknown_keys = set(incoming) - set(default_feature)
            if strict and unknown_keys:
                raise ValueError(f"{feature_key} 설정에 지원하지 않는 필드가 있습니다.")
            for key, value in incoming.items():
                if key in default_feature:
                    merged[feature_key][key] = value

    if workspace == ASSET_WORKSPACE:
        outline = merged["character_outline"]
        if strict:
            outline["enabled"] = _validate_bool(outline["enabled"], "캐릭터 테두리 사용")
            outline["color"] = _validate_hex_color(outline["color"], "테두리 색상")
            outline["width"] = _validate_int(outline["width"], "테두리 두께", 1, 64)
            outline["feather"] = _validate_float(outline["feather"], "경계 부드러움", 0.0, 20.0)
            outline["opacity"] = _validate_int(outline["opacity"], "테두리 불투명도", 0, 100)
        else:
            default_outline = defaults["character_outline"]
            try:
                outline["enabled"] = _validate_bool(outline["enabled"], "캐릭터 테두리 사용")
            except ValueError:
                outline["enabled"] = default_outline["enabled"]
            try:
                outline["color"] = _validate_hex_color(outline["color"], "테두리 색상")
            except ValueError:
                outline["color"] = default_outline["color"]
            try:
                outline["width"] = _validate_int(outline["width"], "테두리 두께", 1, 64)
            except ValueError:
                outline["width"] = default_outline["width"]
            try:
                outline["feather"] = _validate_float(outline["feather"], "경계 부드러움", 0.0, 20.0)
            except ValueError:
                outline["feather"] = default_outline["feather"]
            try:
                outline["opacity"] = _validate_int(outline["opacity"], "테두리 불투명도", 0, 100)
            except ValueError:
                outline["opacity"] = default_outline["opacity"]
    elif workspace == BG_ASSET_WORKSPACE:
        blur = merged["blur"]
        if strict:
            blur["enabled"] = _validate_bool(blur["enabled"], "흐림 효과 사용")
            blur["radius"] = _validate_float(blur["radius"], "흐림 강도", 0.5, 50.0)
        else:
            default_blur = defaults["blur"]
            try:
                blur["enabled"] = _validate_bool(blur["enabled"], "흐림 효과 사용")
            except ValueError:
                blur["enabled"] = default_blur["enabled"]
            try:
                blur["radius"] = _validate_float(blur["radius"], "흐림 강도", 0.5, 50.0)
            except ValueError:
                blur["radius"] = default_blur["radius"]

    return merged


def has_active_postprocess(workspace: str, settings: dict[str, Any] | None) -> bool:
    if not isinstance(settings, dict) or workspace not in VALID_WORKSPACES:
        return False
    normalized = normalize_postprocess_settings(workspace, settings, strict=False)
    if workspace == ASSET_WORKSPACE:
        outline = normalized["character_outline"]
        return bool(outline["enabled"]) and outline["width"] > 0 and outline["opacity"] > 0
    if workspace == BG_ASSET_WORKSPACE:
        blur = normalized["blur"]
        return bool(blur["enabled"]) and blur["radius"] > 0
    return False


def postprocess_revision_token(workspace: str, settings: dict[str, Any] | None) -> str:
    normalized = normalize_postprocess_settings(workspace, settings, strict=False)
    encoded = json.dumps(
        {"workspace": workspace, "settings": normalized},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class AssetPostprocessStore:
    """File-backed workspace post-processing configuration store."""

    def __init__(self, workspace: str, path: str | Path):
        if workspace not in VALID_WORKSPACES:
            raise ValueError(f"지원하지 않는 후처리 워크스페이스입니다: {workspace}")
        self.workspace = workspace
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cached_signature: tuple[Any, ...] | None = None
        self._cached_settings: dict[str, dict[str, Any]] | None = None

    def read(self) -> dict[str, dict[str, Any]]:
        signature = json_file_signature(self.path)
        with self._lock:
            if self._cached_settings is not None and signature == self._cached_signature:
                return deepcopy(self._cached_settings)

        raw: Any = None
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    f"[POSTPROCESS] 설정 파일 읽기 실패, 기본값 사용: "
                    f"workspace={self.workspace}, path={self.path}, error={type(exc).__name__}: {exc}"
                )
                raw = None

        normalized = normalize_postprocess_settings(self.workspace, raw, strict=False)
        with self._lock:
            self._cached_signature = signature
            self._cached_settings = deepcopy(normalized)
        return normalized

    def save(self, patch: dict[str, Any]) -> dict[str, dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with settings_lock(self.path):
            current = self.read()
            updated = normalize_postprocess_settings(
                self.workspace,
                patch,
                base=current,
                strict=True,
            )
            write_json_atomic(self.path, updated)
            signature = json_file_signature(self.path)
            with self._lock:
                self._cached_signature = signature
                self._cached_settings = deepcopy(updated)
            return deepcopy(updated)

    def public_payload(self, settings: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        current = normalize_postprocess_settings(
            self.workspace,
            settings if settings is not None else self.read(),
            strict=False,
        )
        return {
            "workspace": self.workspace,
            "settings": current,
            "active": has_active_postprocess(self.workspace, current),
            "revision": postprocess_revision_token(self.workspace, current),
        }


def _clean_foreground_mask(fg_mask: np.ndarray) -> np.ndarray:
    """Drop isolated tiny alpha-noise specks while keeping the real character silhouette."""
    mask_u8 = fg_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 2:
        return fg_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = np.where(areas >= MIN_OUTLINE_COMPONENT_PIXELS)[0] + 1
    if keep_labels.size == 0 or keep_labels.size == num_labels - 1:
        return fg_mask
    keep_lut = np.zeros(num_labels, dtype=np.bool_)
    keep_lut[keep_labels] = True
    return keep_lut[labels]


def apply_character_outline(
    image: Image.Image,
    config: dict[str, Any],
) -> tuple[Image.Image, bool]:
    """Add a smooth outer outline around transparent-background character cutouts."""
    if not config.get("enabled"):
        return image, False
    width = int(config.get("width", 0))
    opacity = int(config.get("opacity", 0))
    if width <= 0 or opacity <= 0:
        return image, False
    if "A" not in image.getbands():
        return image, False

    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    fg_mask = alpha >= ALPHA_FOREGROUND_THRESHOLD
    if not np.any(fg_mask):
        return image, False
    transparent_ratio = float(np.mean(~fg_mask))
    if transparent_ratio < MIN_TRANSPARENT_RATIO:
        return image, False

    fg_mask = _clean_foreground_mask(fg_mask)
    if not np.any(fg_mask):
        return image, False

    outside = (~fg_mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)

    width_f = float(width)
    feather_f = max(0.0, float(config.get("feather", 1.0)))
    opacity_f = max(0.0, min(1.0, float(opacity) / 100.0))

    if feather_f <= 0.0:
        coverage = np.clip(width_f + 0.5 - dist, 0.0, 1.0)
    else:
        span = max(1.0, feather_f)
        t = np.clip((width_f + feather_f * 0.5 - dist) / span, 0.0, 1.0)
        coverage = t * t * (3.0 - 2.0 * t)

    dst_a = coverage * opacity_f

    src = np.asarray(rgba, dtype=np.float32)
    src_rgb = src[..., :3]
    src_a = src[..., 3] / 255.0
    inv_src_a = 1.0 - src_a
    dst_contrib = dst_a * inv_src_a
    out_a = src_a + dst_contrib

    r, g, b = _parse_hex_rgb(str(config.get("color", "#ffffff")))
    outline_rgb = np.array([r, g, b], dtype=np.float32)

    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (
        src_rgb * src_a[..., None] + outline_rgb * dst_contrib[..., None]
    ) / safe_a[..., None]
    out_rgb = np.where(out_a[..., None] > 1e-6, out_rgb, 0.0)

    out = np.empty_like(src, dtype=np.uint8)
    out[..., :3] = np.rint(np.clip(out_rgb, 0.0, 255.0)).astype(np.uint8)
    out[..., 3] = np.rint(np.clip(out_a * 255.0, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA"), True


def apply_background_blur(
    image: Image.Image,
    config: dict[str, Any],
) -> tuple[Image.Image, bool]:
    """Apply edge-safe Gaussian blur to background asset images."""
    if not config.get("enabled"):
        return image, False
    radius = float(config.get("radius", 0.0))
    if radius <= 0.0:
        return image, False

    target_mode = "RGBA" if "A" in image.getbands() else "RGB"
    converted = image if image.mode == target_mode else image.convert(target_mode)
    arr = np.asarray(converted, dtype=np.uint8)
    blurred = cv2.GaussianBlur(
        arr,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
        borderType=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(blurred, mode=target_mode), True


def apply_postprocess(
    image: Image.Image,
    workspace: str,
    settings: dict[str, Any] | None,
) -> tuple[Image.Image, bool]:
    """Single canonical post-processing pipeline shared by preview, gallery, and export."""
    if not isinstance(settings, dict) or workspace not in VALID_WORKSPACES:
        return image, False
    normalized = normalize_postprocess_settings(workspace, settings, strict=False)
    current = image
    applied_any = False
    if workspace == ASSET_WORKSPACE:
        current, applied = apply_character_outline(current, normalized["character_outline"])
        applied_any = applied_any or applied
    elif workspace == BG_ASSET_WORKSPACE:
        current, applied = apply_background_blur(current, normalized["blur"])
        applied_any = applied_any or applied
    return current, applied_any


def _is_animated_image(image: Image.Image) -> bool:
    return bool(getattr(image, "is_animated", False)) and int(getattr(image, "n_frames", 1)) > 1


def _render_file_bytes(
    filepath: str,
    workspace: str,
    settings: dict[str, Any],
) -> tuple[bytes, str, bool]:
    mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    if not has_active_postprocess(workspace, settings):
        with open(filepath, "rb") as reader:
            return reader.read(), mime, False

    with Image.open(filepath) as opened:
        if _is_animated_image(opened):
            with open(filepath, "rb") as reader:
                return reader.read(), mime, False
        opened.load()
        processed, applied = apply_postprocess(opened, workspace, settings)
        if not applied:
            with open(filepath, "rb") as reader:
                return reader.read(), mime, False
        buf = io.BytesIO()
        processed.save(buf, format="WEBP", quality=90, method=4)
        return buf.getvalue(), "image/webp", True


def render_cached_postprocess_file(
    filepath: str,
    workspace: str,
    settings: dict[str, Any],
) -> tuple[bytes, str, bool]:
    """Render and LRU-cache post-processed image bytes for live gallery and preview views."""
    global _preview_cache_bytes

    real_path = os.path.realpath(filepath)
    stat = os.stat(real_path)
    revision = postprocess_revision_token(workspace, settings)
    cache_key = (real_path, stat.st_mtime_ns, stat.st_size, workspace, revision)

    with _preview_cache_lock:
        cached = _preview_cache.get(cache_key)
        if cached is not None:
            _preview_cache.move_to_end(cache_key)
            return cached

    rendered = _render_file_bytes(real_path, workspace, settings)
    data_size = len(rendered[0])
    if data_size <= _PREVIEW_CACHE_MAX_BYTES // 2:
        with _preview_cache_lock:
            previous = _preview_cache.pop(cache_key, None)
            if previous is not None:
                _preview_cache_bytes -= len(previous[0])
            _preview_cache[cache_key] = rendered
            _preview_cache_bytes += data_size
            while (
                len(_preview_cache) > _PREVIEW_CACHE_MAX_ENTRIES
                or _preview_cache_bytes > _PREVIEW_CACHE_MAX_BYTES
            ):
                _, evicted = _preview_cache.popitem(last=False)
                _preview_cache_bytes -= len(evicted[0])
    return rendered


def register_postprocess_runtime(
    server: Any,
    workspace: str,
    mode: Any,
    *,
    prefix: str,
) -> AssetPostprocessStore:
    """Register GET/POST settings and live preview routes for an asset workspace."""
    store = AssetPostprocessStore(workspace, Path(mode.asset_data_dir) / "postprocess.json")
    if workspace == ASSET_WORKSPACE:
        server.asset_postprocess_store = store
    else:
        server.bg_asset_postprocess_store = store

    async def settings_handler(request: web.Request) -> web.Response:
        try:
            if request.method == "POST":
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("후처리 설정은 JSON 객체여야 합니다.")
                patch = body.get("settings") if "settings" in body else body
                if not isinstance(patch, dict):
                    raise ValueError("후처리 설정 형식을 확인하세요.")
                updated = await asyncio.to_thread(store.save, patch)
                return web.json_response(
                    {"success": True, **store.public_payload(updated)},
                    headers={"Cache-Control": "no-store"},
                )
            return web.json_response(
                {"success": True, **store.public_payload()},
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def preview_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("미리보기 요청 형식을 확인하세요.")
            if workspace == ASSET_WORKSPACE:
                entity = str(body.get("character") or "").strip()
                level2 = str(body.get("outfit") or "").strip()
                level3 = str(body.get("expression") or "").strip()
            else:
                entity = str(body.get("theme") or "").strip()
                level2 = str(body.get("place") or "").strip()
                level3 = str(body.get("atmosphere") or body.get("time") or "").strip()
            filename = str(body.get("filename") or "").strip()
            if not all((entity, level2, level3, filename)):
                raise ValueError("미리보기할 이미지를 선택하세요.")
            if os.path.basename(filename) != filename or filename in (".", ".."):
                raise ValueError("이미지 파일명이 올바르지 않습니다.")

            filepath = mode.get_image_path(entity, level2, level3, filename)
            if not filepath or not os.path.isfile(filepath):
                return web.json_response(
                    {"success": False, "error": "미리보기할 이미지 파일을 찾을 수 없습니다."},
                    status=404,
                )

            raw_settings = body.get("settings")
            if raw_settings is None:
                effective = store.read()
            else:
                effective = normalize_postprocess_settings(
                    workspace,
                    raw_settings,
                    base=store.read(),
                    strict=True,
                )

            image_bytes, content_type, applied = await asyncio.to_thread(
                render_cached_postprocess_file,
                filepath,
                workspace,
                effective,
            )
            return web.Response(
                body=image_bytes,
                content_type=content_type,
                headers={
                    "Cache-Control": "no-store",
                    "X-Postprocess-Applied": "1" if applied else "0",
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[POSTPROCESS] 미리보기 오류: workspace={workspace}, error={type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response(
                {"success": False, "error": f"미리보기를 만들지 못했습니다: {exc}"},
                status=500,
            )

    server.app.router.add_get(prefix, settings_handler)
    server.app.router.add_post(prefix, settings_handler)
    server.app.router.add_post(f"{prefix}/preview", preview_handler)
    return store
