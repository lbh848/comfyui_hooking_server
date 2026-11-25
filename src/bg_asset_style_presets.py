"""Background-only asset style presets and import helper from character asset styles."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from asset_style_presets import AssetStylePresets, StylePresetConflict
from preset_locations import PRESET_LOCATIONS

BG_STYLE_DEFAULT_WIDTH = 1280
BG_STYLE_DEFAULT_HEIGHT = 720

BG_STYLE_TAG_KEYS = {
    f"{prefix}{kind}_preset"
    for prefix in ("", "anima_") for kind in ("artist", "quality", "negative")
}
BG_STYLE_RAW_KEYS = {
    f"raw_{prefix}{kind}_tags"
    for prefix in ("", "anima_") for kind in ("artist", "quality", "negative")
}
BG_STYLE_BOOLEAN_KEYS = {
    "hrf_anima", "hrf_sdxl", "hrf_moss", "hrf_restore_size", "style_lora_enabled",
}
BG_STYLE_NUMBER_RANGES = {
    "img_w": (64, 8192), "img_h": (64, 8192),
    "seed": (-1, 9007199254740991), "hrf_size": (1, 8),
}
BG_STYLE_STRING_KEYS = BG_STYLE_TAG_KEYS | {"asset_model_preset_id", "asset_workflow_type"}
BG_STYLE_KEYS = (
    BG_STYLE_STRING_KEYS | BG_STYLE_RAW_KEYS | BG_STYLE_BOOLEAN_KEYS
    | BG_STYLE_NUMBER_RANGES.keys() | {"style_lora_list"}
)


def validate_bg_style_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"defaults", "values"}:
        raise ValueError("배경 그림체 기본값과 적용값을 확인하세요.")
    for slot in settings.values():
        if not isinstance(slot, dict) or set(slot) - BG_STYLE_KEYS:
            raise ValueError("배경 그림체 설정에 포함할 수 없는 항목이 있습니다.")
        for key, value in slot.items():
            valid = True
            if key in BG_STYLE_STRING_KEYS:
                valid = isinstance(value, str)
            elif key in BG_STYLE_RAW_KEYS:
                valid = isinstance(value, list) and all(isinstance(tag, str) for tag in value)
            elif key in BG_STYLE_BOOLEAN_KEYS:
                valid = isinstance(value, bool)
            elif key in BG_STYLE_NUMBER_RANGES:
                low, high = BG_STYLE_NUMBER_RANGES[key]
                valid = (type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                         and (key == "hrf_size" or value == int(value)))
            elif key == "style_lora_list":
                valid = isinstance(value, list) and all(isinstance(item, dict) for item in value)
            if not valid:
                raise ValueError(f"배경 그림체 설정 값을 확인하세요: {key}")
        if "asset_workflow_type" in slot and slot["asset_workflow_type"] not in {"anima_only", "anima_ilxl", "moss"}:
            raise ValueError("ANIMA, ANIMA+ILXL 또는 MOSS 그림 모델을 선택하세요.")
    json.dumps(settings, allow_nan=False)
    return copy.deepcopy(settings)


def extract_bg_style_from_asset_style(
    asset_style_settings: dict,
    *,
    keep_resolution: bool = False,
    default_width: int = BG_STYLE_DEFAULT_WIDTH,
    default_height: int = BG_STYLE_DEFAULT_HEIGHT,
) -> dict:
    """Copy only pure style settings from a character asset style preset."""
    if not isinstance(asset_style_settings, dict):
        raise ValueError("가져올 에셋 그림체 설정을 확인하세요.")
    result = {"defaults": {}, "values": {}}
    for section in ("defaults", "values"):
        source = asset_style_settings.get(section)
        if not isinstance(source, dict):
            continue
        result[section] = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key in BG_STYLE_KEYS
        }
    if not keep_resolution:
        result["values"]["img_w"] = default_width
        result["values"]["img_h"] = default_height
    else:
        result["values"].setdefault("img_w", default_width)
        result["values"].setdefault("img_h", default_height)
    return validate_bg_style_settings(result)


class BgAssetStylePresets(AssetStylePresets):
    def __init__(self, config_file: str | Path, builtin_file: str | Path | None = None):
        super().__init__(
            config_file,
            builtin_file=builtin_file or PRESET_LOCATIONS["bg_asset_style"].builtin_path(),
        )
        self.path = PRESET_LOCATIONS["bg_asset_style"].local_path(config_file=config_file)

    @staticmethod
    def _validate(settings):
        return validate_bg_style_settings(settings)


def register_bg_style_preset_runtime(server):
    from aiohttp import web

    store = BgAssetStylePresets(server.CONFIG_FILE)

    async def style_presets(request):
        try:
            result = store.change(await request.json()) if request.method == "POST" else store.catalog()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=409 if isinstance(exc, StylePresetConflict) else 400)

    server.app.router.add_get("/api/bg-asset/style-presets", style_presets)
    server.app.router.add_post("/api/bg-asset/style-presets", style_presets)
    return store
