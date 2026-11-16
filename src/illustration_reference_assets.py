"""Reference-asset selection shared by server previews.

The ComfyUI custom node owns the image decoding and node execution; this helper
only mirrors the selected asset IDs for the application's flow previews.
"""
from __future__ import annotations


REFERENCE_REFS = ("body", "face")


def reference_stage_delivery(settings, stage, ref=None):
    key = "detailer_reference" if stage in {"face", "eye"} else "primary_reference"
    mode = settings.get(key) or settings.get("secondary_reference", "none")
    if mode == "llm":
        return "face_only" if ref == "face" else "none"
    return ("face_only" if stage in {"face", "eye"} else "none") if mode == "stage" else mode


def stage_reference_assets(payload, stage):
    """Select reference asset IDs for both execution and flow previews."""
    settings = payload["fields"].get("SNAPREF", {})
    if (not settings.get("enabled")
            or settings.get("strength_scale") == "linear" and float(settings.get("strength", 1.0)) <= 0):
        return {}
    grouped = {}
    for item in settings.get("references", []):
        grouped.setdefault(item["CHAR"], []).append(item)
    images = {}
    for name, refs in grouped.items():
        general, faces = [], []
        for index, item in enumerate(refs):
            # Older saved JSON represents a secondary face as a plain asset.
            is_face = item.get("role") == "face" or (index == 1 and "role" not in item
                       and settings.get("secondary_reference") == "face")
            if item.get("face_asset"):
                faces.append(item["face_asset"])
            (faces if is_face else general).append(item["asset"])
        mode = reference_stage_delivery(settings, stage, refs[0].get("ref"))
        if mode == "face_only":
            keys = faces[:1] if faces else general[:1]
        elif mode == "face":
            keys = general[:1] + faces[:1]
        elif mode == "representative":
            keys = general[:2]
        else:
            keys = general[:1]
        if keys:
            images[name] = list(dict.fromkeys(keys))
    return images
