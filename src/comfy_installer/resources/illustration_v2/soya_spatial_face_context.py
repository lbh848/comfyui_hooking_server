"""Assign detected faces by tracked geometry/RGB ownership, without identity models.

This is a new node. The legacy Face Match and FD/ED implementations are unchanged.
Coordinates are measured on the node's current image, including after hiresfix.
"""
import json
import math

import numpy as np
from PIL import Image
import torch


NODE_NAME = "SoyaSpatialFaceContextV2_mdsoya"


def normalized_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("face_positions.bbox must contain four normalized coordinates")
    box = [float(v) for v in value]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("face_positions.bbox must be finite and within 0..1")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("face_positions.bbox has no area")
    return box


def assign_tracks(boxes, tracks):
    """Name detections by closest tracked face centers, with a global 1:1 match.

    Tracking supplies identity only. Its boxes never become detailer crops.
    """
    from scipy.optimize import linear_sum_assignment

    if not boxes or not tracks:
        return []
    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
    targets = np.array([[(t["bbox"][0] + t["bbox"][2]) / 2,
                         (t["bbox"][1] + t["bbox"][3]) / 2] for t in tracks])
    distances = ((centers[:, None] - targets[None, :]) ** 2).sum(axis=2)
    rows, columns = linear_sum_assignment(distances)
    return [(int(i), tracks[j]["name"], 1.0 / (1.0 + float(distances[i, j])), "sampler")
            for i, j in zip(rows, columns)]


def parse_tracks(face_positions, names):
    tracks = json.loads(face_positions or "[]")
    if not isinstance(tracks, list):
        raise ValueError("face_positions must be an array")
    result = [{"name": t["name"], "bbox": normalized_box(t["bbox"])}
              for t in tracks if isinstance(t, dict) and t.get("name") in names]
    if len({t["name"] for t in result}) != len(result):
        raise ValueError("face_positions contains duplicate character identities")
    return result


def assign_faces(boxes, names, *, mask=None, mask_names=(), panels=()):
    """Fallback for paths with no sampler positions: geometric one-to-one match.

    Faces already assigned to a track never enter this function. RGB ownership and nearest mask
    pixels supply identity for old/no-tracking masks; there is no ambiguity
    cutoff. A tiny assignment DP prevents two faces claiming the same person.
    """
    selected = []
    for panel in panels or [{"x": 0, "y": 0, "width": 1, "height": 1}]:
        indices = [i for i, b in enumerate(boxes)
                   if panel["x"] <= (b[0] + b[2]) / 2 < panel["x"] + panel["width"]
                   and panel["y"] <= (b[1] + b[3]) / 2 < panel["y"] + panel["height"]]
        if not indices:
            continue
        if mask is None:
            if len(names) == 1:
                index = max(indices, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                selected.append((index, names[0], 1.0, "solo"))
            continue
        height, width = mask.shape[:2]
        regions = {}
        for channel, name in enumerate(mask_names):
            if name not in names:
                continue
            ys, xs = np.nonzero(mask[:, :, channel] > 0)
            px, py = (xs + .5) / width, (ys + .5) / height
            keep = ((px >= panel["x"]) & (px < panel["x"] + panel["width"])
                    & (py >= panel["y"]) & (py < panel["y"] + panel["height"]))
            if keep.any():
                regions[name] = (px[keep], py[keep], channel)
        states = {0: (0.0, [])}
        candidates = list(regions)
        for index in indices:
            box = boxes[index]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x0, y0, x1, y1 = math.floor(box[0] * width), math.floor(box[1] * height), math.ceil(box[2] * width), math.ceil(box[3] * height)
            costs = []
            for name in candidates:
                px, py, channel = regions[name]
                coverage = float(mask[y0:y1, x0:x1, channel].mean())
                distance = float(((px - cx) ** 2 + (py - cy) ** 2).min())
                costs.append(distance - coverage)
            updated = dict(states)
            for used, (cost, pairs) in states.items():
                for identity, name in enumerate(candidates):
                    if used & (1 << identity):
                        continue
                    key, total = used | (1 << identity), cost + costs[identity]
                    if key not in updated or total < updated[key][0]:
                        updated[key] = (total, [*pairs, (index, name, 1.0 / (1 + max(0, costs[identity])), "mask")])
            states = updated
        best = min(states, key=lambda key: (-key.bit_count(), states[key][0]))
        selected.extend(states[best][1])
    return sorted(selected)


def make_context(boxes, assignments, width, height, crop_top, crop_bottom):
    """Keep the legacy detector-box crop expansion and integer rounding exactly."""
    context = {"matches": [], "ipa_embeds": {}, "img_H": height, "img_W": width}
    for index, name, score, method in assignments:
        box = boxes[index]
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = (crop_top + crop_bottom) / 2
        crop = [max(0, int(cx - (x1 - x0) * side / 2)),
                max(0, int(cy - (y1 - y0) * crop_top / 2)),
                min(width, int(cx + (x1 - x0) * side / 2)),
                min(height, int(cy + (y1 - y0) * crop_bottom / 2))]
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            continue
        context["matches"].append({"name": name, "bbox": [float(v) for v in box], "crop": crop,
                                   "score": score, "method": method})
    return context


class SoyaSpatialFaceContextV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "bbox_detector": ("BBOX_DETECTOR",),
            "prompt_json": ("STRING", {"forceInput": True}),
            "face_positions": ("STRING", {"forceInput": True}),
            "max_face_count": ("INT", {"default": 5, "min": 1, "max": 20}),
            "yolo_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            "debug": ("BOOLEAN", {"default": False}),
        }}

    # Preserve the existing context/eye-tag fanout without modifying those nodes.
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IPA_FACE_CONTEXT")
    RETURN_NAMES = ("detected_faces", "named_faces", "names", "info", "face_context")
    OUTPUT_IS_LIST = (True, True, True, False, False)
    FUNCTION = "build"
    CATEGORY = "Soya/FaceDetailer"

    def build(self, image, prompt_json, face_positions, bbox_detector,
              max_face_count=5, yolo_confidence=0.5, debug=False):
        from .soya_illustration_json_v2 import read_payload, image_tensor

        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Spatial face context requires one image per queue item")
        payload = read_payload(prompt_json)
        fields = payload["fields"]
        names = fields.get("CHAR_LIST", [])
        if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("CHAR_LIST must contain character names")
        names = list(dict.fromkeys(names))
        tracks = parse_tracks(face_positions, names)
        missing_names = [n for n in names if n not in {t["name"] for t in tracks}]
        multi = fields.get("MULTI_CHAR", {})
        mask_names = multi.get("char_name_list", []) if multi.get("enable") else []
        if mask_names and (len(mask_names) > 3 or len(set(mask_names)) != len(mask_names)):
            raise ValueError("MULTI_CHAR must identify one unique character per RGB channel")
        mask = image_tensor(payload, fields["REGION_MASK"])[0].numpy() if missing_names and mask_names and fields.get("REGION_MASK") else None
        panels = multi.get("panels", [])
        top, bottom = float(fields.get("FACE_CROP_TOP", 3.0)), float(fields.get("FACE_CROP_BOTTOM", 3.0))
        if not all(math.isfinite(v) and 0.1 <= v <= 10 for v in (top, bottom)):
            raise ValueError("FACE_CROP_TOP/BOTTOM must be within 0.1..10")
        del payload, fields
        height, width = image.shape[1:3]
        # Preserve the old YOLO call, confidence filter and largest-face limit.
        # Only CLIP/cache identity matching is replaced, not face detection/crop.
        pixels = (image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        boxes = []
        for prediction in bbox_detector.bbox_model(pixels, verbose=False):
            if prediction.boxes is None:
                continue
            for detection in prediction.boxes:
                if float(detection.conf[0]) < yolo_confidence:
                    continue
                box = detection.xyxy[0].cpu().numpy()
                if np.isfinite(box).all() and box[0] < box[2] and box[1] < box[3]:
                    boxes.append(box)
        total_detected = len(boxes)
        if len(boxes) > max_face_count:
            largest = sorted(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), reverse=True)
            boxes = [boxes[i] for i in sorted(largest[:max_face_count])]
        normalized = [[float(v) / (width if axis % 2 == 0 else height) for axis, v in enumerate(box)] for box in boxes]
        assignments = assign_tracks(normalized, tracks)
        assigned_indices = {a[0] for a in assignments}
        remaining = [i for i in range(len(boxes)) if i not in assigned_indices]
        fallback = assign_faces([normalized[i] for i in remaining], missing_names,
                                mask=mask, mask_names=mask_names, panels=panels)
        assignments.extend((remaining[i], name, score, method) for i, name, score, method in fallback)
        assignments.sort()
        context = make_context(boxes, assignments, width, height, top, bottom)
        info = f"Spatial faces: detected={total_detected}, kept={len(boxes)}, assigned={len(context['matches'])}"
        print(f"[Soya:SpatialFace] {info}")
        if not debug:
            return [], [], [], info, context
        pil_image = Image.fromarray(pixels)
        def crop_tensor(crop):
            return torch.from_numpy(np.array(pil_image.crop(crop)).astype(np.float32) / 255.0).unsqueeze(0)
        all_context = make_context(boxes, [(i, "unknown", 0, "") for i in range(len(boxes))], width, height, top, bottom)
        detected = [crop_tensor(m["crop"]) for m in all_context["matches"]]
        named = [crop_tensor(m["crop"]) for m in context["matches"]]
        return detected, named, [m["name"] for m in context["matches"]], info, context
