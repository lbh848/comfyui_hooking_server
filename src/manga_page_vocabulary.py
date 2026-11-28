"""Closed vocabulary for DETAIL-MANGA layered page planning.

The model selects small enum values; the server turns them into the concrete
MOSS captions. Keeping the wording here means the planning response stays
short and the parser is the only place that knows the projected phrases.
"""

import math
import re

PAGE_HIERARCHY = ("balanced", "asymmetric", "dominant_panel")
PAGE_FLOW = ("linear", "diagonal", "layered", "radial")
PAGE_BOUNDARIES = ("touching", "separated", "overlapping", "mixed")
SHARED_DIVIDER_PREFIX = "A single solid black"
SHARED_DIVIDER_NEGATIVE = ("white gutters", "gaps between panels",
                           "double panel borders", "white page margins")
BOUNDARY_PHRASES = {
    "touching": ("Full color manga, {panel_count} abutting borderless image regions fill one page. "
                 "Adjoining views meet directly at a single black ink line."),
    "separated": "Panels are separated by deliberate white gutters.",
    "overlapping": "Panel frames overlap with clear foreground order.",
    "mixed": "Use shared dividers and selective overlaps; gaps only at planned pauses.",
}
PANEL_SHAPE = ("regular", "wide", "tall", "inset", "irregular", "diagonal",
               "borderless")
PANEL_EDGE = ("top", "right", "bottom", "left")
PANEL_CAMERA_DISTANCE = ("extreme_close_up", "close_up", "medium", "wide")
PANEL_CAMERA_ANGLE = ("eye_level", "high_angle", "low_angle", "side_view",
                      "overhead", "dutch_angle")
PANEL_FOCUS = ("deep_focus", "shallow_focus", "selective_focus")
ACTOR_PANEL_RELATION = ("breaks_border", "spans_adjacent_panels",
                        "page_foreground_overlay")
INTERACTION_SPATIAL = (
    "above_on_top",
    "straddling_on_top",
    "behind",
    "in_front",
    "beside_close",
    "face_to_face_close",
    "below_underneath",
    "carrying_lifting",
    "separated",
)
INTERACTION_SPATIAL_TEMPLATES = {
    "above_on_top": "{source} is positioned directly above and on top of {target} underneath",
    "straddling_on_top": "{source} is straddling on top of {target} underneath",
    "behind": "{source} is positioned directly behind {target}",
    "in_front": "{source} is positioned in the foreground directly in front of {target}",
    "beside_close": "{source} and {target} are pressed close side by side",
    "face_to_face_close": "{source} and {target} are positioned close face to face",
    "below_underneath": "{source} is positioned below {target}",
    "carrying_lifting": "{source} is lifting and supporting {target} off the ground",
    "separated": "{source} and {target} are separated across the frame",
}
_ACTOR_GARMENT_CUE_RE = re.compile(
    r"\b("
    r"(?:(?:white|black|dark|light|red|blue|navy|green|gray|grey|brown|pink|beige|cream|yellow|purple|plaid|striped)\s+){1,2}"
    r"(?:t[\s-]?shirt|shirt|blouse|sweater|vest|cardigan|hoodie|jacket|blazer|coat|dress|skirt|pants|jeans|shorts|uniform|kimono|robe|apron|bra|panties)"
    r"|completely\s+nude|nude|topless|shirtless|bottomless"
    r")\b",
    re.IGNORECASE,
)
BALLOON = ("oval", "angular", "narration", "thought", "trembling", "shout",
           "explicit", "soft")
BALLOON_TAIL = (
    "down", "up", "left", "right",
    "down_left", "down_right", "up_left", "up_right",
    "none",
)
BALLOON_TAIL_PHRASES = {
    "down": "down",
    "up": "up",
    "left": "left",
    "right": "right",
    "down_left": "down-left",
    "down_right": "down-right",
    "up_left": "up-left",
    "up_right": "up-right",
}
_TAIL_COMPASS = (
    "left", "up_left", "up", "up_right",
    "right", "down_right", "down", "down_left",
)
_ACTOR_GENDER_RE = re.compile(
    r"\b(1girl|1female|1boy|1male|2girls|2boys|multiple girls|multiple boys)\b",
    re.IGNORECASE,
)
_ACTOR_GENDER_NOUN = {
    "1girl": "girl",
    "1female": "woman",
    "1boy": "boy",
    "1male": "man",
    "2girls": "girls",
    "2boys": "boys",
    "multiple girls": "girls",
    "multiple boys": "boys",
}
_ACTOR_NATURAL_GENDER_RE = re.compile(
    r"\b(old\s+women|old\s+woman|old\s+men|old\s+man|girls|girl|women|woman|boys|boy|men|man|people|person)\b",
    re.IGNORECASE,
)
_ACTOR_HAIR_RE = re.compile(
    r"\b("
    r"(?:(?:light|dark|pale|deep|bright|pastel|ash|golden|platinum|strawberry|"
    r"chestnut|auburn|ginger|sandy|jet|raven|snow|crimson|emerald|indigo|"
    r"lavender|magenta|cyan)\s+)?"
    r"(?:blonde|blond|black|brown|white|gray|grey|silver|red|blue|green|"
    r"pink|purple|violet|orange|yellow|aqua|teal)"
    r"\s+hair)\b",
    re.IGNORECASE,
)

SHAPE_PHRASES = {
    "regular": "rectangular",
    "wide": "wide rectangular",
    "tall": "tall vertical rectangular",
    "inset": "superimposed inset frame",
    "irregular": "irregular quadrilateral",
    "diagonal": "trapezoid with a slanted edge",
    "borderless": "borderless",
}
CAMERA_DISTANCE_PHRASES = {
    "extreme_close_up": "extreme close-up",
    "close_up": "close-up",
    "medium": "medium shot",
    # "wide shot" is forbidden by the project camera policy.
    "wide": "wide framing",
}
CAMERA_ANGLE_PHRASES = {
    "eye_level": "eye level",
    "high_angle": "high angle",
    "low_angle": "low angle",
    "side_view": "side view",
    "overhead": "overhead view",
    "dutch_angle": "dutch angle",
}
FOCUS_PHRASES = {
    "deep_focus": "deep focus",
    "shallow_focus": "shallow focus",
    "selective_focus": "selective focus",
}
_BALLOON_FINISH = (
    "translucent off-white fill with the underlying background faintly visible "
    "through it, variable-width pen strokes, fine ink grain"
)
BALLOON_PHRASES = {
    # Keep the existing enum for saved plans; describe a drawn contour rather
    # than a geometric ellipse. Ink texture alone leaves the silhouette smooth.
    "oval": "semi-transparent asymmetric freehand speech balloon, uneven rounded contour, " + _BALLOON_FINISH,
    "angular": "semi-transparent angular freehand balloon, " + _BALLOON_FINISH,
    "narration": "semi-transparent hand-drawn rectangular text box, " + _BALLOON_FINISH,
    "thought": ("semi-transparent uneven cloud thought balloon with a small circle trail, "
                + _BALLOON_FINISH),
    "trembling": "semi-transparent trembling freehand balloon, broken wavy contour, " + _BALLOON_FINISH,
    "shout": "semi-transparent uneven hand-drawn starburst balloon, " + _BALLOON_FINISH,
    "explicit": "semi-transparent soft torn freehand balloon, " + _BALLOON_FINISH,
    "soft": "semi-transparent soft asymmetric freehand balloon, uneven rounded contour, " + _BALLOON_FINISH,
}
_TAILED_BALLOONS = {"oval", "angular", "trembling", "shout", "explicit", "soft"}


def normalize_choice(value, allowed: tuple[str, ...], default: str, field: str) -> str:
    """Return a canonical enum value, falling back to ``default`` with a log."""
    candidate = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if candidate in allowed:
        return candidate
    if candidate:
        print(f"[MANGA_VOCAB] 알 수 없는 {field} 값, 기본값 사용: "
              f"value={value!r}, default={default!r}")
    return default


def panel_caption(number: int, panel: dict, *, shape: str,
                  camera_distance: str, camera_angle: str, focus: str,
                  shared_dividers: bool = False) -> str:
    """Compose the deterministic MOSS panel caption from the enums."""
    shape_phrase = SHAPE_PHRASES.get(shape, SHAPE_PHRASES["regular"])
    border = "no border, no gutter" if shape == "borderless" else "thin black border"
    camera = ", ".join(part for part in (
        CAMERA_DISTANCE_PHRASES.get(camera_distance, CAMERA_DISTANCE_PHRASES["medium"]),
        CAMERA_ANGLE_PHRASES.get(camera_angle, CAMERA_ANGLE_PHRASES["eye_level"]),
        FOCUS_PHRASES.get(focus, FOCUS_PHRASES["deep_focus"]),
    ) if part)
    instruction = str(panel.get("instruction") or "").strip()
    if shared_dividers and shape not in {"borderless", "inset"}:
        # A closed border on EVERY panel encourages two strokes with a white
        # strip between them, even when their model-authored bboxes touch.
        parts = [f"PANEL {number}. REGION: {shape_phrase}, edge-to-edge artwork "
                 "bounded by shared dividing strokes."]
    else:
        parts = [f"PANEL {number}. FRAME: {shape_phrase}, {border}."]
    if camera:
        parts.append(camera + ".")
    if instruction:
        parts.append(instruction)
    return " ".join(parts)


def _clip_bbox_to_panel(actor_bbox, panel_bbox):
    if not panel_bbox or len(panel_bbox) != 4:
        return actor_bbox
    y0 = max(actor_bbox[0], panel_bbox[0])
    x0 = max(actor_bbox[1], panel_bbox[1])
    y1 = min(actor_bbox[2], panel_bbox[2])
    x1 = min(actor_bbox[3], panel_bbox[3])
    if y1 > y0 and x1 > x0:
        return [y0, x0, y1, x1]
    return actor_bbox


def _actor_head_anchor(bbox, camera_distance: str = "medium") -> tuple[float, float]:
    y0, x0, y1, x1 = (float(value) for value in bbox)
    distance = str(camera_distance or "medium").strip().lower().replace("-", "_").replace(" ", "_")
    head_ratio = 0.40 if distance in ("extreme_close_up", "close_up") else 0.22
    return (x0 + x1) / 2.0, y0 + (y1 - y0) * head_ratio


def _point_in_bbox(point: tuple[float, float], bbox) -> bool:
    return (float(bbox[1]) <= point[0] <= float(bbox[3])
            and float(bbox[0]) <= point[1] <= float(bbox[2]))


def resolve_balloon_tail(
    balloon_bbox,
    speaker_bbox,
    *,
    fallback_tail: str = "none",
    camera_distance: str = "medium",
    panel_bbox=None,
) -> str:
    """Compute an 8-direction balloon tail from the balloon to the speaker's head."""
    fallback = normalize_choice(fallback_tail, BALLOON_TAIL, "none", "tail")
    if not balloon_bbox or not speaker_bbox:
        return fallback
    clipped = _clip_bbox_to_panel(speaker_bbox, panel_bbox)
    bx = (float(balloon_bbox[1]) + float(balloon_bbox[3])) / 2.0
    by = (float(balloon_bbox[0]) + float(balloon_bbox[2])) / 2.0
    anchor = _actor_head_anchor(clipped, camera_distance)
    if _point_in_bbox(anchor, balloon_bbox):
        anchor = (
            (float(clipped[1]) + float(clipped[3])) / 2.0,
            (float(clipped[0]) + float(clipped[2])) / 2.0,
        )
    dx = anchor[0] - bx
    dy = anchor[1] - by
    if math.hypot(dx, dy) < 16.0:
        return fallback
    angle = math.degrees(math.atan2(dy, dx))
    return _TAIL_COMPASS[int(math.floor((angle + 202.5) / 45.0)) % 8]


def describe_actor_visual(instruction: str) -> str:
    """Extract a short visual noun phrase from an actor instruction without internal IDs."""
    text = str(instruction or "")
    gender_match = _ACTOR_GENDER_RE.search(text)
    if gender_match:
        noun = _ACTOR_GENDER_NOUN.get(gender_match.group(1).lower(), "character")
    else:
        natural_match = _ACTOR_NATURAL_GENDER_RE.search(text)
        noun = " ".join(natural_match.group(1).lower().split()) if natural_match else "character"
    hair_match = _ACTOR_HAIR_RE.search(text)
    if hair_match:
        return f"the {noun} with {hair_match.group(1).lower()}"
    return f"the {noun}"


def _relative_actor_position(
    actor_bbox,
    other_bboxes: list,
    *,
    panel_bbox=None,
    camera_distance: str = "medium",
) -> str:
    """Return a concise relative screen-position phrase comparing one actor against others."""
    if not actor_bbox or not other_bboxes:
        return ""
    actor_x, actor_y = _actor_head_anchor(
        _clip_bbox_to_panel(actor_bbox, panel_bbox), camera_distance)
    other_points = [
        _actor_head_anchor(_clip_bbox_to_panel(box, panel_bbox), camera_distance)
        for box in other_bboxes
    ]
    if panel_bbox and len(panel_bbox) == 4:
        panel_w = max(1.0, float(panel_bbox[3]) - float(panel_bbox[1]))
        panel_h = max(1.0, float(panel_bbox[2]) - float(panel_bbox[0]))
    else:
        all_boxes = [actor_bbox, *other_bboxes]
        panel_w = max(1.0, max(float(b[3]) for b in all_boxes) - min(float(b[1]) for b in all_boxes))
        panel_h = max(1.0, max(float(b[2]) for b in all_boxes) - min(float(b[0]) for b in all_boxes))
    margin_x = max(32.0, 0.12 * panel_w)
    margin_y = max(32.0, 0.12 * panel_h)
    other_xs = [point[0] for point in other_points]
    other_ys = [point[1] for point in other_points]
    if actor_x < min(other_xs) - margin_x:
        return "on the left"
    if actor_x > max(other_xs) + margin_x:
        return "on the right"
    if len(other_xs) >= 2 and min(other_xs) + margin_x < actor_x < max(other_xs) - margin_x:
        return "in the center"
    if actor_y < min(other_ys) - margin_y:
        return "at the top"
    if actor_y > max(other_ys) + margin_y:
        return "at the bottom"
    return ""


def _extract_actor_garment_cue(instruction: str) -> str:
    match = _ACTOR_GARMENT_CUE_RE.search(str(instruction or ""))
    if not match:
        return ""
    cue = " ".join(match.group(1).lower().split())
    if cue in {"completely nude", "nude", "topless", "shirtless", "bottomless"}:
        return cue
    return f"in the {cue}"


def describe_distinct_actors(
    panel_actors: list[dict],
    *,
    panel_bbox=None,
    camera_distance: str = "medium",
) -> dict[str, str]:
    """Map actor names (casefolded) to collision-safe visual descriptions without internal IDs.

    Escalates from gender + hair color -> distinguishing garment/coverage cue ->
    relative bbox screen position whenever two visible actors share the same phrase.
    """
    entries = [
        actor for actor in (panel_actors or [])
        if isinstance(actor, dict) and str(actor.get("name") or "").strip()
    ]
    if not entries:
        return {}
    labels = {
        str(actor["name"]).strip().casefold(): describe_actor_visual(actor.get("instruction") or "")
        for actor in entries
    }
    counts: dict[str, int] = {}
    for phrase in labels.values():
        counts[phrase] = counts.get(phrase, 0) + 1
    if all(count == 1 for count in counts.values()):
        return labels

    # Fallback 1: add a distinguishing garment/coverage cue for colliding entries.
    garment_candidates = {}
    for actor in entries:
        key = str(actor["name"]).strip().casefold()
        base = labels[key]
        if counts.get(base, 0) > 1:
            cue = _extract_actor_garment_cue(actor.get("instruction") or "")
            garment_candidates[key] = f"{base} {cue}" if cue else base
        else:
            garment_candidates[key] = base
    garment_counts: dict[str, int] = {}
    for phrase in garment_candidates.values():
        garment_counts[phrase] = garment_counts.get(phrase, 0) + 1
    for key, candidate in garment_candidates.items():
        if garment_counts.get(candidate, 0) == 1:
            labels[key] = candidate

    # Fallback 2: add relative bbox position for any remaining collision.
    final_counts: dict[str, int] = {}
    for phrase in labels.values():
        final_counts[phrase] = final_counts.get(phrase, 0) + 1
    for actor in entries:
        key = str(actor["name"]).strip().casefold()
        if final_counts.get(labels[key], 0) <= 1:
            continue
        other_boxes = [
            other["bbox"] for other in entries
            if other is not actor and other.get("bbox")
        ]
        pos = _relative_actor_position(
            actor.get("bbox"),
            other_boxes,
            panel_bbox=panel_bbox,
            camera_distance=camera_distance,
        )
        if pos:
            labels[key] = f"{labels[key]} {pos}"
    return labels


def compile_interaction_clauses(
    interactions,
    panel_actors: list[dict],
    *,
    panel_bbox=None,
    camera_distance: str = "medium",
    existing_text: str = "",
) -> str:
    """Compile optional structured interactions into collision-safe MOSS spatial/occlusion prose.

    Resolves source/target character names strictly to visual descriptions so
    internal character IDs never leak into MOSS captions.
    """
    if not isinstance(interactions, list) or not interactions:
        return ""
    distinct = describe_distinct_actors(
        panel_actors, panel_bbox=panel_bbox, camera_distance=camera_distance)
    existing_lower = str(existing_text or "").lower()
    sentences: list[str] = []
    for entry in interactions:
        if not isinstance(entry, dict):
            continue
        raw_source = str(entry.get("source") or "").strip().strip('"“”').strip()
        raw_target = str(entry.get("target") or "").strip().strip('"“”').strip()
        source_label = distinct.get(raw_source.casefold(), "") if raw_source else ""
        target_label = distinct.get(raw_target.casefold(), "") if raw_target else ""
        action = str(entry.get("action") or "").strip().rstrip(".")
        mode = str(entry.get("mode") or "directed").strip().lower()
        raw_spatial = str(entry.get("spatial") or "").strip().lower().replace("-", "_").replace(" ", "_")
        spatial = raw_spatial if raw_spatial in INTERACTION_SPATIAL_TEMPLATES else ""
        contact = str(entry.get("contact") or "").strip().rstrip(".")
        occludes = str(entry.get("occludes") or "").strip().rstrip(".")

        clauses: list[str] = []
        concise_action = action if (action and len(action.split()) <= 4) else ""
        if spatial and source_label and target_label:
            base = INTERACTION_SPATIAL_TEMPLATES[spatial].format(
                source=source_label, target=target_label)
            if concise_action and concise_action.lower() not in base.lower() and concise_action.lower() not in existing_lower:
                base = f"{base}, {concise_action}"
            clauses.append(base)
        elif action:
            if source_label and target_label:
                if mode == "mutual":
                    clauses.append(f"{source_label} and {target_label} engage in {action}")
                elif concise_action:
                    clauses.append(f"{source_label} acts on {target_label}, {concise_action}")
                elif action.lower() not in existing_lower:
                    clauses.append(action)
            elif source_label:
                clauses.append(f"{source_label} performs {action}" if concise_action else action)
            elif action.lower() not in existing_lower:
                clauses.append(action)
        if contact and contact.lower() not in existing_lower:
            clauses.append(contact)
        if occludes and occludes.lower() not in existing_lower:
            clauses.append(occludes)
        if not clauses:
            continue
        # Scrub registered actor IDs accidentally echoed in contact/occludes strings
        # without clobbering single-letter English words (e.g., "a") or short stopwords.
        _stop_names = {
            "an", "in", "on", "to", "of", "at", "by", "he", "it", "is",
            "as", "or", "no", "up", "so", "we", "me", "my", "do", "go", "be",
        }
        combined = "; ".join(clauses)
        for actor in panel_actors or []:
            actor_name = str(actor.get("name") or "").strip()
            replacement = distinct.get(actor_name.casefold())
            if (
                not actor_name
                or not replacement
                or len(actor_name) < 2
                or actor_name.casefold() in _stop_names
            ):
                continue
            flags = re.IGNORECASE if (len(actor_name) >= 3 or "_" in actor_name) else 0
            combined = re.sub(rf"\b{re.escape(actor_name)}\b", replacement, combined, flags=flags)
        sentence = combined[0].upper() + combined[1:] + "."
        if sentence.lower() not in existing_lower and sentence not in sentences:
            sentences.append(sentence)
    return " ".join(sentences)


def describe_speaker_target(
    speaker_actor: dict,
    panel_actors: list[dict],
    *,
    panel_bbox=None,
    camera_distance: str = "medium",
) -> str:
    """Describe the visible speaker visually and, in multi-character shots, by relative position."""
    base = describe_actor_visual(speaker_actor.get("instruction") or "")
    speaker_bbox = speaker_actor.get("bbox")
    other_bboxes = [
        actor["bbox"] for actor in panel_actors
        if actor is not speaker_actor and actor.get("bbox")
    ]
    if not speaker_bbox or not other_bboxes:
        return base
    position = _relative_actor_position(
        speaker_bbox,
        other_bboxes,
        panel_bbox=panel_bbox,
        camera_distance=camera_distance,
    )
    return f"{base} {position}" if position else base


def balloon_caption(balloon: str, tail: str, *, target: str = "") -> str:
    """Compose the deterministic MOSS speech-balloon caption."""
    balloon = normalize_choice(balloon, BALLOON, "oval", "balloon")
    tail = normalize_choice(tail, BALLOON_TAIL, "none", "tail")
    caption = (BALLOON_PHRASES[balloon]
               + ", wider than tall, horizontal left-to-right text")
    direction = BALLOON_TAIL_PHRASES.get(tail, "")
    target_suffix = f" toward {target}" if target else ""
    if balloon in _TAILED_BALLOONS and direction:
        caption += f", tail pointing {direction}{target_suffix}"
    elif balloon == "thought" and direction:
        caption += f", circle trail pointing {direction}{target_suffix}"
    elif balloon == "narration":
        caption += ", no tail"
    return caption + "."


def vocabulary() -> dict:
    """JSON-serializable enum snapshot for the prompt user message."""
    return {
        "page": {
            "hierarchy": list(PAGE_HIERARCHY),
            "flow": list(PAGE_FLOW),
            "boundaries": list(PAGE_BOUNDARIES),
        },
        "panel": {
            # A framed inset is a complete shot on a layered borderless page.
            "shape": list(PANEL_SHAPE),
            "camera_distance": list(PANEL_CAMERA_DISTANCE),
            "camera_angle": list(PANEL_CAMERA_ANGLE),
            "focus": list(PANEL_FOCUS),
            "edge": list(PANEL_EDGE),
            "interaction_spatial": list(INTERACTION_SPATIAL),
        },
        "cross_panel": {"relation": list(ACTOR_PANEL_RELATION)},
        "balloon": {"shape": list(BALLOON), "tail": list(BALLOON_TAIL)},
    }
