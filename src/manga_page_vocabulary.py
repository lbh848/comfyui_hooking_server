"""Closed vocabulary for DETAIL-MANGA layered page planning.

The model selects small enum values; the server turns them into the concrete
MOSS captions. Keeping the wording here means the planning response stays
short and the parser is the only place that knows the projected phrases.
"""

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
BALLOON = ("oval", "angular", "narration", "thought", "trembling", "shout",
           "explicit", "soft")
BALLOON_TAIL = ("down", "up", "left", "right", "none")

SHAPE_PHRASES = {
    "regular": "rectangular",
    "wide": "wide rectangular",
    "tall": "tall vertical rectangular",
    "inset": "inset rectangular frame",
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
_BALLOON_FINISH = ("variable-width pen strokes, fine ink grain, "
                   "off-white fill at 90% opacity")
BALLOON_PHRASES = {
    # Keep the existing enum for saved plans; describe a drawn contour rather
    # than a geometric ellipse. Ink texture alone leaves the silhouette smooth.
    "oval": "asymmetric freehand speech balloon, uneven rounded contour, " + _BALLOON_FINISH,
    "angular": "angular freehand balloon, " + _BALLOON_FINISH,
    "narration": "hand-drawn rectangular text box, " + _BALLOON_FINISH,
    "thought": ("uneven cloud thought balloon with a small circle trail, "
                + _BALLOON_FINISH),
    "trembling": "trembling freehand balloon, broken wavy contour, " + _BALLOON_FINISH,
    "shout": "uneven hand-drawn starburst balloon, " + _BALLOON_FINISH,
    "explicit": "soft torn freehand balloon, " + _BALLOON_FINISH,
    "soft": "soft asymmetric freehand balloon, uneven rounded contour, " + _BALLOON_FINISH,
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


def balloon_caption(balloon: str, tail: str) -> str:
    """Compose the deterministic MOSS speech-balloon caption."""
    balloon = normalize_choice(balloon, BALLOON, "oval", "balloon")
    tail = normalize_choice(tail, BALLOON_TAIL, "none", "tail")
    caption = (BALLOON_PHRASES[balloon]
               + ", wider than tall, horizontal left-to-right text")
    if balloon in _TAILED_BALLOONS and tail != "none":
        caption += f", tail pointing {tail}"
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
            # New plans count primary page divisions. Insets remain supported
            # by the parser for saved plans, but are not offered as main shots.
            "shape": [shape for shape in PANEL_SHAPE if shape != "inset"],
            "camera_distance": list(PANEL_CAMERA_DISTANCE),
            "camera_angle": list(PANEL_CAMERA_ANGLE),
            "focus": list(PANEL_FOCUS),
            "edge": list(PANEL_EDGE),
        },
        "cross_panel": {"relation": list(ACTOR_PANEL_RELATION)},
        "balloon": {"shape": list(BALLOON), "tail": list(BALLOON_TAIL)},
    }
