"""Project shared illustration instructions onto the single-image MOSS schema.

Only instruction text passes through here. Profiles, source passages and other
user-message data retain their original keys and values.
"""
from modes.illustration_prompt_assembly import prompt_sub

MOSS_DETAIL_PROMPT_MODES = ("tags", "natural")
DEFAULT_MOSS_DETAIL_PROMPT_MODE = "tags"


def normalize_moss_detail_prompt_mode(value) -> str:
    """Return a canonical MOSS DETAIL prompt mode ('tags' or 'natural')."""
    raw = str(value or "").strip().lower()
    if raw in MOSS_DETAIL_PROMPT_MODES:
        return raw
    return DEFAULT_MOSS_DETAIL_PROMPT_MODE


def is_moss_natural_detail(toggles=None) -> bool:
    """Return True when MOSS DETAIL should generate natural-language sentences."""
    return normalize_moss_detail_prompt_mode(
        (toggles or {}).get("moss_detail_prompt_mode")
    ) == "natural"


def select_detail_prompt_key(*, moss: bool, moss_manga: bool, composite: bool, toggles=None) -> str:
    """Single source of truth for selecting the DETAIL prompt file key."""
    natural = is_moss_natural_detail(toggles)
    if moss_manga:
        return "detail_manga_natural" if natural else "detail_manga"
    if moss:
        return "detail_moss_natural" if natural else "detail_moss"
    if composite:
        return "composite_detail"
    return "call2_detail"


def _apply_natural_detail_projection(text, *, container_field: str = "`summary.caption`"):
    """Project shared explicit/keyvis/bot-policy tag-forcing wording onto natural-language English prose."""
    for old, new in (
        (
            f"use {container_field} only for directional contact, overlap, occlusion, or off-frame continuation that tags cannot express clearly.",
            f"use {container_field} for shared directional contact, overlap, occlusion, or off-frame continuation.",
        ),
        (
            "POSE AND ACTION TAG REFERENCE",
            "POSE AND ACTION TERMINOLOGY REFERENCE",
        ),
        (
            "When producing image-prompt tags, consult the table when the source moment contains a corresponding pose or action. "
            "Match its meaning and actor/receiver roles from the full context, even when the source uses different words. "
            "Preserve the applicable conventional pose/action terms, including colloquial or opaque terms such as `doggystyle`, `full nelson` and `paizuri`; "
            "unfamiliarity or slang is not a reason to omit them, interpret them literally, or replace them with generic `sitting` or `lying`.",
            "When writing natural-language image descriptions, consult the table when the source moment contains a corresponding pose or action. "
            "Match its meaning and actor/receiver roles from the full context, even when the source uses different words. "
            "Express the applicable conventional pose or action clearly in fluent natural-language English sentences using concrete, recognizable terms—"
            "including established position names such as doggystyle, full nelson and paizuri when supported—rather than replacing them with generic sitting or lying.",
        ),
        (
            "Rows are candidate vocabulary, not fixed tag bundles. When a tag's meaning is clear and its posture conflicts with the actual scene, "
            "freely omit it or replace it with context-appropriate wording: for example, `leaning back` need not remain when the person is upright or leaning forward.",
            "Rows are reference vocabulary, not raw tag lists to copy verbatim. Weave only the supported actions and compatible posture details into fluent "
            "natural-language English sentences, omitting or adjusting any posture cue that conflicts with the actual scene (for example, do not write that the "
            "person is leaning back when they are upright or leaning forward).",
        ),
        (
            "When the source establishes an explicit anatomy or fluid fact, write the matching English image tag instead of a euphemism. "
            "These pairs are translation, not optional suggestions:",
            "When the source establishes an explicit anatomy, coverage or fluid fact, express it directly in fluent natural-language English using the literal "
            "terms below instead of euphemisms. These pairs are direct meaning translations to weave into sentences, not raw tags to list:",
        ),
        (
            "A caption for an explicit act must contain the applicable coverage tag and at least one literal act or anatomy tag. "
            "`between her thighs`, `scooping`, `receiving him deep` and `hips pressed` do not replace those tags; the image model cannot infer the act from them.",
            "A description of an explicit act must explicitly state the applicable coverage state and at least one literal act or anatomical term in plain natural English. "
            "Vague euphemisms such as \"between her thighs\", \"scooping\", \"receiving him deep\" and \"hips pressed\" do not replace literal terms; "
            "the image model cannot infer the act from them.",
        ),
        (
            f"Add `nsfw` exactly once in {container_field}.",
            f"Do not add a raw `nsfw` rating tag to {container_field}; convey the explicit content directly through literal natural-language visual descriptions.",
        ),
        (
            "Do not move exclusion lists or negated instructions into positive tags or composition prose.",
            "Do not move exclusion lists or negated instructions into character captions or composition prose.",
        ),
    ):
        text = text.replace(old, new)
    text = text.replace("positive tags", "character captions")
    text = text.replace("compact tags", "natural-language English sentences")
    return text


def moss_manga_explicit_fields(text, *, natural: bool = False):
    """Project shared explicit.txt rules onto the DETAIL-MANGA layered panel schema."""
    if not text:
        return text
    for old, new in (
        ("characters[].positive", "actors[].instruction"),
        ("characters[n].positive", "actors[n].instruction"),
        ("characters[]", "actors[]"),
        ("characters[n]", "actors[n]"),
        ("`positive`", "`actors[].instruction`"),
        ("`camera`", "`panel.instruction`"),
        ("`scene`", "`panel.instruction`"),
        ("`supplement`", "`panel.instruction`"),
        ("`position`", "`panel.instruction`"),
        ("scene/supplement", "panel.instruction"),
        ("positive/caption", "actor instruction"),
    ):
        text = text.replace(old, new)
    text = prompt_sub(
        r"`panel\.instruction`(?:, `panel\.instruction`)*(?: (?:and|or) `panel\.instruction`)+",
        "`panel.instruction`", text)
    text = text.replace(
        "The trusted active policy remains stricter than this section. "
        "Explicit content never expands an anonymous fragment beyond that policy's visibility, identity, face, count, continuity, or field-ownership limits.",
        "Explicit content never expands a contact fragment beyond DETAIL-MANGA's visibility, identity, face, continuity, or field-ownership limits.",
    )
    if natural:
        text = _apply_natural_detail_projection(text, container_field="`panel.instruction`")
        text = text.replace(
            "Do not add a raw `nsfw` rating tag to `panel.instruction`; convey the explicit content directly through literal natural-language visual descriptions.",
            "Do not add a raw `nsfw` rating tag to `page.caption` or `panel.instruction`; convey the explicit content directly through literal natural-language visual descriptions.",
        )
    else:
        text = text.replace(
            "Add `nsfw` exactly once in `panel.instruction`.",
            "Add `nsfw` only once in `page.caption` when source-established sexual activity or genital exposure is depicted; never repeat it per panel or actor.",
        )
    return text


def moss_instruction_fields(text, *, natural: bool = False):
    # These sections belong to ANIMA's regional renderer or legacy descriptor.
    # Removing them avoids asking the model to reconcile two output contracts.
    text = prompt_sub(
        r"(?ms)^REGIONAL SUBJECT FOCUS[ \t]*\r?\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*\r?$|\Z)",
        "", text)
    text = prompt_sub(
        r"(?ms)^3\. Field ownership\s*\n.*?(?=^4\. Physical coherence)", "", text)
    for old in (
        "For Key Visual, `outfit_state` records its complete logical outfit, including off-frame items; `positive` contains only visible or coverage-defining garments.",
        "Keep the complete logical outfit in `outfit_state`, only visible or coverage-defining garments in `positive`, and do not put logical outfit records in `supplement`.",
        "Preserve the complete logical wardrobe in `outfit_state`, including cropped or occluded garments, but put only visible or coverage-defining clothing in `positive`.",
    ):
        text = text.replace(old,
            "Maintain the complete logical wardrobe as continuity context; include only visible or "
            "coverage-defining garments in the character item's `caption`. Do not output a logical outfit record.")
    # The ordinary DETAIL contract deliberately has no arbitrary tag quota.
    text = prompt_sub(r"(?m)^[ \t]*- Aim for roughly 35 or fewer[^\n]*\n?", "", text)
    text = text.replace(
        "Create exactly one standalone promotional Key Visual",
        "Create exactly one promotional Key Visual in the combined response's `keyvis` object")
    text = text.replace(
        "This is independent from narrative scene placement. Do not select slots, output scenes, copy a presumed planned scene, or refer to another worker.",
        "Choose the Key Visual independently from ordinary scene placement. Write it only in `keyvis`, "
        "without a slot; the assigned ordinary images remain in `scenes` in the same response.")
    text = text.replace("ANIMA reads", "MOSS reads").replace("Anima-facing", "MOSS-facing")
    text = text.replace("character/region entries", "character items")
    text = text.replace("character/region entry", "character item")
    for old, new in (
        ("characters[].positive", "items[].caption"),
        ("characters[n].positive", "items[n].caption"),
        ("characters[]", "items[]"),
        ("characters[n]", "items[n]"),
        ("`positive`", "`caption`"),
        ("`camera`", "`summary.caption`"),
        ("`scene`", "`summary.caption`"),
        ("`supplement`", "`summary.caption`"),
        ("`position`", "`summary.caption`"),
        ("scene/supplement", "summary.caption"),
        ("positive/caption", "character-item caption"),
    ):
        text = text.replace(old, new)
    # Bot policy presets also use unquoted field names. Match field-reference
    # phrases, not narrative uses of words such as scene, camera or positive.
    text = prompt_sub(r"\b(in|into) (?:its |their |the )?positive\b",
                      r"\1 the character item's `caption`", text)
    text = prompt_sub(r"\b(in|into) (scene|supplement)\b",
                      r"\1 `summary.caption`", text)
    text = text.replace("scene, never in items[].caption", "summary.caption, never in items[].caption")
    text = text.replace("camera, tags and prose", "shared and item captions")
    text = text.replace("camera, scene and supplement", "summary.caption")
    text = text.replace("crop, and supplement", "crop, and shared composition")
    text = text.replace("occlusion, and supplement", "occlusion, and shared composition")
    text = text.replace("person's positive", "person's item caption")
    text = text.replace("generator-facing positive", "generator-facing caption")
    text = text.replace(
        "express its visible portion in `summary.caption` and exact geometry in `summary.caption`",
        "express its visible portion and contact geometry together in `summary.caption`")
    text = text.replace(
        "keep only anonymous partner fragments in `summary.caption` and their exact geometry in `summary.caption`",
        "keep anonymous partner fragments and their contact geometry together in `summary.caption`")
    # Several legacy fields now share one caption; state that destination once.
    text = prompt_sub(
        r"`summary\.caption`(?:, `summary\.caption`)*(?: (?:and|or) `summary\.caption`)+",
        "`summary.caption`", text)
    if natural:
        text = _apply_natural_detail_projection(text, container_field="`summary.caption`")
    return text
