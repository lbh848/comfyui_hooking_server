"""Project shared illustration instructions onto the single-image MOSS schema.

Only instruction text passes through here. Profiles, source passages and other
user-message data retain their original keys and values.
"""
from modes.illustration_prompt_assembly import prompt_sub


def moss_instruction_fields(text):
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
    return text
