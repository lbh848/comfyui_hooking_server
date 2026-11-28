"""
tag_classifier - 설명 CSV 카테고리 기반 태그 자동 분류

auto_complete/ 폴더의 CSV에서 [카테고리 > 서브카테고리]를 파싱하여
태그를 3그룹(외모/신체, 복장, 미분류)으로 분류한다.
"""

from app_paths import data_root

import csv
import os
import re
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCOMPLETE_DIR = os.path.join(data_root(BASE_DIR), "auto_complete")

# 태그명(소문자, 공백) → (category, subcategory, description)
_tag_info: dict[str, tuple[str, str, str]] = {}
_loaded = False

# 그룹 분류 규칙
GROUP_RULES = {
    ("인물", "눈"): "외모/신체",
    ("인물", "종족"): "외모/신체",
    ("신체", "*"): "외모/신체",
    ("패션", "헤어스타일"): "외모/신체",
    ("패션", "헤어컬러"): "외모/신체",
    ("패션", "상의"): "복장",
    ("패션", "하의"): "복장",
    ("패션", "의상"): "복장",
    ("패션", "액세서리"): "복장",
    ("패션", "디테일"): "복장",
    ("패션", "양말"): "복장",
}


def _extract_description(desc_field: str) -> str:
    """CSV 설명 필드에서 카테고리 태그와 키워드를 제거한 순수 설명 추출."""
    # [카테고리 > 서브] 제거
    text = re.sub(r'\[([^\]]+)\]', '', desc_field).strip()
    # . 키워드: ... 제거
    text = re.sub(r'\.\s*키워드:.*$', '', text).strip()
    return text


def _load():
    global _tag_info, _loaded
    _tag_info = {}

    if not os.path.isdir(AUTOCOMPLETE_DIR):
        print(f"[TAG_CLASSIFIER] CSV 폴더 없음: {AUTOCOMPLETE_DIR}")
        _loaded = True
        return

    files = [f for f in os.listdir(AUTOCOMPLETE_DIR)
             if f.lower().endswith(".csv")]
    # 자동완성과 같이 큰 설명 CSV의 중복 태그를 먼저 채택한다.
    files.sort(key=lambda f: -os.path.getsize(os.path.join(AUTOCOMPLETE_DIR, f)))
    for f in files:

        fpath = os.path.join(AUTOCOMPLETE_DIR, f)
        try:
            with open(fpath, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 4:
                        continue
                    name = row[0].strip().lower().replace('_', ' ').replace('﻿', '')
                    if not name or name in _tag_info:
                        continue
                    desc_field = row[3].strip()
                    m = re.match(r'\[([^>]+)\s*>\s*([^\]]+)\]', desc_field)
                    if m:
                        cat = m.group(1).strip()
                        subcat = m.group(2).strip()
                        description = _extract_description(desc_field)
                        _tag_info[name] = (cat, subcat, description)
            print(f"[TAG_CLASSIFIER] CSV 로드 완료: {f} ({len(_tag_info):,}개 태그)")
        except Exception as e:
            print(f"[TAG_CLASSIFIER] 로드 실패: {f}: {e}")
            traceback.print_exc()

    _loaded = True


POSE_SUBCATEGORIES = {"자세", "포즈", "움직임", "시선", "표정", "유연성"}


def _classify_tag(tag_name: str) -> tuple[str, str, str]:
    """단일 태그를 분류. 반환: (group, subcategory, description)"""
    if not _loaded:
        _load()

    key = tag_name.strip().lower().replace('_', ' ')
    info = _tag_info.get(key)

    if not info:
        return ("미분류", "", "")

    cat, subcat, description = info

    if cat == "신체" and subcat in POSE_SUBCATEGORIES:
        return ("미분류", subcat, description)

    group = GROUP_RULES.get((cat, subcat))
    if group:
        return (group, subcat, description)

    group = GROUP_RULES.get((cat, "*"))
    if group:
        return (group, subcat, description)

    return ("미분류", subcat, description)


def classify_prompt(prompt: str) -> dict:
    """
    프롬프트 문자열을 받아 태그를 3그룹으로 분류.

    반환: {
        "외모/신체": [{"tag": "...", "sub": "눈", "desc": "..."}, ...],
        "복장": [{"tag": "...", "sub": "상의", "desc": "..."}, ...],
        "미분류": [{"tag": "...", "sub": "...", "desc": "..."}, ...],
    }
    """
    if not _loaded:
        _load()

    groups: dict[str, list[dict]] = {
        "외모/신체": [],
        "복장": [],
        "미분류": [],
    }

    raw_tags = [t.strip() for t in prompt.split(',') if t.strip()]

    for raw in raw_tags:
        inner = raw
        m = re.match(r'[\(\[{]+[\d.]*:?\s*(.+?)[\)\]}]+$', raw)
        if m:
            inner = m.group(1).strip()

        group, sub, desc = _classify_tag(inner)
        groups[group].append({"tag": raw, "sub": sub, "desc": desc})

    return {k: v for k, v in groups.items() if v}


_COVERAGE_WRAPPER_RE = re.compile(r"^[\(\[{]+[\d.]*:?\s*(.+?)\s*(?::[\d.+-]+)?[\)\]}]+$")
_NAKED_GARMENT_RE = re.compile(r"\bnaked\s+(?:shirt|apron|coat|jacket|sweater|hoodie|ribbon|towel|band-?aid|sheet|overalls|tabard|cloak|cape)\b")
_OPEN_OUTERWEAR_RE = re.compile(
    r"\b(?:open|unbuttoned|unzipped)\s+(?:jacket|coat|cardigan|hoodie|vest|blazer|robe|kimono|yukata|haori|parka|cloak|cape|shirt|blouse|flannel|trench\s+coat)\b"
)
_DISPLACED_MODIFIER_RE = re.compile(
    r"\b(?:lift|lifted|lifting|hiked(?:\s+up)?|pulled\s+(?:up|down|aside)|aside|around\s+(?:ankles|one\s+leg|thighs|knees|waist)|slipped\s+(?:off|down)|hanging\s+off)\b"
)
_PARTIAL_UNDRESS_TOP_RE = re.compile(
    r"\b(?:shirt\s+lift(?:ed)?|clothes\s+lift(?:ed)?|bra\s+pull(?:ed)?|bra\s+lift(?:ed)?|open\s+bra|strap\s+slip|breast\s+slip|nipple\s+slip|unbuttoned\s+(?:shirt|blouse)|open\s+shirt|off[\s-]?shoulder\s+(?:shirt|top|sweater|blouse)|pulled\s+up\s+(?:shirt|top|sweater|vest|blouse))\b"
)
_PARTIAL_UNDRESS_BOTTOM_RE = re.compile(
    r"\b(?:panties\s+(?:aside|down|around\s+\w+(?:\s+\w+)?|pull(?:ed)?)|underwear\s+(?:aside|down)|pants\s+(?:down|around\s+\w+(?:\s+\w+)?)|jeans\s+down|shorts\s+down|skirt\s+(?:lift(?:ed)?|around\s+\w+(?:\s+\w+)?|pulled\s+up|hiked(?:\s+up)?)|open\s+fly|unzipped\s+(?:pants|jeans|fly|skirt)|crotchless)\b"
)
_PARTIAL_UNDRESS_GENERAL_RE = re.compile(
    r"\b(?:partially\s+undressed|undressing|clothes\s+down|clothing\s+aside|wardrobe\s+malfunction|torn\s+clothes|disheveled\s+clothes|unbuttoned|unzipped)\b"
)
_UNDERWEAR_TOP_RE = re.compile(
    r"\b(?:bra|brassiere|bralette|sports\s+bra|bikini\s+top|tube\s+top\s+bra|nipple\s+pasties|pasties|sarashi|chest\s+wrap)\b"
)
_UNDERWEAR_BOTTOM_RE = re.compile(
    r"\b(?:panties|panty|underpants|underwear|briefs|boxers|boxer\s+briefs|thong|g[\s-]?string|bikini\s+bottom|bloomers|fundoshi|loincloth)\b"
)
_UNDERWEAR_FULL_RE = re.compile(r"\b(?:lingerie|babydoll|teddy|corset\s+lingerie)\b")
_FOOTWEAR_RE = re.compile(
    r"\b(?:shoes?|dress\s+shoes?|sneakers?|loafers?|boots?|ankle\s+boots?|knee\s+boots?|high\s+heels?|heels|pumps|sandals?|slippers?|uwabaki|flip[\s-]?flops?|clogs?|footwear)\b"
)
_LEGWEAR_RE = re.compile(
    r"\b(?:socks?|ankle\s+socks?|crew\s+socks?|loose\s+socks?|knee\s+highs?|knee[\s-]?socks?|thigh\s*highs?|over[\s-]?knees?|stockings?|leg\s+warmers?|garter\s+(?:belt|straps?))\b"
)
_BOTTOM_RE = re.compile(
    r"\b(?:pants|trousers|slacks|jeans|denim\s+pants|shorts|hotpants|denim\s+shorts|sweatpants|track\s+pants|cargo\s+pants|chinos|skirt|miniskirt|microskirt|pleated\s+skirt|pencil\s+skirt|long\s+skirt|plaid\s+skirt|leggings|tights|pantyhose|spats|culottes|hakama|kilt)\b"
)
_TOP_RE = re.compile(
    r"\b(?:t[\s-]?shirt|shirt|dress\s+shirt|button[\s-]?up|polo\s+shirt|blouse|sweater|knit\s+vest|vest|waistcoat|cardigan|hoodie|sweatshirt|pullover|jacket|blazer|coat|trench\s+coat|parka|windbreaker|turtleneck|tank\s+top|camisole|crop\s+top|tube\s+top|halter\s*top|sleeveless\s+top|knit\s+top|lace\s+top|mesh\s+top|jersey\s+top|jersey|tunic|corset|bustier|bolero|shrug|cape|cloak|poncho|stole|shawl|sleeves?)\b"
)
_FULL_BODY_RE = re.compile(
    r"\b(?:dress|sundress|evening\s+gown|gown|nightgown|one[\s-]?piece|bodysuit|leotard|jumpsuit|romper|overalls|dungarees|school\s+uniform|sailor\s+uniform|serafuku|uniform|maid\s+outfit|kimono|yukata|bathrobe|robe|hanbok|qipao|cheongsam|china\s+dress|swimsuit|bikini|wetsuit|plugsuit|armor|suit|tuxedo|apron|pajamas|tracksuit|gym\s+uniform|casual\s+clothes)\b"
)
_ACCESSORY_RE = re.compile(
    r"\b(?:glasses|sunglasses|monocle|eyepatch|choker|collar|necklace|pendant|earrings?|piercings?|bracelet|wristband|watch|ring|hairpin|hairclip|hairband|headband|hair\s+ribbon|ribbon|bow|scrunchie|tiara|crown|hat|cap|beret|beanie|scarf|necktie|tie|bowtie|belt|gloves?|fingerless\s+gloves?|arm\s+warmers?|mask|blindfold|gag|handcuffs|leash)\b"
)


def _normalize_coverage_tag(raw: str) -> str:
    text = str(raw or "").strip()
    match = _COVERAGE_WRAPPER_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    return re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()


def classify_garment_coverage_slot(tag: str) -> str:
    """Classify a single outfit tag into a canonical garment/coverage bucket.

    Uses fast deterministic English rules first (supporting multi-word color and
    material modifiers), falling back to the already-loaded autocomplete CSV
    taxonomy when available.
    """
    norm = _normalize_coverage_tag(tag)
    if not norm:
        return "other"
    if _NAKED_GARMENT_RE.search(norm) or norm in {
        "nude", "naked", "completely nude", "completely naked", "butt naked",
        "topless", "shirtless", "bottomless", "no panties", "commando", "no bra",
        "bare breasts", "bare chest", "bare torso", "barefoot", "bare legs",
    }:
        return "coverage_state"
    if _OPEN_OUTERWEAR_RE.search(norm):
        return "open_outerwear"
    if _PARTIAL_UNDRESS_TOP_RE.search(norm):
        return "partial_undress_top"
    if _PARTIAL_UNDRESS_BOTTOM_RE.search(norm):
        return "partial_undress_bottom"
    if _PARTIAL_UNDRESS_GENERAL_RE.search(norm):
        return "partial_undress"
    if _FOOTWEAR_RE.search(norm):
        return "footwear"
    if _UNDERWEAR_TOP_RE.search(norm):
        return "underwear_top"
    if _UNDERWEAR_BOTTOM_RE.search(norm):
        return "underwear_bottom"
    if _UNDERWEAR_FULL_RE.search(norm):
        return "underwear_full"
    # Check specific tops (e.g. "dress shirt") before full-body ("dress").
    if _TOP_RE.search(norm) and not re.search(r"\b(?:swimsuit|bikini|school\s+uniform)\b", norm):
        return "top"
    if _BOTTOM_RE.search(norm):
        return "bottom"
    if _FULL_BODY_RE.search(norm):
        return "full_body"
    if _LEGWEAR_RE.search(norm):
        return "legwear"
    if _ACCESSORY_RE.search(norm):
        return "accessory"
    # Optional fallback to already-loaded CSV metadata without triggering I/O.
    if _loaded and norm in _tag_info:
        cat, sub, _ = _tag_info[norm]
        if cat in ("패션", "복장"):
            if "상의" in sub or sub in ("아우터", "재킷", "코트", "겉옷"):
                return "top"
            if "하의" in sub or "치마" in sub or "반바지" in sub:
                return "bottom"
            if sub in ("속옷", "언더웨어"):
                return "underwear_full"
            if sub in ("양말", "양말류"):
                return "legwear"
            if sub in ("신발", "부츠", "구두"):
                return "footwear"
            if sub in ("원피스", "전신", "전신 의상", "한벌옷", "드레스", "교복", "제복", "유니폼", "수영복"):
                return "full_body"
            if "액세서리" in sub or "장식" in sub or "장신구" in sub or "안경" in sub:
                return "accessory"
    return "other"


def _strip_slots_for_coverage(tokens: list[str], *, forced_coverage: str = "") -> tuple[list[str], list[str]]:
    norms = [_normalize_coverage_tag(token) for token in tokens]
    coverage_set = set(norms)
    if forced_coverage:
        coverage_set.add(_normalize_coverage_tag(forced_coverage))

    completely_nude = bool(coverage_set & {"completely nude", "completely naked"})
    nude = completely_nude or bool(coverage_set & {"nude", "naked", "butt naked"})
    topless = bool(coverage_set & {"topless", "shirtless", "bare breasts", "bare chest", "bare torso"})
    bottomless = "bottomless" in coverage_set
    no_panties = bool(coverage_set & {"no panties", "commando"})
    no_bra = "no bra" in coverage_set

    if not (nude or topless or bottomless or no_panties or no_bra):
        return list(tokens), []

    has_displaced_skirt = any(
        "skirt" in norm and bool(_DISPLACED_MODIFIER_RE.search(norm))
        for norm in norms
    )
    kept: list[str] = []
    removed: list[str] = []
    for raw, norm in zip(tokens, norms):
        slot = classify_garment_coverage_slot(raw)
        drop = False
        if completely_nude:
            if slot in {
                "top", "bottom", "full_body", "open_outerwear",
                "underwear_top", "underwear_bottom", "underwear_full",
                "legwear", "footwear",
                "partial_undress", "partial_undress_top", "partial_undress_bottom",
            }:
                drop = True
        elif nude:
            if slot in {
                "top", "bottom", "full_body", "open_outerwear",
                "underwear_top", "underwear_bottom", "underwear_full",
                "partial_undress", "partial_undress_top", "partial_undress_bottom",
            }:
                drop = True
        else:
            if topless and slot in {"top", "underwear_top", "full_body", "partial_undress_top"}:
                drop = True
            if bottomless:
                is_displaced_skirt_tag = (
                    "skirt" in norm and (bool(_DISPLACED_MODIFIER_RE.search(norm)) or has_displaced_skirt)
                )
                if slot in {"bottom", "underwear_bottom", "full_body", "partial_undress_bottom"}:
                    if not is_displaced_skirt_tag:
                        drop = True
            if no_panties and (
                slot == "underwear_bottom"
                or (slot == "partial_undress_bottom" and any(k in norm for k in ("panties", "panty", "underwear")))
            ):
                drop = True
            if no_bra and (
                slot == "underwear_top"
                or (slot == "partial_undress_top" and "bra" in norm)
            ):
                drop = True
        if drop:
            removed.append(raw)
        else:
            kept.append(raw)
    return kept, removed


def resolve_outfit_coverage_conflicts(outfit_tags):
    """Deterministically strip garments and partial-undress tags that conflict with explicit coverage states.

    Accepts either a comma-separated tag string or a list of tags and returns the
    same type with conflicting garments removed.
    """
    if isinstance(outfit_tags, str):
        tokens = [part.strip() for part in outfit_tags.split(",") if part.strip()]
        kept, _removed = _strip_slots_for_coverage(tokens)
        return ", ".join(kept)
    if isinstance(outfit_tags, list):
        tokens = [str(part).strip() for part in outfit_tags if str(part).strip()]
        kept, _removed = _strip_slots_for_coverage(tokens)
        return kept
    return outfit_tags


def strip_wardrobe_items_for_coverage(worn: list[str], coverage: str) -> tuple[list[str], list[str]]:
    """Split ``worn`` garment tags into ``(still_worn, stripped)`` for a coverage transition."""
    tokens = [str(part).strip() for part in (worn or []) if str(part).strip()]
    return _strip_slots_for_coverage(tokens, forced_coverage=coverage)
