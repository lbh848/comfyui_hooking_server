"""삽화 모드 단어 기반 규칙 처리.

RAW 프롬프트의 섹션 구조를 유지하면서 다음 범위에 규칙을 적용한다.

- NAME: 치환 규칙만 적용
- SPEAK: 구조화된 각 발화 줄의 ``NAME:`` 부분에만 치환 규칙 적용
- SETUP/CHAR/SUPPLEMENT: 치환, 제거, 가중치 조정 규칙 적용
"""

import re


_SECTION_MARKER_RE = re.compile(r"^\[([A-Za-z][A-Za-z0-9_]*)\]", re.IGNORECASE | re.MULTILINE)
_PROMPT_RULE_SECTIONS = {"setup", "char", "supplement"}
_REPLACE_ONLY_SECTIONS = {"name"}
_WEIGHTED_TAG_RE = re.compile(
    r"^\(\s*(?P<tag>.+?)\s*:\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$"
)
_WEIGHT_VALUE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def apply_remove_rule(text: str, rule: dict) -> tuple[str, bool]:
    """제거 모드 규칙을 적용하고 ``(결과, 실제 적용 여부)``를 반환한다."""
    if not text:
        return text, False

    pattern_str = (rule.get("pattern") or "").strip()
    if not pattern_str:
        print("[WORD_RULE] 제거 모드 규칙에 pattern이 없어 스킵합니다.")
        return text, False

    trigger = (rule.get("trigger") or "").strip()
    remove_trigger = bool(rule.get("remove_trigger", False))
    if trigger and not re.search(re.escape(trigger), text, flags=re.IGNORECASE):
        return text, False

    parts = pattern_str.split("*")
    regex = ".*".join(re.escape(part) for part in parts)
    try:
        rx = re.compile(regex, flags=re.IGNORECASE)
    except re.error as exc:
        print(f"[WORD_RULE] 제거 패턴 컴파일 실패(pattern={pattern_str!r}): {exc}")
        return text, False

    trigger_lower = trigger.lower()
    segments = [segment.strip() for segment in text.split(",")]
    output = []
    removed = 0
    for segment in segments:
        if segment and rx.fullmatch(segment):
            if trigger and not remove_trigger and segment.lower() == trigger_lower:
                output.append(segment)
                continue
            removed += 1
            continue
        output.append(segment)

    if removed == 0:
        return text, False

    print(
        f"[WORD_RULE] 제거 모드 적용: pattern={pattern_str!r}, "
        f"trigger={trigger!r}, {removed}개 세그먼트 제거"
    )
    return ", ".join(segment for segment in output if segment), True


def apply_weight_rule(text: str, rule: dict) -> tuple[str, bool]:
    """일치하는 쉼표 단위 태그의 가중치를 강제하거나 제거한다."""
    if not text:
        return text, False

    source = (rule.get("source") or "").strip()
    if not source:
        print("[WORD_RULE] 가중치 조정 규칙에 감지 단어(source)가 없어 스킵합니다.")
        return text, False

    remove_weight = bool(rule.get("remove_weight", False))
    weight = str(rule.get("weight", "")).strip()
    if not remove_weight and not _WEIGHT_VALUE_RE.fullmatch(weight):
        print(
            f"[WORD_RULE] 가중치 조정값이 올바른 숫자가 아니어서 스킵합니다"
            f"(source={source!r}, weight={weight!r})."
        )
        return text, False

    output = []
    changed = 0
    source_key = source.casefold()
    for segment in text.split(","):
        leading = segment[: len(segment) - len(segment.lstrip())]
        trailing = segment[len(segment.rstrip()):]
        core = segment.strip()
        weighted_match = _WEIGHTED_TAG_RE.fullmatch(core)
        detected_tag = weighted_match.group("tag").strip() if weighted_match else core

        if detected_tag.casefold() != source_key:
            output.append(segment)
            continue

        if remove_weight:
            replacement = detected_tag if weighted_match else core
        else:
            replacement = f"({detected_tag}:{weight})"

        replaced_segment = leading + replacement + trailing
        output.append(replaced_segment)
        if replaced_segment != segment:
            changed += 1

    if changed == 0:
        return text, False

    action = "가중치 제거" if remove_weight else f"가중치 {weight} 강제"
    print(f"[WORD_RULE] 가중치 조정 적용: source={source!r}, action={action}, {changed}개 태그")
    return ",".join(output), True


def apply_replacement_rules(text: str, rules: list[dict]) -> tuple[str, int]:
    """활성화된 치환 규칙만 적용하고 실제 적용된 규칙 수를 반환한다."""
    if not text:
        return text, 0

    result = text
    applied = 0
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        if (rule.get("type") or "replace").strip().lower() in {"remove", "weight"}:
            continue

        source = (rule.get("source") or "").strip()
        target = (rule.get("target") or "").strip()
        if not source:
            continue

        result, count = re.subn(re.escape(source), target, result, flags=re.IGNORECASE)
        if count > 0:
            applied += 1
    return result, applied


def apply_prompt_rules(positive: str, negative: str, rules: list[dict]) -> tuple[str, str, int]:
    """기존 프롬프트 문자열에 치환과 제거 규칙을 모두 적용한다."""
    applied_rule_indexes = set()
    for index, rule in enumerate(rules):
        if not rule.get("enabled", True):
            continue

        rule_type = (rule.get("type") or "replace").strip().lower()
        if rule_type == "insert":
            # 삽입 타입은 최종 positive 후처리(apply_insert_rules)에서만 동작.
            continue
        if rule_type == "remove":
            positive, did_positive = apply_remove_rule(positive, rule)
            negative, did_negative = apply_remove_rule(negative, rule)
            if did_positive or did_negative:
                applied_rule_indexes.add(index)
            continue

        if rule_type == "weight":
            positive, did_positive = apply_weight_rule(positive, rule)
            negative, did_negative = apply_weight_rule(negative, rule)
            if did_positive or did_negative:
                applied_rule_indexes.add(index)
            continue

        source = (rule.get("source") or "").strip()
        target = (rule.get("target") or "").strip()
        if not source:
            continue

        pattern = re.escape(source)
        positive, positive_count = re.subn(pattern, target, positive, flags=re.IGNORECASE)
        negative, negative_count = re.subn(pattern, target, negative, flags=re.IGNORECASE)
        if positive_count or negative_count:
            applied_rule_indexes.add(index)

    return positive, negative, len(applied_rule_indexes)


def _tag_core(segment: str) -> str:
    """쉼표 단위 세그먼트에서 괄호·가중치 문법을 벗겨낸 태그(casefold)를 반환한다.

    중복 삽입 검사용. ``(word:1.2)`` → ``word``, ``(word)`` → ``word``,
    일반 ``word`` → ``word`` 로 정규화한다.
    """
    core = segment.strip()
    weighted_match = _WEIGHTED_TAG_RE.fullmatch(core)
    if weighted_match:
        return weighted_match.group("tag").strip().casefold()
    if core.startswith("(") and core.endswith(")"):
        inner = core[1:-1].strip()
        # (tag:1.2) 는 위 정규식이 잡지만, 혹시 남은 경우 한 번 더 벗김
        inner_match = _WEIGHTED_TAG_RE.fullmatch(inner)
        if inner_match:
            return inner_match.group("tag").strip().casefold()
        return inner.casefold()
    return core.casefold()


def _word_present_in_region(region: str, word: str) -> bool:
    """영역 텍스트 안에 ``word`` 태그가 콤마 단위로 정확히 존재하는지 검사한다.

    줄바꿈도 구분자로 취급해 섹션 경계를 무시하지 않도록 정규화한다.
    ``blue eyes`` 가 있을 때 ``light blue eyes`` 같은 별개 태그는 False 다.
    """
    word_cf = word.strip().casefold()
    if not word_cf:
        return False
    normalized = re.sub(r"[\r\n]+", ",", region)
    for segment in normalized.split(","):
        if _tag_core(segment) == word_cf:
            return True
    return False


# 품질 섹션 헤더 + 그 직후 한 줄(태그 리스트)을 매칭
_ANIMA_QUALITY_LINE_RE = re.compile(r"(\[ANIMA_QUALITY\]\n)([^\n]*)", re.IGNORECASE)
_SDXL_QUALITY_LINE_RE = re.compile(r"(\[SDXL_QUALITY\]\n)([^\n]*)", re.IGNORECASE)
_ANIMA_REGION_END_RE = re.compile(r"\n\[SDXL_QUALITY\]", re.IGNORECASE)
_SDXL_REGION_END_RE = re.compile(r"\n\[CHAR_LIST\]", re.IGNORECASE)


def _extract_region(text: str, start_re: "re.Pattern", end_re: "re.Pattern") -> str:
    """시작 마커부터 (있으면) 종료 마커 전까지의 영역 문자열을 반환한다."""
    start_match = start_re.search(text)
    if not start_match:
        return ""
    start = start_match.start()
    tail = text[start_match.end():]
    end_match = end_re.search(tail)
    end = start_match.end() + end_match.start() if end_match else len(text)
    return text[start:end]


def _insert_word_after_quality_line(text: str, quality_line_re: "re.Pattern",
                                    word: str, label: str) -> tuple[str, bool]:
    """품질 섹션 헤더 직후의 태그 줄 끝에 ``word`` 를 추가한다."""
    match = quality_line_re.search(text)
    if not match:
        return text, False
    header = match.group(1)
    tag_line = match.group(2).rstrip()
    new_line = f"{tag_line}, {word}" if tag_line.strip() else word
    print(f"[WORD_RULE] 삽입 적용: word={word!r}, 영역={label}")
    return text[:match.start()] + header + new_line + text[match.end():], True


def apply_insert_rules(positive: str, rules: list[dict]) -> tuple[str, int]:
    """최종 positive의 [ANIMA_QUALITY]/[SDXL_QUALITY] 뒤에 단어를 강제 삽입한다.

    각 모델 영역(ANIMA / SDXL)에 해당 단어가 이미 존재하면(가중치·괄호 형태
    포함) 그 영역은 스킵하여 중복을 막는다. 삽입은 평문 태그로만 한다.
    """
    if not positive:
        return positive, 0

    applied_rule_count = 0
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        if (rule.get("type") or "replace").strip().lower() != "insert":
            continue

        word = (rule.get("word") or rule.get("source") or "").strip()
        if not word:
            print("[WORD_RULE] 삽입 규칙에 word가 없어 스킵합니다.")
            continue

        did_insert = False

        anima_region = _extract_region(positive, _ANIMA_QUALITY_LINE_RE, _ANIMA_REGION_END_RE)
        if anima_region:
            if _word_present_in_region(anima_region, word):
                print(f"[WORD_RULE] 삽입 스킵(이미 존재): word={word!r}, 영역=ANIMA")
            else:
                positive, did_anima = _insert_word_after_quality_line(
                    positive, _ANIMA_QUALITY_LINE_RE, word, "ANIMA"
                )
                did_insert = did_insert or did_anima

        sdxl_region = _extract_region(positive, _SDXL_QUALITY_LINE_RE, _SDXL_REGION_END_RE)
        if sdxl_region:
            if _word_present_in_region(sdxl_region, word):
                print(f"[WORD_RULE] 삽입 스킵(이미 존재): word={word!r}, 영역=SDXL")
            else:
                positive, did_sdxl = _insert_word_after_quality_line(
                    positive, _SDXL_QUALITY_LINE_RE, word, "SDXL"
                )
                did_insert = did_insert or did_sdxl

        if did_insert:
            applied_rule_count += 1

    return positive, applied_rule_count


def _apply_rules_to_single_prompt(text: str, rules: list[dict]) -> tuple[str, int]:
    transformed, _unused_negative, applied = apply_prompt_rules(text, "", rules)
    return transformed, applied


def apply_speak_name_replacements(speak_text: str, rules: list[dict]) -> tuple[str, int]:
    """SPEAK의 구조화된 발화자 이름에만 치환 규칙을 적용한다.

    따옴표 대사(``NAME: "..."``)와 이름 있는 생각(``NAME: (...)``)의 콜론
    왼쪽만 대상으로 삼는다. 대사/생각 본문과 감정 표기는 그대로 보존한다.
    """
    if not speak_text:
        return speak_text, 0

    applied_total = 0
    output_lines = []
    # 콜론 앞의 비어 있지 않은 전체 문자열을 NAME으로 본다. 영문 한 단어로
    # 제한하면 ``mariya mikhailovna kujou: "..."`` 같은 이름을 놓친다.
    speaker_re = re.compile(
        r'^(?P<indent>\s*)(?P<speaker>[^:\r\n]+?)(?P<separator>\s*:\s*)(?=["(])'
    )

    for line in speak_text.splitlines(keepends=True):
        match = speaker_re.match(line)
        if not match:
            output_lines.append(line)
            continue

        replaced_speaker, applied = apply_replacement_rules(
            match.group("speaker").strip(), rules
        )
        applied_total += applied
        output_lines.append(
            line[:match.start("speaker")] + replaced_speaker + line[match.end("speaker"):]
        )

    return "".join(output_lines), applied_total


def _preserve_outer_whitespace(text: str, transform) -> tuple[str, int]:
    """섹션 내용 바깥의 개행/공백은 유지하고 실제 내용만 변환한다."""
    if not text or not text.strip():
        return text, 0

    left_length = len(text) - len(text.lstrip())
    right_length = len(text) - len(text.rstrip())
    core_end = len(text) - right_length if right_length else len(text)
    core = text[left_length:core_end]
    transformed, applied = transform(core)
    return text[:left_length] + transformed + text[core_end:], applied


def apply_raw_prompt_rules(raw_prompt: str, rules: list[dict]) -> tuple[str, int]:
    """섹션별 범위를 지키며 RAW 삽화 프롬프트에 단어 규칙을 선적용한다."""
    if not raw_prompt or not rules:
        return raw_prompt, 0

    markers = list(_SECTION_MARKER_RE.finditer(raw_prompt))
    if not markers:
        print("[WORD_RULE] RAW 프롬프트에 지원 섹션이 없어 선처리를 스킵합니다.")
        return raw_prompt, 0

    output = []
    cursor = 0
    applied_total = 0
    for index, marker in enumerate(markers):
        content_start = marker.end()
        content_end = markers[index + 1].start() if index + 1 < len(markers) else len(raw_prompt)
        section_name = marker.group(1).lower()
        content = raw_prompt[content_start:content_end]

        output.append(raw_prompt[cursor:content_start])
        if section_name in _PROMPT_RULE_SECTIONS:
            transformed, applied = _preserve_outer_whitespace(
                content, lambda value: _apply_rules_to_single_prompt(value, rules)
            )
        elif section_name in _REPLACE_ONLY_SECTIONS:
            transformed, applied = _preserve_outer_whitespace(
                content, lambda value: apply_replacement_rules(value, rules)
            )
        elif section_name == "speak":
            transformed, applied = _preserve_outer_whitespace(
                content, lambda value: apply_speak_name_replacements(value, rules)
            )
        else:
            transformed, applied = content, 0

        output.append(transformed)
        applied_total += applied
        cursor = content_end

    output.append(raw_prompt[cursor:])
    return "".join(output), applied_total


# 캐릭터 태그 덮어쓰기 특수 규칙 = 필드(눈/얼굴) × 동작(제거/치환).
#   char_eye_remove   : eye_tags  -> ""
#   char_eye_replace  : eye_tags  -> target
#   char_face_remove  : face_tags -> ""
#   char_face_replace : face_tags -> target
_CHAR_TAG_OVERRIDE_MAP = {
    "char_eye_remove":   ("eye_tags",  "remove",  "캐릭터 눈 제거"),
    "char_eye_replace":  ("eye_tags",  "replace", "캐릭터 눈 치환"),
    "char_face_remove":  ("face_tags", "remove",  "캐릭터 얼굴 제거"),
    "char_face_replace": ("face_tags", "replace", "캐릭터 얼굴 치환"),
}


def apply_char_tag_override_rules(
    characters: list, rules: list, trigger_text: str
) -> list:
    """캐릭터 눈/얼굴 제거·치환 특수 규칙을 characters 복사본에 임시 적용한다.

    원본 ``characters`` (bot.json 캐릭터 데이터)는 절대 훼손하지 않으며, 각
    캐릭터 dict 의 얕은 복사로 이루어진 새 리스트를 반환한다. 빌드 직전
    변수 상에서만 치환/제거를 거친 뒤 프롬프트 빌더로 넘기기 위함이다.

    규칙 타입(``_CHAR_TAG_OVERRIDE_MAP``):
      - ``char_eye_remove``   / ``char_face_remove``  : 해당 필드 -> ""
      - ``char_eye_replace``  / ``char_face_replace`` : 해당 필드 -> target

    trigger 가 발동하면 감지된(전달된) 모든 캐릭터에 일괄 적용한다.
    trigger 가 비어 있거나 매칭되지 않으면 해당 규칙은 스킵한다.
    """
    override_rules = [
        r for r in (rules or [])
        if r.get("enabled", True)
        and (r.get("type") or "").strip().lower() in _CHAR_TAG_OVERRIDE_MAP
    ]
    if not characters or not override_rules:
        return characters

    # 변환 대상 캐릭터를 얕은 복사하여 원본 보존.
    transformed = [dict(c) for c in characters]

    for rule in override_rules:
        rule_type = (rule.get("type") or "").strip().lower()
        field_key, action, label = _CHAR_TAG_OVERRIDE_MAP[rule_type]
        trigger = (rule.get("trigger") or "").strip()
        if not trigger:
            print(
                f"[WORD_RULE] 캐릭터 태그 규칙(type={rule_type})에 trigger가 없어 스킵합니다."
            )
            continue
        if not re.search(re.escape(trigger), trigger_text or "", flags=re.IGNORECASE):
            continue

        for char_data in transformed:
            char_name = char_data.get("name", "")
            before = char_data.get(field_key, "")
            if action == "remove":
                if not before:
                    continue
                char_data[field_key] = ""
                print(
                    f"[WORD_RULE] {label} 적용: char={char_name}, "
                    f"trigger={trigger!r}, 이전 {field_key}={before!r}"
                )
            else:  # replace
                target = rule.get("target", "")
                char_data[field_key] = target
                print(
                    f"[WORD_RULE] {label} 적용: char={char_name}, "
                    f"trigger={trigger!r}, 이전 {field_key}={before!r}, "
                    f"새 {field_key}={target!r}"
                )

    return transformed
