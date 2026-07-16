"""삽화 모드 단어 기반 규칙 처리.

RAW 프롬프트의 섹션 구조를 유지하면서 다음 범위에 규칙을 적용한다.

- NAME: 치환 규칙만 적용
- SPEAK: 구조화된 각 발화 줄의 ``NAME:`` 부분에만 치환 규칙 적용
- SETUP/CHAR/SUPPLEMENT: 치환과 제거 규칙 모두 적용
"""

import re


_SECTION_MARKER_RE = re.compile(r"^\[([A-Za-z][A-Za-z0-9_]*)\]", re.IGNORECASE | re.MULTILINE)
_PROMPT_RULE_SECTIONS = {"setup", "char", "supplement"}
_REPLACE_ONLY_SECTIONS = {"name"}


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


def apply_replacement_rules(text: str, rules: list[dict]) -> tuple[str, int]:
    """활성화된 치환 규칙만 적용하고 실제 적용된 규칙 수를 반환한다."""
    if not text:
        return text, 0

    result = text
    applied = 0
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        if (rule.get("type") or "replace").strip().lower() == "remove":
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
        if rule_type == "remove":
            positive, did_positive = apply_remove_rule(positive, rule)
            negative, did_negative = apply_remove_rule(negative, rule)
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
    speaker_re = re.compile(r'^(?P<indent>\s*)(?P<speaker>[A-Za-z0-9_]+)(?P<separator>\s*:\s*)(?=["(])')

    for line in speak_text.splitlines(keepends=True):
        match = speaker_re.match(line)
        if not match:
            output_lines.append(line)
            continue

        replaced_speaker, applied = apply_replacement_rules(match.group("speaker"), rules)
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
