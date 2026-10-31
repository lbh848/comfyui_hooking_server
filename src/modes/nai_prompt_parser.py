"""NovelAI 프롬프트를 Canonical AST로 해석하고 대상 문법으로 변환한다.

NAI 문법 해석과 모델별 출력 보정을 분리한다. 파서는 중첩 강조, 숫자 강조,
랜덤 선택, V4 캐릭터 구간, 관계 지시문, 프롬프트 청크를 구조로 보존한다.
ANIMA 어댑터는 그 AST를 Comfy 명시적 weight 문법으로 출력하며 최종 weight의
절댓값을 설정 한도에서 clamp한 뒤 Decimal ROUND_HALF_UP으로 소수 첫째 자리에
맞춘다. 반올림 전 값은 Canonical AST와 fragment provenance에 계속 보존한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
import traceback
from typing import Any


NAI_WEIGHT_STEP = Decimal("1.05")
DEFAULT_MAX_ABS_WEIGHT = Decimal("1.5")
ANIMA_WEIGHT_QUANTUM = Decimal("0.1")

_NUMBER_TEXT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMERIC_EMPHASIS_RE = re.compile(rf"(?P<weight>{_NUMBER_TEXT})::")
_EXPLICIT_PAREN_WEIGHT_RE = re.compile(
    rf"(?s)^(?P<body>.*):\s*(?P<weight>{_NUMBER_TEXT})\s*$"
)
_PROMPT_CHUNK_RE = re.compile(r"!macro:([^!]+)!")
_ARTIST_TAG_RE = re.compile(r"(?i)^\s*artist:\s*(?P<name>.+?)\s*$")
_RELATION_RE = re.compile(r"(?i)^\s*(?P<role>source|target|mutual)#(?P<action>.+?)\s*$")
_TEXT_DIRECTIVE_RE = re.compile(r"(?is)^\s*text:\s*(?P<text>.+?)\s*$")


@dataclass
class CanonicalNode:
    kind: str
    raw: str
    start: int = 0
    end: int = 0
    text: str = ""
    syntax: str = ""
    factor: Decimal | None = None
    name: str = ""
    children: list["CanonicalNode"] = field(default_factory=list)
    options: list["CanonicalNode"] = field(default_factory=list)


@dataclass
class CanonicalRegion:
    role: str
    source_text: str
    ast: CanonicalNode


@dataclass
class CanonicalPrompt:
    source_text: str
    expanded_text: str
    regions: list[CanonicalRegion]
    warnings: list[dict]


class NAIParserError(ValueError):
    """입력 형식 자체가 파싱 불가능할 때 사용하는 오류."""


def _warning(code: str, message: str, **extra) -> dict:
    warning = {"code": code, "message": message, **extra}
    print(f"[NAI_PARSER] {code}: {message} extra={extra!r}")
    return warning


def _decimal_text(value: Decimal, *, places: int = 8) -> str:
    if value.is_zero():
        return "0"
    quantizer = Decimal(1).scaleb(-places)
    rounded = value.quantize(quantizer)
    result = format(rounded, "f").rstrip("0").rstrip(".")
    return result or "0"


def _is_escaped(text: str, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 1


def _find_matching_bracket(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if _is_escaped(text, index):
            continue
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _numeric_prefix_at(text: str, index: int) -> re.Match | None:
    match = _NUMERIC_EMPHASIS_RE.match(text, index)
    if match is None:
        return None
    if index > 0 and text[index - 1] not in " \t\r\n,([{|":
        return None
    return match


def _find_numeric_close(text: str, body_start: int) -> int:
    cursor = body_start
    nested = 1
    while cursor < len(text) - 1:
        if _is_escaped(text, cursor):
            cursor += 1
            continue
        nested_match = _numeric_prefix_at(text, cursor)
        if nested_match is not None:
            nested += 1
            cursor = nested_match.end()
            continue
        if text.startswith("::", cursor):
            nested -= 1
            if nested == 0:
                return cursor
            cursor += 2
            continue
        cursor += 1
    return -1


def _find_randomizer_close(text: str, body_start: int) -> int:
    cursor = body_start
    while cursor < len(text) - 1:
        if text.startswith("||", cursor) and not _is_escaped(text, cursor):
            return cursor
        cursor += 1
    return -1


def _split_top_level_single_pipes(text: str) -> list[str]:
    """||randomizer|| 내부와 괄호 내부를 제외한 단일 |만 V4 구간으로 나눈다."""
    result: list[str] = []
    start = 0
    cursor = 0
    stack: list[str] = []
    closing = {"{": "}", "[": "]", "(": ")"}
    while cursor < len(text):
        if _is_escaped(text, cursor):
            cursor += 2
            continue
        if text.startswith("||", cursor):
            end = _find_randomizer_close(text, cursor + 2)
            if end < 0:
                cursor += 2
            else:
                cursor = end + 2
            continue
        numeric = _numeric_prefix_at(text, cursor)
        if numeric is not None:
            end = _find_numeric_close(text, numeric.end())
            if end >= 0:
                cursor = end + 2
                continue
        char = text[cursor]
        if char in closing:
            # 이모티콘의 :( 같은 괄호는 닫힘이 없으면 문법 괄호로 보지 않는다.
            if _find_matching_bracket(text, cursor, char, closing[char]) >= 0:
                stack.append(char)
        elif char in "}])":
            if stack and closing[stack[-1]] == char:
                stack.pop()
        elif char == "|" and not stack:
            result.append(text[start:cursor])
            start = cursor + 1
        cursor += 1
    result.append(text[start:])
    return result


def _split_randomizer_options(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    cursor = 0
    stack: list[str] = []
    closing = {"{": "}", "[": "]", "(": ")"}
    while cursor < len(text):
        if _is_escaped(text, cursor):
            cursor += 2
            continue
        char = text[cursor]
        if char in closing:
            if _find_matching_bracket(text, cursor, char, closing[char]) >= 0:
                stack.append(char)
        elif char in "}])":
            if stack and closing[stack[-1]] == char:
                stack.pop()
        elif char == "|" and not stack:
            result.append(text[start:cursor])
            start = cursor + 1
        cursor += 1
    result.append(text[start:])
    return result


def _expand_prompt_chunks(
    text: str,
    prompt_chunks: dict[str, str],
    warnings: list[dict],
) -> str:
    if not prompt_chunks or "!macro:" not in text:
        if "!macro:" in text:
            for match in _PROMPT_CHUNK_RE.finditer(text):
                warnings.append(_warning(
                    "unresolved_prompt_chunk",
                    f"NAI Prompt Chunk를 찾을 수 없습니다: {match.group(0)}",
                    name=match.group(1).strip(),
                    offset=match.start(),
                ))
        return text

    expanded = text
    seen_states = {expanded}
    warned_missing: set[str] = set()
    for _ in range(32):
        changed = False

        def replace(match: re.Match) -> str:
            nonlocal changed
            name = match.group(1).strip()
            replacement = prompt_chunks.get(name)
            if replacement is None:
                if name not in warned_missing:
                    warnings.append(_warning(
                        "unresolved_prompt_chunk",
                        f"NAI Prompt Chunk를 찾을 수 없습니다: {match.group(0)}",
                        name=name,
                        offset=match.start(),
                    ))
                    warned_missing.add(name)
                return match.group(0)
            changed = True
            return replacement

        candidate = _PROMPT_CHUNK_RE.sub(replace, expanded)
        if not changed:
            return candidate
        if candidate == expanded:
            warnings.append(_warning(
                "cyclic_prompt_chunk",
                "NAI Prompt Chunk가 자기 자신을 참조해 확장을 중단했습니다.",
            ))
            return candidate
        if candidate in seen_states:
            warnings.append(_warning(
                "cyclic_prompt_chunk",
                "NAI Prompt Chunk가 순환 참조되어 확장을 중단했습니다.",
            ))
            return candidate
        seen_states.add(candidate)
        expanded = candidate

    warnings.append(_warning(
        "prompt_chunk_depth_exceeded",
        "NAI Prompt Chunk 확장이 32단계를 초과해 중단되었습니다.",
    ))
    return expanded


class _RegionParser:
    def __init__(self, text: str, warnings: list[dict], *, base_offset: int = 0):
        self.text = text
        self.warnings = warnings
        self.base_offset = base_offset
        self.cursor = 0

    def parse(self) -> CanonicalNode:
        children = self._parse_sequence()
        return CanonicalNode(
            kind="sequence",
            raw=self.text,
            start=self.base_offset,
            end=self.base_offset + len(self.text),
            children=children,
        )

    def _parse_subtext(self, text: str, offset: int) -> CanonicalNode:
        return _RegionParser(text, self.warnings, base_offset=offset).parse()

    def _parse_sequence(self) -> list[CanonicalNode]:
        nodes: list[CanonicalNode] = []
        buffer_start = self.cursor

        def flush(until: int) -> None:
            nonlocal buffer_start
            if until <= buffer_start:
                return
            raw = self.text[buffer_start:until]
            nodes.append(CanonicalNode(
                kind="text",
                raw=raw,
                text=raw,
                start=self.base_offset + buffer_start,
                end=self.base_offset + until,
            ))

        while self.cursor < len(self.text):
            start = self.cursor
            if _is_escaped(self.text, start):
                self.cursor += 2
                continue

            if self.text.startswith("||", start):
                close = _find_randomizer_close(self.text, start + 2)
                if close < 0:
                    self.warnings.append(_warning(
                        "unclosed_randomizer",
                        "닫히지 않은 NAI Randomizer를 원문 그대로 유지합니다.",
                        offset=self.base_offset + start,
                    ))
                    self.cursor += 2
                    continue
                flush(start)
                body = self.text[start + 2:close]
                options = [
                    self._parse_subtext(option, self.base_offset + start + 2)
                    for option in _split_randomizer_options(body)
                ]
                end = close + 2
                nodes.append(CanonicalNode(
                    kind="randomizer",
                    raw=self.text[start:end],
                    start=self.base_offset + start,
                    end=self.base_offset + end,
                    options=options,
                ))
                self.cursor = end
                buffer_start = self.cursor
                continue

            numeric = _numeric_prefix_at(self.text, start)
            if numeric is not None:
                close = _find_numeric_close(self.text, numeric.end())
                if close < 0:
                    self.warnings.append(_warning(
                        "unclosed_numeric_emphasis",
                        "닫히지 않은 NAI 숫자 emphasis를 원문 그대로 유지합니다.",
                        offset=self.base_offset + start,
                    ))
                    self.cursor = numeric.end()
                    continue
                flush(start)
                try:
                    factor = Decimal(numeric.group("weight"))
                except InvalidOperation as exc:
                    print(
                        "[NAI_PARSER] 숫자 emphasis 변환 실패: "
                        f"value={numeric.group('weight')!r}, offset={self.base_offset + start}"
                    )
                    traceback.print_exc()
                    raise NAIParserError("NAI 숫자 emphasis 값이 올바르지 않습니다.") from exc
                body_start = numeric.end()
                body = self.text[body_start:close]
                end = close + 2
                nodes.append(CanonicalNode(
                    kind="emphasis",
                    raw=self.text[start:end],
                    start=self.base_offset + start,
                    end=self.base_offset + end,
                    syntax="numeric_double_colon",
                    factor=factor,
                    children=[self._parse_subtext(body, self.base_offset + body_start)],
                ))
                self.cursor = end
                buffer_start = self.cursor
                continue

            char = self.text[start]
            if char in "{[":
                closing = "}" if char == "{" else "]"
                close = _find_matching_bracket(self.text, start, char, closing)
                if close < 0:
                    self.warnings.append(_warning(
                        "unclosed_nai_emphasis",
                        f"닫히지 않은 NAI 강조 기호 '{char}'를 원문 그대로 유지합니다.",
                        delimiter=char,
                        offset=self.base_offset + start,
                    ))
                    self.cursor += 1
                    continue
                flush(start)
                body = self.text[start + 1:close]
                factor = NAI_WEIGHT_STEP if char == "{" else Decimal(1) / NAI_WEIGHT_STEP
                nodes.append(CanonicalNode(
                    kind="emphasis",
                    raw=self.text[start:close + 1],
                    start=self.base_offset + start,
                    end=self.base_offset + close + 1,
                    syntax="nai_curly" if char == "{" else "nai_square",
                    factor=factor,
                    children=[self._parse_subtext(body, self.base_offset + start + 1)],
                ))
                self.cursor = close + 1
                buffer_start = self.cursor
                continue

            if char == "(":
                close = _find_matching_bracket(self.text, start, "(", ")")
                if close >= 0:
                    body = self.text[start + 1:close]
                    explicit = _EXPLICIT_PAREN_WEIGHT_RE.match(body)
                    if explicit is not None:
                        flush(start)
                        try:
                            factor = Decimal(explicit.group("weight"))
                        except InvalidOperation as exc:
                            print(
                                "[NAI_PARSER] 괄호 숫자 weight 변환 실패: "
                                f"value={explicit.group('weight')!r}, offset={self.base_offset + start}"
                            )
                            traceback.print_exc()
                            raise NAIParserError("괄호 weight 값이 올바르지 않습니다.") from exc
                        body_text = explicit.group("body")
                        nodes.append(CanonicalNode(
                            kind="emphasis",
                            raw=self.text[start:close + 1],
                            start=self.base_offset + start,
                            end=self.base_offset + close + 1,
                            syntax="explicit_parenthesis",
                            factor=factor,
                            children=[self._parse_subtext(
                                body_text,
                                self.base_offset + start + 1,
                            )],
                        ))
                        self.cursor = close + 1
                        buffer_start = self.cursor
                        continue
                    # muji (uimss), watercolor (medium) 같은 태그 이름 일부.
                    self.cursor = close + 1
                    continue

            if self.text.startswith("!macro:", start):
                end = self.text.find("!", start + len("!macro:"))
                if end >= 0:
                    flush(start)
                    raw = self.text[start:end + 1]
                    nodes.append(CanonicalNode(
                        kind="macro",
                        raw=raw,
                        name=raw[len("!macro:"):-1].strip(),
                        start=self.base_offset + start,
                        end=self.base_offset + end + 1,
                    ))
                    self.cursor = end + 1
                    buffer_start = self.cursor
                    continue

            if char in "}]":
                self.warnings.append(_warning(
                    "unexpected_nai_closing",
                    f"짝이 없는 NAI 닫는 기호 '{char}'를 원문 그대로 유지합니다.",
                    delimiter=char,
                    offset=self.base_offset + start,
                ))
            self.cursor += 1

        flush(len(self.text))
        return nodes


def parse_nai_prompt(
    text: str,
    *,
    prompt_chunks: dict[str, str] | None = None,
) -> CanonicalPrompt:
    if not isinstance(text, str):
        print(f"[NAI_PARSER] 입력 형식 오류: type={type(text).__name__}")
        raise NAIParserError("NAI 프롬프트는 문자열이어야 합니다.")
    warnings: list[dict] = []
    valid_chunks: dict[str, str] = {}
    for name, value in (prompt_chunks or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            warnings.append(_warning(
                "invalid_prompt_chunk",
                "문자열이 아닌 NAI Prompt Chunk를 제외했습니다.",
                name=repr(name),
                value_type=type(value).__name__,
            ))
            continue
        valid_chunks[name.strip()] = value
    expanded = _expand_prompt_chunks(text, valid_chunks, warnings)
    raw_regions = _split_top_level_single_pipes(expanded)
    if len(raw_regions) > 1:
        warnings.append(_warning(
            "v4_multi_character_prompt",
            f"단일 | 구분자를 NAI V4 base/character 구간 {len(raw_regions)}개로 해석했습니다.",
            region_count=len(raw_regions),
        ))
    regions = []
    for index, region_text in enumerate(raw_regions):
        role = "base" if index == 0 else f"character_{index}"
        regions.append(CanonicalRegion(
            role=role,
            source_text=region_text,
            ast=_RegionParser(region_text, warnings).parse(),
        ))
    return CanonicalPrompt(
        source_text=text,
        expanded_text=expanded,
        regions=regions,
        warnings=warnings,
    )


def _node_to_dict(node: CanonicalNode) -> dict:
    result: dict[str, Any] = {
        "kind": node.kind,
        "raw": node.raw,
        "start": node.start,
        "end": node.end,
    }
    if node.text:
        result["text"] = node.text
    if node.syntax:
        result["syntax"] = node.syntax
    if node.factor is not None:
        result["factor"] = _decimal_text(node.factor)
    if node.name:
        result["name"] = node.name
    if node.children:
        result["children"] = [_node_to_dict(child) for child in node.children]
    if node.options:
        result["options"] = [_node_to_dict(option) for option in node.options]
    return result


def canonical_prompt_to_dict(prompt: CanonicalPrompt) -> dict:
    return {
        "kind": "prompt",
        "source_syntax": "nai",
        "source_text": prompt.source_text,
        "expanded_text": prompt.expanded_text,
        "regions": [
            {
                "role": region.role,
                "source_text": region.source_text,
                "ast": _node_to_dict(region.ast),
            }
            for region in prompt.regions
        ],
    }


def _uniform_content(node: CanonicalNode) -> tuple[str, Decimal, list[str]] | None:
    if node.kind == "text":
        return node.text, Decimal(1), []
    if node.kind == "sequence":
        parts = []
        effective_factor: Decimal | None = None
        syntax_path: list[str] | None = None
        for child in node.children:
            uniform = _uniform_content(child)
            if uniform is None:
                return None
            text, factor, child_path = uniform
            if not text.strip():
                parts.append(text)
                continue
            if effective_factor is None:
                effective_factor = factor
                syntax_path = child_path
            elif factor != effective_factor or child_path != syntax_path:
                return None
            parts.append(text)
        return "".join(parts), effective_factor or Decimal(1), syntax_path or []
    if node.kind == "emphasis" and len(node.children) == 1:
        uniform = _uniform_content(node.children[0])
        if uniform is None:
            return None
        text, factor, path = uniform
        return text, factor * (node.factor or Decimal(1)), [node.syntax, *path]
    return None


def _split_plain_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    start = 0
    cursor = 0
    parenthesis_depth = 0
    while cursor < len(text):
        if _is_escaped(text, cursor):
            cursor += 2
            continue
        char = text[cursor]
        if char == "(":
            if _find_matching_bracket(text, cursor, "(", ")") >= 0:
                parenthesis_depth += 1
        elif char == ")" and parenthesis_depth:
            parenthesis_depth -= 1
        elif char in ",\r\n" and parenthesis_depth == 0:
            value = text[start:cursor].strip()
            if value:
                fragments.append(value)
            start = cursor + 1
        cursor += 1
    value = text[start:].strip()
    if value:
        fragments.append(value)
    return fragments


def _normalize_artist_text(text: str) -> tuple[str, list[str]]:
    parts = _split_plain_fragments(text)
    if not parts:
        return text.strip(), []
    normalized = []
    artists = []
    for part in parts:
        match = _ARTIST_TAG_RE.match(part)
        if match is None:
            normalized.append(part)
            continue
        name = match.group("name").strip()
        normalized.append(f"@{name}")
        artists.append(name)
    return ", ".join(normalized), artists


def _clamp_weight(value: Decimal, maximum: Decimal) -> tuple[Decimal, bool]:
    if value > maximum:
        return maximum, True
    if value < -maximum:
        return -maximum, True
    return value, False


def _target_weight_text(weight: Decimal, target: str) -> str:
    if target == "anima":
        return format(weight.quantize(ANIMA_WEIGHT_QUANTUM, rounding=ROUND_HALF_UP), ".1f")
    return _decimal_text(weight, places=4)


def _render_weighted(text: str, weight: Decimal, target: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    if weight == Decimal(1):
        return clean
    return f"({clean}:{_target_weight_text(weight, target)})"


def _escape_literal_parentheses(text: str) -> str:
    """AST에서 weight가 아닌 것으로 확정된 괄호만 Comfy 리터럴로 escape한다."""
    result: list[str] = []
    for index, char in enumerate(text):
        if char in "()" and not _is_escaped(text, index):
            result.append("\\")
        result.append(char)
    return "".join(result)


def adapt_prompt(
    prompt: CanonicalPrompt,
    *,
    target: str = "anima",
    max_abs_weight: Decimal | float | str = DEFAULT_MAX_ABS_WEIGHT,
    randomizer_strategy: str = "dynamic_prompt",
) -> dict:
    if target not in {"anima", "sdxl"}:
        print(f"[NAI_ADAPTER] 지원하지 않는 대상: target={target!r}")
        raise NAIParserError(f"지원하지 않는 프롬프트 대상입니다: {target}")
    try:
        maximum = Decimal(str(max_abs_weight))
    except InvalidOperation as exc:
        print(f"[NAI_ADAPTER] 최대 weight 변환 실패: value={max_abs_weight!r}")
        traceback.print_exc()
        raise NAIParserError("최대 weight 값이 올바르지 않습니다.") from exc
    if maximum <= 0:
        print(f"[NAI_ADAPTER] 최대 weight 범위 오류: value={maximum}")
        raise NAIParserError("최대 weight는 0보다 커야 합니다.")
    if randomizer_strategy not in {"dynamic_prompt", "first"}:
        print(f"[NAI_ADAPTER] Randomizer 전략 오류: strategy={randomizer_strategy!r}")
        raise NAIParserError("지원하지 않는 Randomizer 변환 방식입니다.")

    warnings = list(prompt.warnings)
    output_fragments: list[dict] = []
    target_regions: list[dict] = []

    def make_fragment(
        source_text: str,
        content: str,
        raw_weight: Decimal,
        *,
        region_role: str,
        syntax_path: list[str],
        kind: str = "tag",
        extra: dict | None = None,
    ) -> dict | None:
        normalized, artists = _normalize_artist_text(content)
        if not normalized:
            return None
        clamped_weight, clamped = _clamp_weight(raw_weight, maximum)
        weight = (
            clamped_weight.quantize(ANIMA_WEIGHT_QUANTUM, rounding=ROUND_HALF_UP)
            if target == "anima"
            else clamped_weight
        )
        relation = _RELATION_RE.match(normalized)
        text_directive = _TEXT_DIRECTIVE_RE.match(normalized)
        metadata: dict[str, Any] = {
            "source_syntax": "nai",
            "target_syntax": target,
            "region": region_role,
            "syntax_path": syntax_path,
            "raw_weight": _decimal_text(raw_weight),
            "clamped_weight": _decimal_text(clamped_weight),
            "weight": _target_weight_text(weight, target),
            "clamped": clamped,
            "rounded": weight != clamped_weight,
            "weight_quantum": "0.1" if target == "anima" else None,
            "artist_names": artists,
        }
        if extra:
            metadata.update(extra)
        output_kind = kind
        target_content = normalized
        if relation is not None:
            output_kind = "relation"
            metadata["relation"] = {
                "role": relation.group("role").lower(),
                "action": relation.group("action").strip(),
            }
            target_content = relation.group("action").strip()
            warnings.append(_warning(
                "relation_requires_structured_target",
                f"{relation.group('role')}# 관계를 구조로 보존했습니다. 현재 평면 프리셋 저장 전 확인이 필요합니다.",
                region=region_role,
                source=source_text,
            ))
        elif text_directive is not None:
            output_kind = "text_directive"
            metadata["text_directive"] = text_directive.group("text").strip()
            warnings.append(_warning(
                "text_directive_requires_review",
                "NAI Text: 지시문을 별도 directive로 보존했습니다.",
                region=region_role,
                source=source_text,
            ))
        if raw_weight < 0:
            warnings.append(_warning(
                "negative_emphasis_preserved",
                "음수 emphasis를 보존했습니다. ANIMA 결과를 직접 확인해주세요.",
                source=source_text,
                weight=_target_weight_text(weight, target),
            ))
        if clamped:
            warnings.append(_warning(
                "emphasis_weight_clamped",
                f"최종 emphasis {_decimal_text(raw_weight)}를 대상 한도 {_decimal_text(maximum)}로 제한했습니다.",
                source=source_text,
                raw_weight=_decimal_text(raw_weight),
                weight=_decimal_text(weight),
            ))
        adapter_content = (
            target_content
            if output_kind == "randomizer"
            else _escape_literal_parentheses(target_content)
        )
        rendered = _render_weighted(adapter_content, weight, target)
        return {
            "source_text": source_text.strip(),
            "text": rendered,
            "kind": output_kind,
            "weight": _target_weight_text(weight, target),
            "raw_weight": _decimal_text(raw_weight),
            "clamped_weight": _decimal_text(clamped_weight),
            "changed": source_text.strip() != rendered,
            "metadata": metadata,
        }

    def render_node(
        node: CanonicalNode,
        inherited: Decimal,
        region_role: str,
        syntax_path: list[str],
    ) -> list[dict]:
        if node.kind == "sequence":
            rendered: list[dict] = []
            for child in node.children:
                rendered.extend(render_node(child, inherited, region_role, syntax_path))
            return rendered
        if node.kind == "text":
            return [
                fragment
                for value in _split_plain_fragments(node.text)
                if (fragment := make_fragment(
                    value,
                    value,
                    inherited,
                    region_role=region_role,
                    syntax_path=syntax_path,
                )) is not None
            ]
        if node.kind == "macro":
            fragment = make_fragment(
                node.raw,
                node.raw,
                inherited,
                region_role=region_role,
                syntax_path=[*syntax_path, "unresolved_prompt_chunk"],
                kind="macro",
                extra={"macro_name": node.name},
            )
            return [fragment] if fragment else []
        if node.kind == "emphasis":
            uniform = _uniform_content(node)
            if uniform is not None:
                content, factor, path = uniform
                fragment = make_fragment(
                    node.raw,
                    content,
                    inherited * factor,
                    region_role=region_role,
                    syntax_path=[*syntax_path, *path],
                )
                return [fragment] if fragment else []
            rendered: list[dict] = []
            next_weight = inherited * (node.factor or Decimal(1))
            next_path = [*syntax_path, node.syntax]
            for child in node.children:
                rendered.extend(render_node(child, next_weight, region_role, next_path))
            return rendered
        if node.kind == "randomizer":
            options = []
            for option in node.options:
                values = render_node(option, Decimal(1), region_role, [])
                options.append(", ".join(value["text"] for value in values if value["text"]))
            options = [option for option in options if option]
            if not options:
                warnings.append(_warning(
                    "empty_randomizer",
                    "비어 있는 NAI Randomizer를 제외했습니다.",
                    source=node.raw,
                ))
                return []
            target_content = (
                "{" + "|".join(options) + "}"
                if randomizer_strategy == "dynamic_prompt"
                else options[0]
            )
            warnings.append(_warning(
                "randomizer_adapted",
                (
                    "NAI Randomizer를 Comfy Dynamic Prompt 후보로 변환했습니다."
                    if randomizer_strategy == "dynamic_prompt"
                    else "NAI Randomizer의 첫 번째 후보를 결정적으로 선택했습니다."
                ),
                source=node.raw,
                option_count=len(options),
                strategy=randomizer_strategy,
            ))
            fragment = make_fragment(
                node.raw,
                target_content,
                inherited,
                region_role=region_role,
                syntax_path=[*syntax_path, "nai_randomizer"],
                kind="randomizer",
                extra={"randomizer_options": options, "randomizer_strategy": randomizer_strategy},
            )
            return [fragment] if fragment else []
        warnings.append(_warning(
            "unknown_canonical_node",
            f"알 수 없는 Canonical AST 노드를 원문으로 유지합니다: {node.kind}",
            source=node.raw,
        ))
        fragment = make_fragment(
            node.raw,
            node.raw,
            inherited,
            region_role=region_role,
            syntax_path=syntax_path,
            kind="unknown",
        )
        return [fragment] if fragment else []

    for region in prompt.regions:
        fragments = render_node(region.ast, Decimal(1), region.role, [])
        for fragment in fragments:
            fragment["region"] = region.role
            output_fragments.append(fragment)
        target_regions.append({
            "role": region.role,
            "prompt": ", ".join(fragment["text"] for fragment in fragments),
            "fragment_count": len(fragments),
        })

    structured = {
        "base_prompt": target_regions[0]["prompt"] if target_regions else "",
        "character_prompts": [
            {"role": region["role"], "prompt": region["prompt"]}
            for region in target_regions[1:]
        ],
        "relations": [
            fragment["metadata"]["relation"] | {"region": fragment["region"]}
            for fragment in output_fragments
            if "relation" in fragment["metadata"]
        ],
        "text_directives": [
            {
                "region": fragment["region"],
                "text": fragment["metadata"]["text_directive"],
            }
            for fragment in output_fragments
            if "text_directive" in fragment["metadata"]
        ],
    }
    return {
        "target": target,
        "max_abs_weight": _decimal_text(maximum),
        "weight_quantum": "0.1" if target == "anima" else None,
        "weight_rounding": "ROUND_HALF_UP" if target == "anima" else None,
        "prompt": " | ".join(region["prompt"] for region in target_regions),
        "regions": target_regions,
        "fragments": output_fragments,
        "structured": structured,
        "warnings": warnings,
    }


def convert_nai_prompt(
    text: str,
    *,
    prompt_chunks: dict[str, str] | None = None,
    target: str = "anima",
    max_abs_weight: Decimal | float | str = DEFAULT_MAX_ABS_WEIGHT,
    randomizer_strategy: str = "dynamic_prompt",
) -> dict:
    canonical = parse_nai_prompt(text, prompt_chunks=prompt_chunks)
    adapted = adapt_prompt(
        canonical,
        target=target,
        max_abs_weight=max_abs_weight,
        randomizer_strategy=randomizer_strategy,
    )
    return {
        "source_text": text,
        "expanded_text": canonical.expanded_text,
        "canonical": canonical_prompt_to_dict(canonical),
        **adapted,
    }


__all__ = [
    "CanonicalNode",
    "CanonicalPrompt",
    "CanonicalRegion",
    "DEFAULT_MAX_ABS_WEIGHT",
    "NAIParserError",
    "NAI_WEIGHT_STEP",
    "adapt_prompt",
    "canonical_prompt_to_dict",
    "convert_nai_prompt",
    "parse_nai_prompt",
]
