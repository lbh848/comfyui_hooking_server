"""Presentation-only variable positions; the LLM still receives ordinary text."""
from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar


_symbolic = ContextVar("illustration_symbolic_prompts", default=False)


@contextmanager
def symbolic_prompts():
    """Render the same builders using named placeholders, without execution data."""
    token = _symbolic.set(True)
    try:
        yield
    finally:
        _symbolic.reset(token)


def is_symbolic():
    return _symbolic.get()


class PromptText(str):
    def __new__(cls, value="", spans=()):
        result = super().__new__(cls, value)
        result.variable_spans = tuple(spans)
        return result

    def __add__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        spans = self.variable_spans + tuple(
            (start + len(self), end + len(self), label)
            for start, end, label in getattr(other, "variable_spans", ())
        )
        return PromptText(str.__add__(self, other), spans)

    def __radd__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return PromptText(other) + self

    def __getitem__(self, key):
        value = str.__getitem__(self, key)
        if not isinstance(key, slice):
            return value
        left, right, step = key.indices(len(self))
        if step != 1:
            return value
        spans = [(max(start, left) - left, min(end, right) - left, label)
                 for start, end, label in self.variable_spans if max(start, left) < min(end, right)]
        return PromptText(value, spans)

    def strip(self, chars=None):
        left = len(self) - len(str.lstrip(self, chars))
        right = len(str.rstrip(self, chars))
        return self[left:max(left, right)]

    def rstrip(self, chars=None):
        return self[:len(str.rstrip(self, chars))]

    def replace(self, old, new, count=-1):
        if not old:
            return str.replace(self, old, new, count)
        parts, cursor, replaced = [], 0, 0
        while count < 0 or replaced < count:
            index = self.find(old, cursor)
            if index < 0:
                break
            parts.extend((self[cursor:index], new))
            cursor = index + len(old)
            replaced += 1
        parts.append(self[cursor:])
        return prompt_join("", parts)


def variable(value, label):
    if is_symbolic():
        if label == "프로필·복장 기준":
            return prompt_join(" · ", (variable("", item) for item in (
                "캐릭터 얼굴·눈·절대·부정 태그", "lb-xnai.lb.extra 외형·복장",
            )))
        if label == "이 호출의 장면·프로필·상태":
            # Explain the contents of DETAIL's existing scene bundle in the
            # preview. Runtime serialization and its variable span stay intact.
            return prompt_join(" · ", (variable("", item) for item in (
                "선택된 장면", "현재 대화의 등록 인물", "캐릭터 프로필", "복장·머리 상태", "시간·장소",
            )))
        label = {
            "과거 대화": "과거 문맥 입력",
            "대사 귀속용 과거 대화": "과거 문맥 입력",
            "시점별 확정 프로필": "캐릭터 프로필",
            "복장·머리 상태 타임라인": "복장·머리 상태",
            "시간·장소 상태 타임라인": "시간·장소",
            "슬롯이 삽입된 현재 원문": "현재 원문과 구간",
            "역번역 변수: {character_names}": "보호할 캐릭터 이름",
            "역번역 변수: {chunk_index}": "역번역 묶음 번호",
            "역번역 변수: {chunk_total}": "역번역 묶음 수",
        }.get(label, label)
        value = "{" + label + "}"
    text = str(value)
    return PromptText(text, ((0, len(text), label),) if text else ())


def prompt_join(separator, parts):
    result = PromptText()
    for index, part in enumerate(parts):
        if index:
            result += separator
        result += part
    return result


def prompt_sub(pattern, replacement, text):
    """Retain insertion positions while existing template macros are rendered."""
    text = text if isinstance(text, PromptText) else PromptText(text)
    parts, cursor = [], 0
    for match in re.finditer(pattern, text):
        parts.extend((text[cursor:match.start()], replacement(match) if callable(replacement) else match.expand(replacement)))
        cursor = match.end()
    parts.append(text[cursor:])
    return prompt_join("", parts)


def symbolic_call2_template(text):
    """Show conditional template branches without choosing actual settings."""
    labels = {
        "lb-xnai.nsfw": "NSFW 설정", "lb-xnai.supplement": "보충 묘사 설정",
        "lb-xnai.kv.off": "대표 이미지 설정", "lb-xnai.background.minimal": "배경 묘사 설정",
        "lb-xnai.compat.comfy": "Comfy 호환 설정", "lb-xnai.compat.charPrompt": "캐릭터 프롬프트 형식",
        "lb-xnai.context": "과거 문맥 사용 설정", "lb-xnai.characters": "인물 수 설정",
        "lb-xnai-history": "캐릭터 태그 이력",
    }
    text = PromptText(text)
    def getter(match):
        key = match.group(1).removeprefix("toggle_")
        return variable("", labels.get(key, key))
    text = prompt_sub(r"\{\{(?:getglobalvar|getvar)::([^{}]+)\}\}", getter, text)
    def condition(match):
        expr = match.group(1)
        for key, label in labels.items():
            expr = expr.replace(key, label)
        expr = expr.removeprefix("toggle::").removeprefix("keep::")
        return "\n[조건: " + variable("", expr) + " 충족 시]\n"
    text = prompt_sub(r"\{\{#when(?:::|\s+)([^{}]*?)\}\}", condition, text)
    text = text.replace("{{:else}}", "\n[그 외 조건]\n")
    text = prompt_sub(r"\{\{/(?:when|compat-comfy|nsfw|supplement|context|history-length|history-null)\}\}", "\n[조건 끝]\n", text)
    return text.strip()


def input_variables(messages):
    if not any(isinstance(message.get("content"), PromptText) for message in messages):
        return None
    result = []
    for message in messages:
        text = message.get("content") or ""
        result.append([{
            "start": len(text[:start].encode("utf-16-le")) // 2,
            "end": len(text[:end].encode("utf-16-le")) // 2,
            "label": label,
        } for start, end, label in getattr(text, "variable_spans", ())])
    return result
