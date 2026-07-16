"""챈섭(NAI 호환)용 삽화 프롬프트 빌더.

로컬 ComfyUI 워크플로우 제어 블럭이나 LoRA 트리거는 만들지 않고,
삽화 모드의 ANIMA 프리셋과 장면 섹션만 평탄한 POSITIVE/NEGATIVE로 만든다.
"""

from __future__ import annotations

import re


def _as_tags(value) -> list[str]:
    """프리셋 값을 정리된 태그 리스트로 변환한다."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _join_parts(*parts) -> str:
    return ", ".join(str(part).strip() for part in parts if str(part).strip())


def convert_to_nai_syntax(prompt: str) -> str:
    """PocketRisu와 동일하게 비이스케이프 괄호 가중치를 NAI 중괄호로 바꾼다.

    ``\\(``/``\\)``는 캐릭터명 같은 리터럴 괄호이므로 그대로 보존한다.
    """
    if not prompt:
        return ""
    converted = re.sub(r"\s*\|\s*", ", ", prompt)
    converted = (
        converted.replace(r"\(", "\ue000")
        .replace(r"\)", "\ue001")
        .replace("(", "{")
        .replace(")", "}")
        .replace("\ue000", "(")
        .replace("\ue001", ")")
    )
    return re.sub(r",\s*,+", ",", converted).strip(" ,")


class ChansubPromptBuilder:
    """삽화 설정을 NAI 호환 POSITIVE/NEGATIVE로 평탄화한다."""

    @staticmethod
    def build_positive_prompt(
        setup: str,
        char: str,
        supplement: str,
        tags: dict,
        settings: dict,
    ) -> str:
        artist_presets = tags.get("artist_presets", {}) or {}
        quality_presets = tags.get("quality_presets", {}) or {}

        artist_name = settings.get("anima_artist_preset", "")
        quality_name = settings.get("anima_quality_preset", "")

        artist_tags = _as_tags(artist_presets.get(artist_name, []))
        if quality_name and quality_name in quality_presets:
            quality_tags = _as_tags(quality_presets.get(quality_name, []))
        else:
            quality_tags = _as_tags(tags.get("anima_quality", []))

        positive = _join_parts(
            ", ".join(quality_tags),
            ", ".join(artist_tags),
            setup,
            char,
            supplement,
        )
        return convert_to_nai_syntax(positive)

    @staticmethod
    def build_negative_prompt(tags: dict, settings: dict) -> str:
        negative_presets = tags.get("negative_presets", {}) or {}
        preset_name = settings.get("anima_negative_preset", "")
        if preset_name and preset_name in negative_presets:
            negative_tags = _as_tags(negative_presets.get(preset_name, []))
        else:
            negative_tags = _as_tags(tags.get("anima_negative", []))
        return convert_to_nai_syntax(", ".join(negative_tags))

    def build(
        self,
        setup: str,
        char: str,
        supplement: str,
        tags: dict,
        settings: dict,
    ) -> dict:
        return {
            "positive": self.build_positive_prompt(setup, char, supplement, tags, settings),
            "negative": self.build_negative_prompt(tags, settings),
            "width": int(settings.get("img_w", 756) or 756),
            "height": int(settings.get("img_h", 756) or 756),
        }
