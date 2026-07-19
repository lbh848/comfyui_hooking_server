"""챈섭용 삽화 프롬프트 빌더.

로컬 ComfyUI 워크플로우 제어 블럭이나 LoRA 트리거는 만들지 않고,
삽화 모드에서 선택한 ANIMA/SDXL 프리셋과 장면 섹션만 Comfy 문법의 평탄한
POSITIVE/NEGATIVE로 만든다. HTTP 요청 외형만 NAI API 형식이다.
"""

from __future__ import annotations


def _as_tags(value) -> list[str]:
    """프리셋 값을 정리된 태그 리스트로 변환한다."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _join_parts(*parts) -> str:
    return ", ".join(str(part).strip() for part in parts if str(part).strip())


def _get_workflow_type(settings: dict) -> str:
    workflow_type = str(
        settings.get("chansub_workflow_type", "anima") or "anima"
    ).strip().lower()
    if workflow_type not in ("anima", "sdxl"):
        print(
            f"[CHANSUB_PROMPT] 알 수 없는 워크플로우 계열 "
            f"{workflow_type!r}, anima로 폴백"
        )
        return "anima"
    return workflow_type


def _get_quality_tags(tags: dict, settings: dict) -> list[str]:
    """현재 챈섭 워크플로우와 프리셋에 해당하는 품질 태그를 반환한다."""
    quality_presets = tags.get("quality_presets", {}) or {}
    workflow_type = _get_workflow_type(settings)
    quality_name = settings.get(f"{workflow_type}_quality_preset", "")
    if quality_name and quality_name in quality_presets:
        return _as_tags(quality_presets.get(quality_name, []))
    quality_key = "anima_quality" if workflow_type == "anima" else "quality"
    return _as_tags(tags.get(quality_key, []))


def _get_artist_tags(tags: dict, settings: dict) -> list[str]:
    """현재 챈섭 워크플로우와 프리셋에 해당하는 아티스트 태그를 반환한다."""
    artist_presets = tags.get("artist_presets", {}) or {}
    workflow_type = _get_workflow_type(settings)
    artist_name = settings.get(f"{workflow_type}_artist_preset", "")
    return _as_tags(artist_presets.get(artist_name, []))


class ChansubPromptBuilder:
    """삽화 설정을 Comfy 문법의 POSITIVE/NEGATIVE로 평탄화한다."""

    @staticmethod
    def build_positive_prompt(
        setup: str,
        char: str,
        supplement: str,
        tags: dict,
        settings: dict,
    ) -> str:
        workflow_type = _get_workflow_type(settings)

        artist_tags = _get_artist_tags(tags, settings)
        quality_tags = _get_quality_tags(tags, settings)

        scene_supplement = supplement if workflow_type == "anima" else ""

        positive = _join_parts(
            ", ".join(artist_tags),
            ", ".join(quality_tags),
            setup,
            char,
            scene_supplement,
        )
        return positive

    @staticmethod
    def build_negative_prompt(tags: dict, settings: dict) -> str:
        negative_presets = tags.get("negative_presets", {}) or {}
        workflow_type = _get_workflow_type(settings)
        preset_name = settings.get(f"{workflow_type}_negative_preset", "")
        if preset_name and preset_name in negative_presets:
            negative_tags = _as_tags(negative_presets.get(preset_name, []))
        else:
            negative_key = "anima_negative" if workflow_type == "anima" else "negative"
            negative_tags = _as_tags(tags.get(negative_key, []))
        return ", ".join(negative_tags)

    def build(
        self,
        setup: str,
        char: str,
        supplement: str,
        tags: dict,
        settings: dict,
    ) -> dict:
        artist_tags = _get_artist_tags(tags, settings)
        quality_tags = _get_quality_tags(tags, settings)
        return {
            "positive": self.build_positive_prompt(setup, char, supplement, tags, settings),
            "negative": self.build_negative_prompt(tags, settings),
            "width": int(settings.get("img_w", 756) or 756),
            "height": int(settings.get("img_h", 756) or 756),
            "quality_tag_start": len(artist_tags),
            "quality_tag_count": len(quality_tags),
        }


def build_v1_prompt(setup: str, char: str, supplement: str, tags: dict, settings: dict) -> dict:
    """V1(ILXL/UPSCALE 스타일) 포맷 조립.

    ANIMA 품질/부정 프리셋만 소비하고 LoRA·아티스트·SDXL 분기는 없는 단순 조립.
    구조:
      [Positive]
      {ANIMA 품질 태그}, {setup}, {char}, {supplement},
      [ILXL]
      {setup}, {char},
      [Negative]
      {ANIMA 부정 태그}
    """
    v1_settings = dict(settings or {})
    v1_settings["chansub_workflow_type"] = "anima"  # ANIMA 프리셋 강제
    quality_tags = _get_quality_tags(tags, v1_settings)
    negative = ChansubPromptBuilder.build_negative_prompt(tags, v1_settings)

    positive_section = _join_parts(", ".join(quality_tags), setup, char, supplement)
    ilxl_section = _join_parts(setup, char)
    positive = (
        f"[Positive]\n{positive_section}\n\n"
        f"[ILXL]\n{ilxl_section}"
    )
    return {
        "positive": positive,
        "negative": negative,
        "width": int(settings.get("img_w", 756) or 756),
        "height": int(settings.get("img_h", 756) or 756),
        "quality_tag_start": 0,
        "quality_tag_count": 0,
    }
