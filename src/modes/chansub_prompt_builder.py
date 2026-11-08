"""챈섭용 삽화 프롬프트 빌더.

로컬 ComfyUI 워크플로우 제어 블럭이나 LoRA 트리거는 만들지 않고,
삽화 모드에서 선택한 ANIMA/SDXL 프리셋과 장면 섹션만 Comfy 문법의 평탄한
POSITIVE/NEGATIVE로 만든다. HTTP 요청 외형만 NAI API 형식이다.
"""

from __future__ import annotations

from modes.word_rules import apply_flat_insert_rules
from preset_catalog import resolve_preset_tags


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
    workflow_type = _get_workflow_type(settings)
    quality_name = settings.get(f"{workflow_type}_quality_preset", "")
    return _as_tags(resolve_preset_tags(tags, "quality_presets", quality_name))


def _get_artist_tags(tags: dict, settings: dict) -> list[str]:
    """현재 챈섭 워크플로우와 프리셋에 해당하는 아티스트 태그를 반환한다."""
    workflow_type = _get_workflow_type(settings)
    artist_name = settings.get(f"{workflow_type}_artist_preset", "")
    return _as_tags(resolve_preset_tags(tags, "artist_presets", artist_name))


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
        background_tags = resolve_preset_tags(tags, "background_presets", settings.get(f"{workflow_type}_background_preset", ""))

        positive = _join_parts(
            ", ".join(artist_tags),
            ", ".join(quality_tags),
            ", ".join(background_tags),
            setup,
            char,
            scene_supplement,
        )
        return positive

    @staticmethod
    def build_negative_prompt(
        tags: dict,
        settings: dict,
    ) -> str:
        workflow_type = _get_workflow_type(settings)
        preset_name = settings.get(f"{workflow_type}_negative_preset", "")
        negative_tags = _as_tags(resolve_preset_tags(tags, "negative_presets", preset_name))
        return ", ".join(negative_tags)

    def build(
        self,
        setup: str,
        char: str,
        supplement: str,
        tags: dict,
        settings: dict,
        insert_rules: list[dict] | None = None,
    ) -> dict:
        artist_tags = _get_artist_tags(tags, settings)
        quality_tags = _get_quality_tags(tags, settings)
        positive = self.build_positive_prompt(setup, char, supplement, tags, settings)
        positive, applied_insert_rules, inserted_tag_count = apply_flat_insert_rules(
            positive,
            insert_rules or [],
            quality_tag_start=len(artist_tags),
            quality_tag_count=len(quality_tags),
        )
        return {
            "positive": positive,
            "negative": self.build_negative_prompt(
                tags,
                settings,
            ),
            "width": int(settings.get("img_w", 756) or 756),
            "height": int(settings.get("img_h", 756) or 756),
            "quality_tag_start": len(artist_tags),
            "quality_tag_count": len(quality_tags) + inserted_tag_count,
            "applied_insert_rules": applied_insert_rules,
        }
