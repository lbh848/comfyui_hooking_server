"""유기형 말풍선(MVP) 설정·데이터 클래스.

Phase 1(일반 대사 풍선)에서 외곽선/꼬리/후보 생성에 필요한 값만 담는다.
참고 구현(만화풍 말풍선 개선 작업서)을 현 프로젝트 스타일에 맞춰 옮겼고,
TailConfig 에 tail_width_scale(꼬리 두께 배율)을 추가했다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np


Point: TypeAlias = tuple[int, int]
Box: TypeAlias = tuple[int, int, int, int]


@dataclass(slots=True)
class OrganicShapeConfig:
    """유기형 타원 외곽선 생성 설정. 저주파 사인파 합성으로 큰 굴곡만 만든다."""

    point_count: int = 180
    wobble: float = 0.055            # 0.035~0.075 권장. 상한 0.30(과장 표현용).
    min_radial_scale: float = 0.78   # 텍스트 안전 영역 보장을 위한 하한. 강한 굴곡 허용.
    max_radial_scale: float = 1.22
    asymmetry: float = 0.025         # 축별 약한 비대칭 성분.


@dataclass(slots=True)
class CandidateSearchConfig:
    """위치당 후보 생성 설정(Phase 2에서 사용)."""

    candidate_count: int = 12
    center_jitter_ratio: float = 0.055
    radius_jitter_ratio: float = 0.045
    wobble_jitter: float = 0.012
    tail_angle_jitter_deg: float = 8.0


@dataclass(slots=True)
class CandidateScoreWeights:
    """후보 평가 가중치(Phase 2에서 사용)."""

    text_overflow: float = 5000.0
    image_outside: float = 2500.0
    face_overlap: float = 900.0
    body_overlap: float = 240.0
    other_bubble_overlap: float = 650.0
    excessive_empty_space: float = 30.0
    extreme_aspect_ratio: float = 80.0
    tail_length: float = 0.12


@dataclass(slots=True)
class BubbleCandidate:
    """단일 말풍선 후보. Phase 1에서는 contour/tail_tip 정도만 쓴다."""

    contour: np.ndarray
    center: Point
    radius: Point
    tail_tip: Point
    seed: int
    score: float = float("inf")
    debug: dict[str, float] = field(default_factory=dict)
