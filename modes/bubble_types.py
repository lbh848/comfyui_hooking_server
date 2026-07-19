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
    wobble: float = 0.055            # 0.035~0.075 권장. 0.1+는 과장 표현이 아니면 금지.
    min_radial_scale: float = 0.90   # 텍스트 안전 영역 보장을 위한 하한.
    max_radial_scale: float = 1.10
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


@dataclass(slots=True)
class TextBoxShapeConfig:
    """text-safe organic rounded box 생성 설정.

    텍스트 박스 → content_safe_box → 둥근 skeleton → 외곽 방향 organic offset 파이프라인의
    기하/굴곡 파라미터. base_outset(외곽 기본 확장)은 고정이고 wobble(진폭 변동)만
    line/aspect 보정으로 축소된다.
    """

    perimeter_points: int = 220

    padding_x_ratio: float = 0.45   # 폰트 대비 좌우 패딩 비율(레퍼런스값)
    padding_y_ratio: float = 0.34
    min_padding_x: int = 14
    min_padding_y: int = 10

    corner_radius_ratio: float = 0.26
    min_corner_radius: int = 16
    max_corner_radius: int = 52

    base_outset_ratio: float = 0.10   # 외곽 기본 확장(고정 마진)
    min_base_outset: float = 4.0
    max_base_outset: float = 16.0

    wobble_ratio: float = 0.05   # 진폭 변동 비율(자동 보정 대상)
    min_outset: float = 1.0
    max_outset_ratio: float = 0.18
    corner_outset_gain: float = 0.12

    border_margin_extra: int = 1   # border 라인 두께 절반 외에 추가로 확보하는 여유


@dataclass(slots=True)
class BubbleGeometry:
    """placement rect(envelope)를 안쪽으로 역산한 text-safe 기하.

    rect 는 최종 풍선 전체의 hard envelope다. 각 박스는 rect 안쪽으로 겹겹이 inset 된다:
      outer_box ⊇ skeleton_box ⊇ content_safe_box ⊇ text_box
    organic offset(0..max_outset)이 skeleton 둘레를 outer_box 쪽으로 밀어내므로
    contour 는 항상 outer_box(=rect inset border_margin) 내부에 머무른다.
    """

    rect: Box
    text_box: Box
    content_safe_box: Box
    outer_box: Box
    skeleton_box: Box
    corner_radius: int
    corner_guard: int
    base_outset: float
    max_outset: float
    border_margin: int
