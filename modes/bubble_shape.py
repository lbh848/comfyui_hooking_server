"""유기형 말풍선 외곽선 생성.

2~5차 사인파를 조합해 큰 저주파 굴곡만 넣은 닫힌 타원 contour를 만든다.
점마다 무작위 노이즈를 넣지 않는다(만화풍 말풍선 작업서 금지 사항).
동일 seed에서 동일 형태가 재현되어야 한다 → numpy.random.default_rng(seed).
"""

from __future__ import annotations

import math

import numpy as np

from .bubble_types import (
    Box,
    BubbleGeometry,
    OrganicShapeConfig,
    Point,
    TextBoxShapeConfig,
)


def _smooth_circular(values: np.ndarray, radius: int = 5) -> np.ndarray:
    """원형 배열을 이동평균으로 부드럽게 한다(0도/360도가 이어지는 각도 데이터)."""
    if radius <= 0:
        return values.astype(np.float32, copy=True)

    radius = min(radius, max(1, len(values) // 4))
    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size

    padded = np.concatenate(
        [values[-radius:], values, values[:radius]]
    ).astype(np.float32)

    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def make_organic_ellipse(
    center: Point,
    radius: Point,
    seed: int = 0,
    config: OrganicShapeConfig | None = None,
    free_space_scale: np.ndarray | None = None,
) -> np.ndarray:
    """저주파 굴곡이 들어간 닫힌 타원 contour 생성.

    Args:
        center: 말풍선 중심 (cx, cy).
        radius: 기본 반지름 (rx, ry).
        seed: 형태 재현 seed.
        config: 유기형 외곽선 설정.
        free_space_scale: 각도별 공간 여유 배율(point_count 길이). Phase 3에서 사용.

    Returns:
        shape=(N, 2), dtype=int32 contour.
    """
    config = config or OrganicShapeConfig()

    cx, cy = center
    rx, ry = radius

    if rx <= 0 or ry <= 0:
        raise ValueError(f"radius must be positive: {(rx, ry)}")

    if config.point_count < 32:
        raise ValueError("point_count must be at least 32")

    rng = np.random.default_rng(seed)

    theta = np.linspace(
        0.0,
        2.0 * math.pi,
        config.point_count,
        endpoint=False,
        dtype=np.float32,
    )

    # 점별 무작위 떨림이 아닌 2~5차 큰 굴곡만 사용.
    harmonics = np.zeros_like(theta, dtype=np.float32)

    for frequency, base_weight in (
        (2, 0.52),
        (3, 0.28),
        (4, 0.15),
        (5, 0.07),
    ):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = rng.uniform(0.72, 1.0)
        harmonics += (
            base_weight
            * amplitude
            * np.sin(frequency * theta + phase)
        ).astype(np.float32)

    # x/y 축이 같이 움직이면 단순 방사형 찌그러짐처럼 보이므로 축별 약한 비대칭.
    x_phase = rng.uniform(0.0, 2.0 * math.pi)
    y_phase = rng.uniform(0.0, 2.0 * math.pi)

    x_asymmetry = config.asymmetry * np.sin(theta + x_phase)
    y_asymmetry = config.asymmetry * np.sin(theta + y_phase)

    radial_scale = 1.0 + config.wobble * harmonics
    radial_scale = np.clip(
        radial_scale,
        config.min_radial_scale,
        config.max_radial_scale,
    )

    if free_space_scale is not None:
        free_space_scale = np.asarray(
            free_space_scale,
            dtype=np.float32,
        )

        if free_space_scale.shape != radial_scale.shape:
            raise ValueError(
                "free_space_scale length must match point_count: "
                f"{free_space_scale.shape} != {radial_scale.shape}"
            )

        free_space_scale = _smooth_circular(
            free_space_scale,
            radius=max(3, config.point_count // 48),
        )
        # 공간 적응 배율이 과도한 움푹함을 만들지 않도록 제한.
        free_space_scale = np.clip(free_space_scale, 0.92, 1.10)

        radial_scale *= free_space_scale

    x = cx + rx * radial_scale * (1.0 + x_asymmetry) * np.cos(theta)
    y = cy + ry * radial_scale * (1.0 + y_asymmetry) * np.sin(theta)

    contour = np.stack([x, y], axis=1)
    return np.rint(contour).astype(np.int32)


# ─── text-safe organic rounded box ──────────────────────────────────
# 텍스트 박스를 먼저 확정하고 그 바깥을 둥근 skeleton + 외곽 방향 저주파 offset으로
# 감싼다. placement rect를 최종 풍선 전체의 hard envelope으로 쓰고, 기하를 rect 안쪽으로
# 역산한다(만화풍 말풍선 개선 작업서 v2).

def box_width(box: Box) -> int:
    return max(0, int(round(box[2] - box[0])))


def box_height(box: Box) -> int:
    return max(0, int(round(box[3] - box[1])))


def box_short_side(box: Box) -> int:
    return min(box_width(box), box_height(box))


def expand_box(box: Box, inset: float) -> Box:
    """box를 모든 방향으로 inset 만큼 줄인다(inset<0 이면 확장). 정수 반올림."""
    x1, y1, x2, y2 = box
    i = int(round(inset))
    return (x1 + i, y1 + i, x2 - i, y2 - i)


def inset_box_xy(box: Box, inset_x: float, inset_y: float) -> Box:
    x1, y1, x2, y2 = box
    ix = int(round(inset_x))
    iy = int(round(inset_y))
    return (x1 + ix, y1 + iy, x2 - ix, y2 - iy)


def box_center(box: Box) -> Point:
    return (
        int(round((box[0] + box[2]) * 0.5)),
        int(round((box[1] + box[3]) * 0.5)),
    )


def corner_guard_for_radius(radius: int) -> int:
    """rounded corner가 content_safe_box 모서리를 자르지 않도록 안쪽으로 더 빼는 거리.

    모서리 중심에서 호까지의 직선거리 = radius/√2 이므로, 사각형 모서리가 잘리는 폭은
    radius·(1 - 1/√2) 가 된다. 여기에 수치 여유 2px을 더한다.
    """
    value = radius * (1.0 - 1.0 / math.sqrt(2.0))
    return int(math.ceil(value + 2.0))


def _text_box_params(
    content_short_side: float,
    config: TextBoxShapeConfig,
) -> tuple[int, int, float, float]:
    """content(text+padding) short side 기반으로 geometry 파라미터 산출.

    estimate_envelope_extra(placement 전) 와 build_text_safe_geometry(rect 역산) 가
    동일한 파라미터를 쓰도록 이 함수를 단일 소스로 둔다. raw_short 계수(0.62)는
    content 안에서 둥근 모서리/외곽 확장이 자연스럽도록 보정한 대표 짧은 변 길이다.
    """
    short = max(8.0, float(content_short_side))
    raw_short = short * 0.62
    corner_radius = int(np.clip(
        raw_short * config.corner_radius_ratio,
        config.min_corner_radius,
        config.max_corner_radius,
    ))
    corner_guard = corner_guard_for_radius(corner_radius)
    base_outset = float(np.clip(
        raw_short * config.base_outset_ratio,
        config.min_base_outset,
        config.max_base_outset,
    ))
    max_outset = float(np.clip(
        raw_short * config.max_outset_ratio,
        max(config.min_outset + 1.0, base_outset),
        max(base_outset, raw_short * config.max_outset_ratio),
    ))
    return corner_radius, corner_guard, base_outset, max_outset


def build_text_safe_geometry(
    rect: Box,
    text_w: float,
    text_h: float,
    font_size: float,
    *,
    padding_x: float,
    padding_y: float,
    border_w: float,
    config: TextBoxShapeConfig | None = None,
) -> BubbleGeometry:
    """placement rect(envelope) 안쪽으로 text-safe 기하를 역산한다.

    rect: 최종 풍선 전체의 hard envelope(배치 알고리즘이 content+guard+outset+border
      여유까지 감안해 잡은 박스). contour 는 절대 rect 를 넘지 않는다.
    text_w/text_h: 중앙 정렬 텍스트의 실측 크기.
    padding_x/padding_y: 텍스트→content_safe_box 사이 패딩(font_size·em).
    """
    config = config or TextBoxShapeConfig()

    rect_x1, rect_y1, rect_x2, rect_y2 = rect
    rect = (int(round(rect_x1)), int(round(rect_y1)), int(round(rect_x2)), int(round(rect_y2)))

    # border 라인이 rect 안에 그려지도록 라인 두께 절반 + 여유를 뺀다.
    border_margin = int(math.ceil(max(1.0, float(border_w)) / 2.0)) + int(config.border_margin_extra)
    bm = min(border_margin, max(0, box_short_side(rect) // 4))
    outer_box = expand_box(rect, bm)

    # content(text+padding) short side 기반 파라미터. estimate_envelope_extra 와 동일 소스.
    content_short = min(
        text_w + 2.0 * padding_x,
        text_h + 2.0 * padding_y,
    )
    corner_radius, corner_guard, base_outset, max_outset = _text_box_params(content_short, config)

    # skeleton_box: offset(0..max_outset) 이 외곽으로 밀어 outer_box 경계에 닿도록
    # max_outset 만큼 안쪽으로 둔다 → contour ⊆ outer_box ⊆ rect.
    skeleton_box = expand_box(outer_box, max_outset)
    # content_safe_box: rounded corner 안쪽 안전 영역.
    content_safe_box = expand_box(skeleton_box, corner_guard)
    # text_box: 패딩 안쪽. (중앙 정렬이므로 content_safe_box 를 균등 inset.)
    text_box = inset_box_xy(content_safe_box, padding_x, padding_y)

    return BubbleGeometry(
        rect=rect,
        text_box=text_box,
        content_safe_box=content_safe_box,
        outer_box=outer_box,
        skeleton_box=skeleton_box,
        corner_radius=corner_radius,
        corner_guard=corner_guard,
        base_outset=base_outset,
        max_outset=max_outset,
        border_margin=bm,
    )


def _allocate_counts(lengths: list[float], total_points: int) -> list[int]:
    total = max(sum(lengths), 1e-6)
    counts = [max(6, int(round(total_points * value / total))) for value in lengths]
    difference = total_points - sum(counts)
    index = 0
    while difference != 0 and index < total_points * 20:
        target = index % len(counts)
        if difference > 0:
            counts[target] += 1
            difference -= 1
        elif counts[target] > 6:
            counts[target] -= 1
            difference += 1
        index += 1
    return counts


def sample_rounded_rect_perimeter(
    box: Box,
    corner_radius: int,
    point_count: int = 220,
) -> tuple[np.ndarray, np.ndarray]:
    """rounded rectangle 둘레 점과 각 점의 corner weight(0=변, 1=모서리 호) 반환.

    진행 방향: top→top-right→right→bottom-right→bottom→bottom-left→left→top-left.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid box: {box}")

    radius = int(np.clip(corner_radius, 1, min(width, height) // 2))

    horizontal = max(1.0, width - 2.0 * radius)
    vertical = max(1.0, height - 2.0 * radius)
    arc = 0.5 * math.pi * radius

    counts = _allocate_counts(
        [horizontal, arc, vertical, arc, horizontal, arc, vertical, arc],
        max(point_count, 64),
    )

    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def add_line(start, end, count):
        t = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float32)[:, None]
        s = np.asarray(start, dtype=np.float32)
        e = np.asarray(end, dtype=np.float32)
        points.append(s * (1.0 - t) + e * t)
        weights.append(np.zeros(count, dtype=np.float32))

    def add_arc(center, start_angle, end_angle, count):
        angles = np.linspace(start_angle, end_angle, count, endpoint=False, dtype=np.float32)
        cx, cy = center
        pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
        points.append(pts.astype(np.float32))
        weights.append(np.ones(count, dtype=np.float32))

    add_line((x1 + radius, y1), (x2 - radius, y1), counts[0])
    add_arc((x2 - radius, y1 + radius), -0.5 * math.pi, 0.0, counts[1])
    add_line((x2, y1 + radius), (x2, y2 - radius), counts[2])
    add_arc((x2 - radius, y2 - radius), 0.0, 0.5 * math.pi, counts[3])
    add_line((x2 - radius, y2), (x1 + radius, y2), counts[4])
    add_arc((x1 + radius, y2 - radius), 0.5 * math.pi, math.pi, counts[5])
    add_line((x1, y2 - radius), (x1, y1 + radius), counts[6])
    add_arc((x1 + radius, y1 + radius), math.pi, 1.5 * math.pi, counts[7])

    return np.concatenate(points, axis=0), np.concatenate(weights, axis=0)


def compute_outward_normals(contour: np.ndarray) -> np.ndarray:
    """닫힌 contour의 각 점에서 외곽 법선 단위벡터. CCW/CW 모두 cover 하도록
    진행방향 tangent=(next-prev), outward=(tangent_y, -tangent_x) 로 정의."""
    prev_pts = np.roll(contour, 1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)
    tangent = next_pts - prev_pts
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    outward = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-6)
    return outward.astype(np.float32)


def make_low_frequency_offsets(
    point_count: int,
    seed: int,
    base_outset: float,
    wobble_amplitude: float,
    min_outset: float,
    max_outset: float,
    corner_weight: np.ndarray,
    corner_outset_gain: float,
) -> np.ndarray:
    """외곽 normal 방향으로 더할 저주파 offset. 2~5차 사인파 합성 + 원형 smoothing.

    base_outset(고정 마진) 위에 wobble_amplitude 만큼의 변동을 더한다.
    점별 무작위 노이즈는 금지(작업서). 결과는 [min_outset, max_outset] 로 clip.
    """
    rng = np.random.default_rng(seed)

    phase = np.linspace(0.0, 2.0 * math.pi, point_count, endpoint=False, dtype=np.float32)
    signal = np.zeros(point_count, dtype=np.float32)
    for frequency, weight in ((1, 0.20), (2, 0.48), (3, 0.24), (4, 0.10)):
        random_phase = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = rng.uniform(0.72, 1.0)
        signal += (weight * amplitude * np.sin(frequency * phase + random_phase)).astype(np.float32)

    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 1e-6:
        signal /= max_abs

    offsets = base_outset + base_outset * wobble_amplitude * 3.0 * signal
    offsets *= (1.0 + corner_weight * corner_outset_gain)
    offsets = _smooth_circular(offsets, radius=max(3, point_count // 55))
    return np.clip(offsets, min_outset, max_outset).astype(np.float32)


def make_organic_text_box(
    geometry: BubbleGeometry,
    seed: int,
    config: TextBoxShapeConfig | None = None,
    wobble_amplitude: float | None = None,
) -> np.ndarray:
    """skeleton 둘레 위 각 점에서 외곽 normal 방향으로 저주파 offset을 더한 contour.

    skeleton = skeleton_box 의 rounded rect 둘레. offset 은 [min_outset, max_outset]
    범위로 외곽 방향으로만 더해지므로 contour 는 skeleton_box ⊇ outer_box 내부에
    머무른다(= rect hard envelope 준수).
    """
    config = config or TextBoxShapeConfig()
    if wobble_amplitude is None:
        wobble_amplitude = config.wobble_ratio

    skeleton, corner_weight = sample_rounded_rect_perimeter(
        geometry.skeleton_box,
        geometry.corner_radius,
        config.perimeter_points,
    )
    normals = compute_outward_normals(skeleton)

    offsets = make_low_frequency_offsets(
        point_count=len(skeleton),
        seed=seed,
        base_outset=geometry.base_outset,
        wobble_amplitude=wobble_amplitude,
        min_outset=config.min_outset,
        max_outset=geometry.max_outset,
        corner_weight=corner_weight,
        corner_outset_gain=config.corner_outset_gain,
    )

    contour = skeleton + normals * offsets[:, None]
    return np.rint(contour).astype(np.int32)


def estimate_envelope_extra(
    content_short_side: float,
    border_w: float,
    config: TextBoxShapeConfig | None = None,
) -> tuple[int, int, float]:
    """placement 에 전달할 body_size 산출용: content 외곽에 더해야 할 extra per side.

    returns (corner_guard, border_margin, max_outset). content bubble_size 에
    2*(corner_guard + border_margin + max_outset) 를 더하면 envelope size 가 된다.
    placement 전(아직 rect 없음)에 content 기반으로 근사 산출한다.
    """
    config = config or TextBoxShapeConfig()
    corner_radius, corner_guard, _base_outset, max_outset = _text_box_params(content_short_side, config)
    border_margin = int(math.ceil(max(1.0, float(border_w)) / 2.0)) + int(config.border_margin_extra)
    return corner_guard, border_margin, max_outset
