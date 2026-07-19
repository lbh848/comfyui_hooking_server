"""유기형 말풍선 외곽선 생성.

2~5차 사인파를 조합해 큰 저주파 굴곡만 넣은 닫힌 타원 contour를 만든다.
점마다 무작위 노이즈를 넣지 않는다(만화풍 말풍선 작업서 금지 사항).
동일 seed에서 동일 형태가 재현되어야 한다 → numpy.random.default_rng(seed).
"""

from __future__ import annotations

import math

import numpy as np

from .bubble_types import OrganicShapeConfig, Point


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
