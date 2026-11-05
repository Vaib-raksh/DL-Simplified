"""Facial-landmark feature extraction for the rule and learning baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .landmarks import EYE_LEFT_INDICES, EYE_RIGHT_INDICES, MOUTH_INDICES


@dataclass(frozen=True)
class FacialFeatures:
    """Frame-level measurements and their tracking validity."""

    left_ear: float | None
    right_ear: float | None
    average_ear: float | None
    mar: float | None
    valid: bool
    reason: str = ""


def _pixel_point(
    landmark: Any, width: int, height: int
) -> tuple[float, float] | None:
    if width <= 0 or height <= 0:
        return None
    try:
        x = float(landmark.x) * width
        y = float(landmark.y) * height
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _ratio(
    points: Sequence[tuple[float, float] | None],
    vertical_pairs: tuple[tuple[int, int], ...],
    horizontal_pair: tuple[int, int],
) -> float | None:
    if any(point is None for point in points):
        return None
    valid_points = [point for point in points if point is not None]
    denominator = _distance(
        valid_points[horizontal_pair[0]], valid_points[horizontal_pair[1]]
    )
    if denominator <= 0:
        return None
    numerator = sum(
        _distance(valid_points[first], valid_points[second])
        for first, second in vertical_pairs
    )
    return numerator / (len(vertical_pairs) * denominator)


def extract_features(
    face_landmarks: Sequence[Any] | None, width: int, height: int
) -> FacialFeatures:
    """Calculate EAR and MAR from normalized landmarks in pixel space.

    EAR uses six points per eye:
    ``(outer-corner, upper-inner, upper-outer, inner-corner, lower-outer,
    lower-inner)``. MAR uses the outer and inner vertical mouth pairs:
    ``(top, bottom)`` at landmarks ``0/17`` and ``13/14``, divided by the
    outer mouth width ``61/291``.
    """
    if not face_landmarks:
        return FacialFeatures(None, None, None, None, False, "no face landmarks")

    required = max((*EYE_LEFT_INDICES, *EYE_RIGHT_INDICES, *MOUTH_INDICES))
    if len(face_landmarks) <= required:
        return FacialFeatures(None, None, None, None, False, "incomplete landmarks")

    points = {
        index: _pixel_point(face_landmarks[index], width, height)
        for index in set(EYE_LEFT_INDICES + EYE_RIGHT_INDICES + MOUTH_INDICES)
    }
    left = [points[index] for index in EYE_LEFT_INDICES]
    right = [points[index] for index in EYE_RIGHT_INDICES]
    mouth = [points[index] for index in MOUTH_INDICES]
    left_ear = _ratio(left, ((1, 5), (2, 4)), (0, 3))
    right_ear = _ratio(right, ((1, 5), (2, 4)), (0, 3))
    mar = _ratio(mouth, ((2, 3), (4, 5)), (0, 1))
    if left_ear is None or right_ear is None or mar is None:
        return FacialFeatures(
            left_ear, right_ear, None, mar, False, "invalid landmark geometry"
        )
    return FacialFeatures(
        left_ear, right_ear, (left_ear + right_ear) / 2, mar, True
    )
