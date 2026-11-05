"""MediaPipe Tasks face-landmark detection helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


EYE_LEFT_INDICES = (33, 160, 158, 133, 153, 144)
EYE_RIGHT_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (61, 291, 0, 17, 13, 14, 78, 308)


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the MediaPipe Tasks face landmarker."""

    model_path: Path
    max_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class TimestampGenerator:
    """Generate strictly increasing timestamps for sequential video inference."""

    def __init__(self) -> None:
        self._last_timestamp_ms = -1

    def next(self, timestamp_ms: int) -> int:
        """Return ``timestamp_ms`` while guaranteeing strict monotonicity."""
        candidate = max(0, int(timestamp_ms))
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate


def project_root() -> Path:
    """Return the project directory containing ``src``."""
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    """Return the bundled MediaPipe model path."""
    return project_root() / "models" / "face_landmarker.task"


def timestamp_for_frame(
    frame_index: int,
    source_fps: float | None,
    elapsed_seconds: float | None = None,
) -> int:
    """Create a video timestamp from source timing, with a safe fallback."""
    if source_fps is not None and source_fps > 0:
        return round(frame_index * 1000 / source_fps)
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        return round(elapsed_seconds * 1000)
    return round(time.monotonic() * 1000)


class FaceLandmarkDetector:
    """Convert OpenCV frames and run the MediaPipe Tasks face landmarker."""

    def __init__(self, config: DetectorConfig) -> None:
        model_path = config.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model asset was not found: {model_path}"
            )
        if config.max_faces < 1:
            raise ValueError("max_faces must be at least 1")

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=config.max_faces,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamps = TimestampGenerator()

    def detect(self, frame_bgr: Any, timestamp_ms: int) -> Any:
        """Detect landmarks in one BGR OpenCV frame."""
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            raise ValueError("frame_bgr must be a color image")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(
            image, self._timestamps.next(timestamp_ms)
        )

    def close(self) -> None:
        """Release the MediaPipe task resources."""
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _point(landmark: Any, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized landmark to a pixel coordinate."""
    return (
        min(width - 1, max(0, round(landmark.x * width))),
        min(height - 1, max(0, round(landmark.y * height))),
    )


def draw_relevant_landmarks(frame_bgr: Any, result: Any) -> bool:
    """Draw eye and mouth landmarks; return whether a face was available."""
    if frame_bgr is None or result is None or not result.face_landmarks:
        return False

    height, width = frame_bgr.shape[:2]
    face = result.face_landmarks[0]
    groups = (
        (EYE_LEFT_INDICES, (0, 255, 0)),
        (EYE_RIGHT_INDICES, (0, 255, 0)),
        (MOUTH_INDICES, (255, 180, 0)),
    )
    for indices, color in groups:
        points = [_point(face[index], width, height) for index in indices]
        for start, end in zip(points, points[1:] + points[:1]):
            cv2.line(frame_bgr, start, end, color, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame_bgr, point, 2, color, -1, cv2.LINE_AA)
    return True
