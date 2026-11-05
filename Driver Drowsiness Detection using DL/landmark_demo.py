"""Run the local webcam or recorded-video facial-landmark demo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.landmarks import (
    DetectorConfig,
    FaceLandmarkDetector,
    default_model_path,
    draw_relevant_landmarks,
    timestamp_for_frame,
)
from src.features import extract_features
from src.rules import RuleConfig, TemporalRuleBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Recorded video path. Omit this option to use the default webcam.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--ear-threshold", type=float, default=RuleConfig.ear_threshold)
    parser.add_argument("--mar-threshold", type=float, default=RuleConfig.mar_threshold)
    parser.add_argument(
        "--closure-seconds", type=float, default=RuleConfig.closure_duration_seconds
    )
    parser.add_argument(
        "--mouth-seconds", type=float, default=RuleConfig.mouth_duration_seconds
    )
    parser.add_argument(
        "--alert-cooldown", type=float, default=RuleConfig.alert_cooldown_seconds
    )
    return parser.parse_args()


def run(
    input_path: Path | None,
    camera_index: int,
    model_path: Path,
    rule_config: RuleConfig | None = None,
) -> None:
    source: int | str = camera_index if input_path is None else str(input_path)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) if input_path else None
    frame_index = 0
    started = time.perf_counter()
    processed_frames = 0

    try:
        with FaceLandmarkDetector(DetectorConfig(model_path=model_path)) as detector:
            rules = TemporalRuleBaseline(rule_config)
            while True:
                success, frame = capture.read()
                if not success:
                    if input_path:
                        print("Reached the end of the video.")
                    else:
                        print("Could not read a frame from the webcam.")
                    break

                elapsed = time.perf_counter() - started
                timestamp_ms = timestamp_for_frame(
                    frame_index, source_fps, elapsed if not input_path else None
                )
                result = detector.detect(frame, timestamp_ms)
                has_face = draw_relevant_landmarks(frame, result)
                features = extract_features(
                    result.face_landmarks[0] if has_face else None,
                    frame.shape[1],
                    frame.shape[0],
                )
                decision = rules.update(features, timestamp_ms)
                processed_frames += 1
                frame_index += 1

                elapsed = time.perf_counter() - started
                processing_fps = processed_frames / elapsed if elapsed > 0 else 0.0
                status = "Face tracked" if features.valid else "No reliable face"
                cv2.putText(
                    frame,
                    f"{status} | {decision.status} | FPS: {processing_fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if has_face else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"EAR: {features.average_ear:.3f}" if features.valid else "EAR: n/a",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"MAR: {features.mar:.3f}" if features.valid else "MAR: n/a",
                    (15, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if decision.alert:
                    cv2.putText(
                        frame,
                        decision.reason,
                        (15, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Face Landmarks - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run(
        args.input,
        args.camera_index,
        args.model,
        RuleConfig(
            ear_threshold=args.ear_threshold,
            mar_threshold=args.mar_threshold,
            closure_duration_seconds=args.closure_seconds,
            mouth_duration_seconds=args.mouth_seconds,
            alert_cooldown_seconds=args.alert_cooldown,
        ),
    )


if __name__ == "__main__":
    main()
