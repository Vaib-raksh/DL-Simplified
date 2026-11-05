"""Subject-safe UTA-RLDD preprocessing and temporal-window utilities."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .features import FacialFeatures, extract_features
from .landmarks import DetectorConfig, FaceLandmarkDetector, timestamp_for_frame

FEATURE_ORDER = ("left_ear", "right_ear", "average_ear", "mar")
REQUIRED_MANIFEST_COLUMNS = ("subject_id", "video_id", "label", "video_path")
OFFICIAL_LABELS = {
    "alert": "alert",
    "0": "alert",
    "low vigilant": "low vigilant",
    "5": "low vigilant",
    "drowsy": "drowsy",
    "10": "drowsy",
}


@dataclass(frozen=True)
class VideoRecord:
    subject_id: str
    video_id: str
    label: str
    video_path: Path


@dataclass(frozen=True)
class FrameRecord:
    subject_id: str
    video_id: str
    label: str
    timestamp_ms: int
    features: FacialFeatures


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class WindowConfig:
    target_fps: float = 10.0
    window_seconds: float = 3.0
    stride_seconds: float = 1.0
    max_gap_seconds: float = 0.5


def load_manifest(dataset_root: Path, manifest_path: Path) -> list[VideoRecord]:
    """Load an explicit manifest; labels and subject identity are never guessed."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest was not found: {manifest_path}")
    records: list[VideoRecord] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            values = {column: (row.get(column) or "").strip() for column in REQUIRED_MANIFEST_COLUMNS}
            if not all(values.values()):
                raise ValueError(f"Manifest row {row_number} has an empty required value")
            label = OFFICIAL_LABELS.get(values["label"].lower())
            if label is None:
                raise ValueError(
                    f"Manifest row {row_number} has unsupported label "
                    f"{values['label']!r}; expected alert, low vigilant, or drowsy"
                )
            path = Path(values["video_path"])
            if not path.is_absolute():
                path = dataset_root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Video from manifest row {row_number} was not found: {path}")
            records.append(VideoRecord(values["subject_id"], values["video_id"], label, path))
    if not records:
        raise ValueError("Manifest contains no video records")
    return records


def assign_subject_splits(
    records: Sequence[VideoRecord], config: SplitConfig
) -> dict[str, str]:
    """Assign each subject to exactly one deterministic train/validation/test split."""
    if not 0 <= config.validation_fraction < 1 or not 0 <= config.test_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if config.validation_fraction + config.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    subjects = sorted({record.subject_id for record in records})
    randomizer = random.Random(config.seed)
    randomizer.shuffle(subjects)
    test_count = round(len(subjects) * config.test_fraction)
    validation_count = round(len(subjects) * config.validation_fraction)
    split_by_subject: dict[str, str] = {}
    for subject in subjects[:test_count]:
        split_by_subject[subject] = "test"
    for subject in subjects[test_count : test_count + validation_count]:
        split_by_subject[subject] = "validation"
    for subject in subjects[test_count + validation_count :]:
        split_by_subject[subject] = "train"
    return split_by_subject


def _feature_vector(features: FacialFeatures) -> np.ndarray:
    return np.array(
        [
            features.left_ear if features.left_ear is not None else np.nan,
            features.right_ear if features.right_ear is not None else np.nan,
            features.average_ear if features.average_ear is not None else np.nan,
            features.mar if features.mar is not None else np.nan,
        ],
        dtype=np.float32,
    )


def _valid_runs(
    records: Sequence[FrameRecord], max_gap_ms: int
) -> Iterable[list[FrameRecord]]:
    run: list[FrameRecord] = []
    previous_timestamp: int | None = None
    for record in records:
        if not record.features.valid:
            if run:
                yield run
            run, previous_timestamp = [], None
            continue
        if previous_timestamp is not None and record.timestamp_ms - previous_timestamp > max_gap_ms:
            if run:
                yield run
            run = []
        run.append(record)
        previous_timestamp = record.timestamp_ms
    if run:
        yield run


def construct_windows(
    records: Sequence[FrameRecord], config: WindowConfig
) -> tuple[np.ndarray, list[dict[str, str | int]]]:
    """Resample valid contiguous runs to fixed-rate, fixed-duration windows."""
    if config.target_fps <= 0 or config.window_seconds <= 0 or config.stride_seconds <= 0:
        raise ValueError("Sampling rate, window duration, and stride must be positive")
    window_size = round(config.target_fps * config.window_seconds)
    stride = round(config.target_fps * config.stride_seconds)
    if window_size < 2 or stride < 1:
        raise ValueError("Sampling settings produce an unusable window")
    all_windows: list[np.ndarray] = []
    metadata: list[dict[str, str | int]] = []
    grouped: dict[tuple[str, str, str], list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault((record.subject_id, record.video_id, record.label), []).append(record)
    for (subject_id, video_id, label), video_records in grouped.items():
        video_records = sorted(video_records, key=lambda item: item.timestamp_ms)
        for run in _valid_runs(video_records, round(config.max_gap_seconds * 1000)):
            if len(run) < 2:
                continue
            source_times = np.array([record.timestamp_ms for record in run], dtype=np.float64)
            source_values = np.stack([_feature_vector(record.features) for record in run])
            start = source_times[0]
            end = source_times[-1]
            duration = (window_size - 1) * 1000 / config.target_fps
            for start_ms in np.arange(start, end - duration + 1, config.stride_seconds * 1000):
                target_times = start_ms + np.arange(window_size) * 1000 / config.target_fps
                if target_times[-1] > end:
                    break
                window = np.column_stack(
                    [np.interp(target_times, source_times, source_values[:, index]) for index in range(4)]
                ).astype(np.float32)
                all_windows.append(window)
                metadata.append({
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "label": label,
                    "start_timestamp_ms": int(round(start_ms)),
                })
    if not all_windows:
        return np.empty((0, window_size, len(FEATURE_ORDER)), dtype=np.float32), metadata
    return np.stack(all_windows), metadata


def fit_normalization(
    windows: np.ndarray, metadata: Sequence[dict[str, str | int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature normalization using train windows only."""
    train_values = [window for window, item in zip(windows, metadata) if item["split"] == "train"]
    if not train_values:
        raise ValueError("Cannot fit normalization: no training windows were created")
    values = np.concatenate(train_values, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def preprocess_records(
    records: Sequence[FrameRecord],
    split_by_subject: dict[str, str],
    window_config: WindowConfig,
) -> tuple[np.ndarray, list[dict[str, str | int]], np.ndarray, np.ndarray]:
    windows, metadata = construct_windows(records, window_config)
    for item in metadata:
        item["split"] = split_by_subject[item["subject_id"]]
    mean, std = fit_normalization(windows, metadata)
    return (windows - mean) / std, metadata, mean, std


def extract_video_records(
    records: Sequence[VideoRecord], model_path: Path
) -> list[FrameRecord]:
    """Extract frame features using one reusable MediaPipe detector."""
    frame_records: list[FrameRecord] = []
    with FaceLandmarkDetector(DetectorConfig(model_path=model_path)) as detector:
        for record in records:
            capture = cv2.VideoCapture(str(record.video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {record.video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_index = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    timestamp = timestamp_for_frame(frame_index, fps if fps > 0 else None)
                    result = detector.detect(frame, timestamp)
                    face = result.face_landmarks[0] if result.face_landmarks else None
                    features = extract_features(face, frame.shape[1], frame.shape[0])
                    frame_records.append(
                        FrameRecord(record.subject_id, record.video_id, record.label, timestamp, features)
                    )
                    frame_index += 1
            finally:
                capture.release()
    return frame_records


def save_preprocessed(
    output_path: Path,
    windows: np.ndarray,
    metadata: Sequence[dict[str, str | int]],
    mean: np.ndarray,
    std: np.ndarray,
    config: dict[str, object],
    frame_records: Sequence[FrameRecord] = (),
) -> None:
    """Persist arrays and JSON metadata without serializing source video."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({str(item["label"]) for item in metadata})
    label_mapping = {label: index for index, label in enumerate(labels)}
    split = np.array([str(item["split"]) for item in metadata])
    encoded_labels = np.array([label_mapping[str(item["label"])] for item in metadata], dtype=np.int64)
    frame_features = (
        np.stack([_feature_vector(record.features) for record in frame_records], axis=0)
        if frame_records
        else np.empty((0, len(FEATURE_ORDER)), dtype=np.float32)
    )
    np.savez_compressed(
        output_path,
        windows=windows,
        labels=encoded_labels,
        split=split,
        subject_ids=np.array([str(item["subject_id"]) for item in metadata]),
        video_ids=np.array([str(item["video_id"]) for item in metadata]),
        start_timestamp_ms=np.array([int(item["start_timestamp_ms"]) for item in metadata]),
        frame_features=frame_features,
        frame_valid=np.array(
            [record.features.valid for record in frame_records], dtype=bool
        ),
        frame_labels=np.array([str(record.label) for record in frame_records]),
        frame_subject_ids=np.array(
            [str(record.subject_id) for record in frame_records]
        ),
        frame_video_ids=np.array([str(record.video_id) for record in frame_records]),
        frame_timestamp_ms=np.array(
            [record.timestamp_ms for record in frame_records], dtype=np.int64
        ),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "feature_order": FEATURE_ORDER,
                "label_mapping": label_mapping,
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(),
                "window_count": len(metadata),
                "frame_count": len(frame_records),
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def smoke_records() -> list[FrameRecord]:
    """Return a tiny deterministic fixture for pipeline tests, never real data."""
    records: list[FrameRecord] = []
    for subject in ("synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d", "synthetic-e"):
        for index in range(40):
            value = 0.2 + index / 1000
            features = FacialFeatures(value, value, value, 0.3, True)
            records.append(FrameRecord(subject, f"{subject}-video", "alert", index * 100, features))
    return records


def config_as_dict(split: SplitConfig, window: WindowConfig) -> dict[str, object]:
    return {"split": asdict(split), "window": asdict(window), "synthetic_fixture": False}
