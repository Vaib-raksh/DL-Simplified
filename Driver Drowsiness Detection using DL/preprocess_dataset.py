"""Extract and save subject-safe temporal feature windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import (
    SplitConfig,
    WindowConfig,
    assign_subject_splits,
    config_as_dict,
    extract_video_records,
    load_manifest,
    preprocess_records,
    save_preprocessed,
    smoke_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("features.npz"))
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny synthetic fixture; never use its output as real performance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_config = SplitConfig(args.validation_fraction, args.test_fraction, args.seed)
    window_config = WindowConfig(
        args.target_fps, args.window_seconds, args.stride_seconds, args.max_gap_seconds
    )
    if args.smoke:
        records = smoke_records()
        split_records = [
            type("VideoRecord", (), {"subject_id": record.subject_id})() for record in records
        ]
        split_by_subject = assign_subject_splits(split_records, split_config)
        config = config_as_dict(split_config, window_config)
        config["synthetic_fixture"] = True
    else:
        if args.manifest is None or args.model is None:
            raise SystemExit("--manifest and --model are required unless --smoke is used")
        records = load_manifest(args.dataset_root, args.manifest)
        split_by_subject = assign_subject_splits(records, split_config)
        records = extract_video_records(records, args.model)
        config = config_as_dict(split_config, window_config)
        config["dataset_root"] = str(args.dataset_root.resolve())
        config["manifest"] = str(args.manifest.resolve())
    windows, metadata, mean, std = preprocess_records(records, split_by_subject, window_config)
    save_preprocessed(args.output, windows, metadata, mean, std, config, records)
    print(f"Saved {len(metadata)} windows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
