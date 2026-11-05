import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.landmarks import (
    TimestampGenerator,
    default_model_path,
    timestamp_for_frame,
)
from src.features import extract_features
from src.rules import RuleConfig, TemporalRuleBaseline
from src.dataset import (
    FrameRecord,
    SplitConfig,
    VideoRecord,
    WindowConfig,
    assign_subject_splits,
    construct_windows,
    fit_normalization,
    load_manifest,
    load_manifest,
)
from src.features import FacialFeatures


class TimestampTests(unittest.TestCase):
    def test_timestamps_are_strictly_increasing(self) -> None:
        generator = TimestampGenerator()
        timestamps = [generator.next(value) for value in (0, 0, 30, 20)]
        self.assertEqual(timestamps, [0, 1, 30, 31])

    def test_frame_timestamp_uses_source_fps(self) -> None:
        self.assertEqual(timestamp_for_frame(30, 30.0), 1000)

    def test_frame_timestamp_uses_elapsed_time_without_fps(self) -> None:
        self.assertEqual(timestamp_for_frame(0, 0.0, 1.25), 1250)


class PathTests(unittest.TestCase):
    def test_default_model_is_relative_to_project_source(self) -> None:
        model_path = default_model_path()
        self.assertEqual(model_path.name, "face_landmarker.task")
        self.assertTrue(model_path.is_absolute())
        self.assertEqual(model_path.parent.name, "models")


class FeatureTests(unittest.TestCase):
    def test_feature_ratios_use_pixel_aspect_ratio(self) -> None:
        points = [type("Point", (), {"x": 0.0, "y": 0.0})() for _ in range(400)]
        for index, x, y in (
            (33, 0.1, 0.4), (160, 0.2, 0.3), (158, 0.2, 0.3),
            (133, 0.3, 0.4), (153, 0.2, 0.5), (144, 0.2, 0.5),
            (362, 0.7, 0.4), (385, 0.8, 0.3), (387, 0.8, 0.3),
            (263, 0.9, 0.4), (373, 0.8, 0.5), (380, 0.8, 0.5),
            (61, 0.4, 0.4), (291, 0.6, 0.4), (0, 0.5, 0.35),
            (17, 0.5, 0.55), (13, 0.5, 0.35), (14, 0.5, 0.55),
        ):
            points[index].x, points[index].y = x, y
        features = extract_features(points, 200, 100)
        self.assertTrue(features.valid)
        self.assertAlmostEqual(features.average_ear, 0.5, places=5)
        self.assertAlmostEqual(features.mar, 0.5, places=5)

    def test_invalid_landmarks_do_not_become_zero(self) -> None:
        features = extract_features(None, 200, 100)
        self.assertFalse(features.valid)
        self.assertIsNone(features.average_ear)

    def test_zero_width_geometry_is_invalid(self) -> None:
        points = [type("Point", (), {"x": 0.0, "y": 0.0})() for _ in range(400)]
        for index in (
            33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
            61, 291, 0, 17, 13, 14,
        ):
            points[index].x = 0.5
            points[index].y = 0.5
        features = extract_features(points, 200, 100)
        self.assertFalse(features.valid)
        self.assertEqual(features.reason, "invalid landmark geometry")


class RuleTests(unittest.TestCase):
    def _features(self, ear: float, mar: float):
        return type(
            "Features",
            (),
            {
                "valid": True,
                "average_ear": ear,
                "mar": mar,
                "reason": "",
            },
        )()

    def test_closure_uses_elapsed_time_and_alert_cooldown(self) -> None:
        rules = TemporalRuleBaseline(
            RuleConfig(closure_duration_seconds=1.0, alert_cooldown_seconds=5.0)
        )
        closed = self._features(0.1, 0.2)
        self.assertEqual(rules.update(closed, 0).status, "POSSIBLE BLINK")
        self.assertTrue(rules.update(closed, 1000).alert)
        self.assertFalse(rules.update(closed, 2000).alert)

    def test_tracking_gap_resets_temporal_buffer(self) -> None:
        rules = TemporalRuleBaseline(RuleConfig(closure_duration_seconds=1.0))
        closed = self._features(0.1, 0.2)
        self.assertEqual(rules.update(closed, 0).status, "POSSIBLE BLINK")
        missing = self._features(None, None)
        missing.valid = False
        self.assertEqual(rules.update(missing, 2000).status, "UNABLE TO ASSESS")
        self.assertEqual(rules.update(closed, 2001).status, "POSSIBLE BLINK")


class DatasetTests(unittest.TestCase):
    def test_subject_split_has_no_overlap(self) -> None:
        records = [
            VideoRecord(f"subject-{index}", f"video-{index}", "alert", __file__)
            for index in range(10)
        ]
        splits = assign_subject_splits(records, SplitConfig(0.2, 0.2, 7))
        groups = {
            split: {subject for subject, value in splits.items() if value == split}
            for split in ("train", "validation", "test")
        }
        self.assertEqual(set().union(*groups.values()), set(splits))
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_windows_do_not_cross_invalid_tracking_gaps(self) -> None:
        valid = FacialFeatures(0.2, 0.2, 0.2, 0.3, True)
        invalid = FacialFeatures(None, None, None, None, False, "gap")
        records = [
            FrameRecord("s", "v", "alert", 0, valid),
            FrameRecord("s", "v", "alert", 100, valid),
            FrameRecord("s", "v", "alert", 200, invalid),
            FrameRecord("s", "v", "alert", 1000, valid),
            FrameRecord("s", "v", "alert", 1100, valid),
        ]
        windows, metadata = construct_windows(
            records, WindowConfig(target_fps=10, window_seconds=0.3, stride_seconds=0.1)
        )
        self.assertEqual(len(metadata), 0)
        self.assertEqual(windows.shape[0], 0)

    def test_normalization_uses_training_windows_only(self) -> None:
        windows = __import__("numpy").array(
            [[[1.0], [1.0]], [[3.0], [3.0]]], dtype="float32"
        )
        metadata = [
            {"split": "train", "subject_id": "a", "label": "alert"},
            {"split": "test", "subject_id": "b", "label": "alert"},
        ]
        mean, std = fit_normalization(windows, metadata)
        self.assertEqual(float(mean[0]), 1.0)
        self.assertEqual(float(std[0]), 1.0)

    def test_manifest_canonicalizes_only_official_labels(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "video.mp4").write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "subject_id,video_id,label,video_path\n"
                "s1,v1,5,video.mp4\n",
                encoding="utf-8",
            )
            records = load_manifest(root, manifest)
            self.assertEqual(records[0].label, "low vigilant")
            manifest.write_text(
                "subject_id,video_id,label,video_path\n"
                "s1,v1,yawning,video.mp4\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest(root, manifest)


if __name__ == "__main__":
    unittest.main()
