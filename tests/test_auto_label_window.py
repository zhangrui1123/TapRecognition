"""Strict spike-window behavior for auto-labeling."""

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tap_recognition.labels import detect_second_tap_frame
from tools import auto_label_imu


class AutoLabelWindowTest(unittest.TestCase):
    def detect_pair(self, peaks: list[int]) -> tuple[int, int] | None:
        energy = np.zeros(400, dtype=np.float32)
        energy[peaks] = 1.0
        with (
            patch("tap_recognition.labels.acc_energy_delta", return_value=energy),
            patch("tap_recognition.labels._local_peaks", return_value=peaks),
        ):
            return detect_second_tap_frame(
                np.zeros((400, 3), dtype=np.float32),
                sample_rate=100.0,
                first_tap_after_sec=2.0,
                second_tap_before_sec=2.85,
            )

    def test_strict_window_accepts_interior_pair(self) -> None:
        self.assertEqual(self.detect_pair([201, 230]), (201, 230))

    def test_strict_window_excludes_start_and_end_boundaries(self) -> None:
        self.assertIsNone(self.detect_pair([200, 230]))
        self.assertIsNone(self.detect_pair([250, 285]))

    def test_strict_window_falls_back_to_weak_interior_pair(self) -> None:
        peaks = [50, 68, 210, 228]
        energy = np.zeros(400, dtype=np.float32)
        energy[50] = energy[68] = 1.0
        energy[210] = energy[228] = 1e-7
        with (
            patch("tap_recognition.labels.acc_energy_delta", return_value=energy),
            patch("tap_recognition.labels._local_peaks", return_value=peaks),
        ):
            pair = detect_second_tap_frame(
                np.zeros((400, 3), dtype=np.float32),
                sample_rate=100.0,
                first_tap_after_sec=2.0,
                second_tap_before_sec=2.85,
            )

        self.assertEqual(pair, (210, 228))

    def test_strict_window_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be before"):
            detect_second_tap_frame(
                np.zeros((1, 3), dtype=np.float32),
                first_tap_after_sec=2.85,
                second_tap_before_sec=2.0,
            )

    def test_strict_recording_without_window_stops_before_labeling(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "imuStrict_Miguel_knock_twice_left.csv"
            csv_path.touch()
            stderr = StringIO()
            with (
                patch("sys.argv", ["auto_label_imu.py", directory]),
                patch.object(auto_label_imu, "label_csv") as label_csv,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                auto_label_imu.main()

        self.assertEqual(error.exception.code, 2)
        self.assertIn("refusing to label 1 imuStrict recording(s)", stderr.getvalue())
        label_csv.assert_not_called()

    def test_regular_recording_ignores_strict_window_with_warning(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "imu_Miguel_knock_twice_left.csv"
            csv_path.touch()
            stderr = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "auto_label_imu.py",
                        directory,
                        "--strict-spike-window-ms",
                        "2000",
                        "2850",
                    ],
                ),
                patch.object(
                    auto_label_imu,
                    "label_csv",
                    return_value=(csv_path.with_suffix(".txt"), 1, 0),
                ) as label_csv,
                redirect_stderr(stderr),
            ):
                auto_label_imu.main()

        label_csv.assert_called_once_with(csv_path, 10, None)
        self.assertIn("is ignored for 1 regular imu_ recording(s)", stderr.getvalue())

    def test_strict_recording_receives_strict_window(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "imuStrict_Miguel_knock_twice_left.csv"
            csv_path.touch()
            with (
                patch(
                    "sys.argv",
                    [
                        "auto_label_imu.py",
                        directory,
                        "--strict-spike-window-ms",
                        "2000",
                        "2850",
                    ],
                ),
                patch.object(
                    auto_label_imu,
                    "label_csv",
                    return_value=(csv_path.with_suffix(".txt"), 1, 0),
                ) as label_csv,
            ):
                auto_label_imu.main()

        label_csv.assert_called_once_with(csv_path, 10, (2000.0, 2850.0))


if __name__ == "__main__":
    unittest.main()
