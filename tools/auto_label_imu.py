"""Auto-label IMU CSVs using 3 s collection segments and acc-energy peak pairing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tap_recognition.dataset import load_full_recording, load_segment_row_bounds
from tap_recognition.labels import (
    class_id_from_filename,
    detect_second_tap_frame,
    three_class_frame_labels,
)


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("ascii"))


def _is_strict_recording(csv_path: Path) -> bool:
    return csv_path.name.startswith("imuStrict")


def _is_regular_recording(csv_path: Path) -> bool:
    return csv_path.name.startswith("imu_")


def label_csv(
    csv_path: Path,
    half_frames: int = 10,
    strict_spike_window_ms: tuple[float, float] | None = None,
) -> tuple[Path, int, int]:
    csv_path = Path(csv_path)
    class_id = class_id_from_filename(csv_path.name)
    label_path = csv_path.with_suffix(".txt")

    first_tap_after_sec = None
    second_tap_before_sec = None
    if strict_spike_window_ms is not None:
        start_ms, end_ms = strict_spike_window_ms
        if start_ms < 0 or start_ms >= end_ms:
            raise ValueError("strict spike window requires 0 <= START_MS < END_MS")
        first_tap_after_sec = start_ms / 1000.0
        second_tap_before_sec = end_ms / 1000.0

    if class_id is None:
        _write_text(label_path, "")
        return label_path, 0, 0

    imu, sample_rate = load_full_recording(csv_path)
    bounds = load_segment_row_bounds(csv_path)
    events: list[tuple[int, int]] = []
    n_miss = 0

    if bounds:
        for start, end in bounds:
            pair = detect_second_tap_frame(
                imu[start:end, :3],
                imu[start:end, 3:],
                sample_rate=sample_rate,
                first_tap_after_sec=first_tap_after_sec,
                second_tap_before_sec=second_tap_before_sec,
            )
            if pair is None:
                n_miss += 1
                continue
            _n1, n2 = pair
            events.append((start + int(n2), class_id))
    else:
        pair = detect_second_tap_frame(
            imu[:, :3],
            imu[:, 3:],
            sample_rate=sample_rate,
            first_tap_after_sec=first_tap_after_sec,
            second_tap_before_sec=second_tap_before_sec,
        )
        if pair is None:
            n_miss += 1
        else:
            events.append((int(pair[1]), class_id))

    if events:
        body = "\n".join(f"{frame}, {cls}" for frame, cls in events) + "\n"
        _write_text(label_path, body)
        dense = three_class_frame_labels(
            len(imu), events, half_frames=half_frames
        )
        dense_path = csv_path.with_name(csv_path.stem + ".labels.txt")
        np.savetxt(
            dense_path,
            dense,
            delimiter=",",
            header="p_none,p_left,p_right",
            comments="",
            fmt="%.6f",
        )
    else:
        _write_text(label_path, "")
    return label_path, len(events), n_miss


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label IMU CSV recordings")
    parser.add_argument("paths", nargs="+", help="CSV file or directory")
    parser.add_argument("--half-frames", type=int, default=10)
    parser.add_argument(
        "--strict-spike-window-ms",
        nargs=2,
        type=float,
        metavar=("START_MS", "END_MS"),
        help=(
            "accept a pair only when its first spike is strictly after START_MS "
            "and its second spike is strictly before END_MS, relative to each segment"
        ),
    )
    args = parser.parse_args()
    strict_spike_window_ms = None
    if args.strict_spike_window_ms:
        start_ms, end_ms = args.strict_spike_window_ms
        if start_ms < 0 or start_ms >= end_ms:
            parser.error("--strict-spike-window-ms requires 0 <= START_MS < END_MS")
        strict_spike_window_ms = (start_ms, end_ms)

    csvs: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            csvs.extend(
                p
                for p in sorted(path.glob("*.csv"))
                if (
                    (_is_regular_recording(p) or _is_strict_recording(p))
                    and not p.name.endswith(".labels.csv")
                )
            )
        else:
            csvs.append(path)

    strict_csvs = [csv_path for csv_path in csvs if _is_strict_recording(csv_path)]
    if strict_csvs and strict_spike_window_ms is None:
        parser.error(
            f"refusing to label {len(strict_csvs)} imuStrict recording(s) without "
            "--strict-spike-window-ms START_MS END_MS"
        )

    regular_csvs = [csv_path for csv_path in csvs if _is_regular_recording(csv_path)]
    if regular_csvs and strict_spike_window_ms is not None:
        print(
            "WARNING: --strict-spike-window-ms is ignored for "
            f"{len(regular_csvs)} regular imu_ recording(s); they use unrestricted pairing.",
            file=sys.stderr,
        )

    for csv_path in csvs:
        window_for_csv = None if _is_regular_recording(csv_path) else strict_spike_window_ms
        label_path, n_events, n_miss = label_csv(
            csv_path,
            args.half_frames,
            window_for_csv,
        )
        print(f"{csv_path.name}: events={n_events} miss={n_miss} -> {label_path.name}")


if __name__ == "__main__":
    main()
