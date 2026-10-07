"""PyTorch dataset for synthetic and recorded IMU double-tap windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import (
    CLASS_LEFT,
    CLASS_NONE,
    binary_peak_to_three_class,
    three_class_frame_labels,
)
from .physics import IMUSimulator
from .recording import IMU_COLUMNS, LabelParams, estimate_sample_rate_hz


@dataclass(frozen=True)
class RecordedWindowMeta:
    source_file: str
    trigger_frame: int | None = None


@dataclass(frozen=True)
class SessionExclusion:
    """One raw CSV segment omitted from training and evaluation."""

    source_file: str
    segment_index: int
    reason: str


def load_session_exclusions(
    exclusions_file: str | Path | None,
) -> list[SessionExclusion]:
    """Load the shared CSV exclusion registry, if configured."""
    if exclusions_file is None:
        return []
    path = Path(exclusions_file)
    if not path.exists():
        return []

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        required = {"source_file", "segment_index", "reason"}
        if rows.fieldnames is None or not required <= set(rows.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        exclusions: list[SessionExclusion] = []
        seen: set[tuple[str, int]] = set()
        for line, row in enumerate(rows, start=2):
            source_file = Path(row["source_file"].strip()).name
            reason = row["reason"].strip()
            if not source_file or not reason:
                raise ValueError(f"{path}:{line} requires source_file and reason")
            try:
                segment_index = int(row["segment_index"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line} has invalid segment_index") from exc
            if segment_index < 0:
                raise ValueError(f"{path}:{line} has negative segment_index")
            key = (source_file, segment_index)
            if key in seen:
                raise ValueError(f"{path}:{line} duplicates {source_file} segment {segment_index}")
            seen.add(key)
            exclusions.append(SessionExclusion(source_file, segment_index, reason))
    return exclusions


def load_trigger_frames(label_path: Path) -> list[int]:
    return [frame for frame, _cls in load_trigger_events(label_path)]


def load_trigger_events(label_path: Path) -> list[tuple[int, int]]:
    """Load (frame_index, class_id) from a sidecar .txt. class_id 1=left, 2=right."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    events: list[tuple[int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        frame = int(float(parts[0]))
        class_id = int(float(parts[1])) if len(parts) > 1 else 1
        events.append((frame, class_id))
    events.sort(key=lambda item: item[0])
    return events


def load_full_recording(csv_path: Path) -> tuple[np.ndarray, float]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    timestamp_ms = np.array([float(r["timestamp_ms"]) for r in rows], dtype=np.float64)
    imu = np.array(
        [[float(r[c]) for c in IMU_COLUMNS] for r in rows],
        dtype=np.float32,
    )
    sample_rate = estimate_sample_rate_hz(timestamp_ms)
    return imu, sample_rate


def load_segment_row_bounds(csv_path: Path) -> list[tuple[int, int]]:
    """Half-open [start, end) row ranges for each contiguous segment_index run."""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "segment_index" not in rows[0]:
        return []

    bounds: list[tuple[int, int]] = []
    start = 0
    current = int(rows[0]["segment_index"])
    for i, row in enumerate(rows):
        seg = int(row["segment_index"])
        if seg != current:
            bounds.append((start, i))
            start = i
            current = seg
    bounds.append((start, len(rows)))
    return bounds


def load_indexed_segment_row_bounds(csv_path: Path) -> list[tuple[int, int, int]]:
    """Return (segment_index, start, end) for contiguous CSV segment runs."""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "segment_index" not in rows[0]:
        return []

    bounds: list[tuple[int, int, int]] = []
    start = 0
    current = int(rows[0]["segment_index"])
    for i, row in enumerate(rows):
        segment_index = int(row["segment_index"])
        if segment_index != current:
            bounds.append((current, start, i))
            start = i
            current = segment_index
    bounds.append((current, start, len(rows)))
    return bounds


class IMUDoubleTapDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 4000,
        window_samples: int = 300,
        sample_rate: float = 100.0,
        seed: int = 42,
        highpass: bool = True,
    ):
        self.sim = IMUSimulator(
            sample_rate=sample_rate,
            window_samples=window_samples,
            seed=seed,
        )
        self.highpass = highpass
        self.samples: list[tuple] = []
        for i in range(num_samples):
            y, peak, meta = self.sim.generate_window()
            if highpass:
                y = self.sim.highpass(y)
            class_id = CLASS_LEFT if meta.label == 1 else CLASS_NONE
            labels = binary_peak_to_three_class(peak, class_id)
            self.samples.append((y, labels, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y, labels, window_label = self.samples[idx]
        return {
            "imu": torch.from_numpy(y),
            "frame_labels": torch.from_numpy(labels),
            "window_label": torch.tensor(window_label, dtype=torch.long),
        }


class RecordedIMUDataset(Dataset):
    """Sliding-window dataset built from labeled CSV recordings."""

    def __init__(
        self,
        data_dir: str | Path,
        window_samples: int = 300,
        label_params: LabelParams | None = None,
        highpass: bool = True,
        negative_windows_per_file: int = 20,
        trigger_context_samples: int = 200,
        exclusion_margin: int = 200,
        dt_min: float = 0.12,
        dt_max: float = 0.45,
        seed: int = 42,
        csv_paths: Sequence[str | Path] | None = None,
        session_exclusions_file: str | Path | None = None,
    ):
        self.window_samples = window_samples
        self.label_params = label_params or LabelParams()
        self.highpass = highpass
        self.trigger_context_samples = trigger_context_samples
        self.exclusion_margin = exclusion_margin
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.rng = np.random.default_rng(seed)
        self.sim = IMUSimulator(
            sample_rate=100.0,
            window_samples=window_samples,
            seed=seed,
        )
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.window_meta: list[RecordedWindowMeta] = []
        self.session_exclusions = load_session_exclusions(session_exclusions_file)
        self._excluded_by_file: dict[str, set[int]] = {}
        for exclusion in self.session_exclusions:
            self._excluded_by_file.setdefault(exclusion.source_file, set()).add(
                exclusion.segment_index
            )

        csv_files = sorted(
            p for p in Path(data_dir).glob("*.csv") if not p.name.endswith(".labels.csv")
        )
        if csv_paths is not None:
            allowed = {Path(p).resolve() for p in csv_paths}
            csv_files = [p for p in csv_files if p.resolve() in allowed]
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        for csv_path in csv_files:
            imu, sample_rate = load_full_recording(csv_path)
            if highpass:
                self.sim.sample_rate = sample_rate
                imu = self.sim.highpass(imu)

            label_path = csv_path.with_suffix(".txt")
            trigger_events = load_trigger_events(label_path)
            trigger_frames = [frame for frame, _cls in trigger_events]
            indexed_segments = load_indexed_segment_row_bounds(csv_path)
            excluded = self._excluded_by_file.get(csv_path.name, set())
            if indexed_segments:
                available = {segment_index for segment_index, _start, _end in indexed_segments}
                unknown = excluded - available
                if unknown:
                    raise ValueError(
                        f"{csv_path.name} does not contain excluded segment(s): {sorted(unknown)}"
                    )
                if excluded:
                    print(f"  Excluding session(s) {sorted(excluded)} from {csv_path.name}")
                segments = [
                    (start, end)
                    for segment_index, start, end in indexed_segments
                    if segment_index not in excluded
                ]
            else:
                if excluded:
                    raise ValueError(
                        f"{csv_path.name} has exclusions but no segment_index column"
                    )
                segments = []

            if indexed_segments:
                self._add_segment_windows(
                    csv_path.name, imu, segments, trigger_events
                )
                continue

            for trigger_frame, class_id in trigger_events:
                window, labels = self._window_for_trigger(
                    imu, trigger_frame, sample_rate, class_id
                )
                if window is not None:
                    self.samples.append((window, labels, class_id))
                    self.window_meta.append(
                        RecordedWindowMeta(csv_path.name, trigger_frame)
                    )

            negative_starts = self._sample_negative_starts(
                len(imu),
                trigger_frames,
                negative_windows_per_file,
            )
            none_labels = three_class_frame_labels(window_samples, [])
            for start in negative_starts:
                window = imu[start : start + window_samples]
                if len(window) < window_samples:
                    continue
                self.samples.append(
                    (window.astype(np.float32), none_labels.copy(), CLASS_NONE)
                )
                self.window_meta.append(RecordedWindowMeta(csv_path.name, None))

        if not self.samples:
            raise ValueError(f"No training windows built from {data_dir}")

    def _window_for_trigger(
        self,
        imu: np.ndarray,
        trigger_frame: int,
        sample_rate: float,
        class_id: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if trigger_frame < 0 or trigger_frame >= len(imu):
            return None, None

        start = trigger_frame - self.trigger_context_samples
        end = start + self.window_samples

        if start < 0:
            start = 0
            end = self.window_samples
        if end > len(imu):
            end = len(imu)
            start = max(0, end - self.window_samples)

        window = imu[start:end]
        if len(window) < self.window_samples:
            return None, None

        t2_frame_in_window = trigger_frame - start
        labels = three_class_frame_labels(
            self.window_samples,
            [(t2_frame_in_window, class_id)],
            half_frames=self.label_params.half_frames,
        )
        return window.astype(np.float32), labels

    def _add_segment_windows(
        self,
        source_name: str,
        imu: np.ndarray,
        segments: list[tuple[int, int]],
        trigger_events: list[tuple[int, int]],
    ) -> None:
        """One training sequence per 3 s collection segment."""
        for seg_start, seg_end in segments:
            local_events = [
                (frame - seg_start, class_id)
                for frame, class_id in trigger_events
                if seg_start <= frame < seg_end
            ]
            if trigger_events and not local_events:
                continue
            window, labels = self._fit_to_window(
                imu[seg_start:seg_end], local_events
            )
            class_id = local_events[0][1] if local_events else CLASS_NONE
            trigger_frame = (
                seg_start + local_events[0][0] if local_events else None
            )
            self.samples.append((window, labels, class_id))
            self.window_meta.append(RecordedWindowMeta(source_name, trigger_frame))

    def _fit_to_window(
        self,
        imu_seg: np.ndarray,
        local_events: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.window_samples
        n = len(imu_seg)
        half = self.label_params.half_frames

        if n == target:
            labels = three_class_frame_labels(n, local_events, half_frames=half)
            return imu_seg.astype(np.float32), labels

        if n > target:
            if local_events:
                n2 = max(frame for frame, _cls in local_events)
                start = n2 - self.trigger_context_samples
                start = max(0, min(start, n - target))
            else:
                start = 0
            end = start + target
            shifted = [
                (frame - start, class_id)
                for frame, class_id in local_events
                if start <= frame < end
            ]
            labels = three_class_frame_labels(target, shifted, half_frames=half)
            return imu_seg[start:end].astype(np.float32), labels

        pad = target - n
        window = np.pad(imu_seg, ((0, pad), (0, 0)), mode="edge")
        labels = three_class_frame_labels(n, local_events, half_frames=half)
        labels = np.concatenate(
            [labels, three_class_frame_labels(pad, [], half_frames=half)],
            axis=0,
        )
        return window.astype(np.float32), labels

    def _sample_negative_starts(
        self,
        n_samples: int,
        trigger_frames: list[int],
        count: int,
    ) -> list[int]:
        if n_samples <= self.window_samples:
            return []

        forbidden: list[tuple[int, int]] = []
        for frame in trigger_frames:
            forbidden.append(
                (
                    max(0, frame - self.exclusion_margin),
                    min(n_samples, frame + self.exclusion_margin),
                )
            )

        starts: list[int] = []
        attempts = 0
        max_attempts = count * 50
        while len(starts) < count and attempts < max_attempts:
            attempts += 1
            start = int(self.rng.integers(0, n_samples - self.window_samples + 1))
            end = start + self.window_samples
            if any(not (end <= lo or start >= hi) for lo, hi in forbidden):
                continue
            starts.append(start)
        return starts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y, labels, window_label = self.samples[idx]
        return {
            "imu": torch.from_numpy(y),
            "frame_labels": torch.from_numpy(labels),
            "window_label": torch.tensor(window_label, dtype=torch.long),
        }
