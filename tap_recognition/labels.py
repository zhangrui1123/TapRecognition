"""Gaussian second-tap labels and IMU peak pairing (mathematical_model.md)."""

from __future__ import annotations

import numpy as np

DT_MIN = 0.12
DT_MAX = 0.45
NUM_CLASSES = 3  # none, left second-tap, right second-tap
CLASS_NONE = 0
CLASS_LEFT = 1
CLASS_RIGHT = 2
LABEL_HALF_FRAMES = 10


def is_valid_delta_t(
    delta_t: float,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
) -> bool:
    """Return True when the inter-tap interval is within the allowed range."""
    return dt_min <= delta_t <= dt_max


def class_id_from_filename(name: str) -> int | None:
    """Map collection filename to second-tap class; None means negative file."""
    lower = name.lower()
    if "knock_twice_left" in lower:
        return CLASS_LEFT
    if "knock_twice_right" in lower:
        return CLASS_RIGHT
    if "knock_twice" in lower:
        return CLASS_LEFT
    return None


def acc_energy_delta(acc: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(acc.astype(np.float32), axis=1)
    return np.abs(np.diff(mag, prepend=mag[0]))


def _local_peaks(x: np.ndarray, k: int = 3) -> list[int]:
    peaks: list[int] = []
    for i in range(k, len(x) - k):
        if x[i] >= x[i - k : i].max() and x[i] >= x[i + 1 : i + 1 + k].max():
            peaks.append(i)
    return peaks


def detect_second_tap_frame(
    acc: np.ndarray,
    gyro: np.ndarray | None = None,
    *,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
    sample_rate: float = 100.0,
    first_tap_after_sec: float | None = None,
    second_tap_before_sec: float | None = None,
) -> tuple[int, int] | None:
    """
    Find (n1, n2) in one 3 s collection segment using acc-energy peaks.

    When provided, first_tap_after_sec and second_tap_before_sec define a
    strict pair window. The first tap must be after its start and the second
    tap must be before its end. Strong candidates are preferred, but if none
    fall in the requested window, the best local-peak pair in that window is
    returned so weak intended taps are not hidden by stronger early activity.
    """
    del gyro  # reserved; pairing is driven by acc energy
    if first_tap_after_sec is not None and first_tap_after_sec < 0:
        raise ValueError("first_tap_after_sec must be non-negative")
    if second_tap_before_sec is not None and second_tap_before_sec < 0:
        raise ValueError("second_tap_before_sec must be non-negative")
    if (
        first_tap_after_sec is not None
        and second_tap_before_sec is not None
        and first_tap_after_sec >= second_tap_before_sec
    ):
        raise ValueError("first_tap_after_sec must be before second_tap_before_sec")

    acc_d = acc_energy_delta(acc)
    med = float(np.median(acc_d))
    mad = float(np.median(np.abs(acc_d - med))) + 1e-6
    thr = med + 6.0 * mad
    peaks = [i for i in _local_peaks(acc_d, 3) if acc_d[i] >= thr]
    if len(peaks) < 2:
        order = np.argsort(acc_d)[::-1]
        cand: list[int] = []
        for i in order[:12]:
            idx = int(i)
            if all(abs(idx - j) >= 8 for j in cand):
                cand.append(idx)
            if len(cand) >= 4:
                break
        peaks = sorted(cand)

    def valid_pairs(candidate_peaks: list[int]) -> list[tuple[int, int, float, float, float]]:
        pairs: list[tuple[int, int, float, float, float]] = []
        for i, t1 in enumerate(candidate_peaks):
            for t2 in candidate_peaks[i + 1 :]:
                delta_t = (t2 - t1) / sample_rate
                if (
                    dt_min <= delta_t <= dt_max
                    and (
                        first_tap_after_sec is None
                        or t1 / sample_rate > first_tap_after_sec
                    )
                    and (
                        second_tap_before_sec is None
                        or t2 / sample_rate < second_tap_before_sec
                    )
                ):
                    pairs.append((t1, t2, delta_t, float(acc_d[t1]), float(acc_d[t2])))
        return pairs

    pairs = valid_pairs(peaks)
    if not pairs and first_tap_after_sec is not None:
        # The global robust threshold can be dominated by an early accidental
        # movement. Within an explicitly requested window, rank all local peaks
        # instead of declaring the intended, weaker pair missing.
        pairs = valid_pairs(_local_peaks(acc_d, 3))
    if not pairs:
        return None
    pairs.sort(key=lambda p: (min(p[3], p[4]), p[1]), reverse=True)
    return pairs[0][0], pairs[0][1]


def confirm_prior_tap(
    acc: np.ndarray,
    n2: int,
    *,
    sample_rate: float = 100.0,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
    mad_k: float = 5.0,
    local_k: int = 2,
) -> bool:
    """
    Causal double-tap check: n2 must be a local acc-energy peak, and a
    prior peak n1 must exist with Δt in [dt_min, dt_max] and both above
    a robust energy threshold. No fallback to weaker peaks.
    """
    if n2 < 0 or n2 >= len(acc):
        return False
    acc_d = acc_energy_delta(acc[:, :3] if acc.ndim == 2 else acc)
    if n2 >= len(acc_d):
        return False
    slop = 8
    lo = max(0, n2 - slop)
    hi = min(len(acc_d), n2 + slop + 1)
    n2 = lo + int(np.argmax(acc_d[lo:hi]))
    k = local_k
    if n2 < k or n2 + k >= len(acc_d):
        return False
    if float(acc_d[n2]) < float(acc_d[n2 - k : n2 + k + 1].max()) - 1e-9:
        return False

    i_lo = n2 - int(round(dt_max * sample_rate))
    i_hi = n2 - int(round(dt_min * sample_rate))
    if i_hi <= 0 or i_lo >= n2:
        return False
    i_lo = max(k, i_lo)
    if i_hi <= i_lo:
        return False

    baseline = acc_d[max(0, n2 - int(round(sample_rate))) : n2 + 1]
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med))) + 1e-6
    thr = med + mad_k * mad
    if float(acc_d[n2]) < thr:
        return False

    region = acc_d[i_lo : i_hi + 1]
    n1 = i_lo + int(np.argmax(region))
    if float(acc_d[n1]) < thr:
        return False
    if n1 < k or float(acc_d[n1]) < float(acc_d[n1 - k : n1 + k + 1].max()) - 1e-9:
        return False
    return is_valid_delta_t((n2 - n1) / sample_rate, dt_min, dt_max)


def gaussian_second_tap_labels(
    n_samples: int,
    sample_rate: float,
    t2_sec: float | None,
    *,
    sigma: float = 0.04,
    peak_offset: float = 0.02,
) -> np.ndarray:
    """Legacy 1-D Gaussian peak at the second tap. Prefer three_class_frame_labels."""
    labels = np.zeros(n_samples, dtype=np.float32)
    if t2_sec is None:
        return labels

    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    t_peak = t2_sec + peak_offset
    labels = np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)
    return labels.astype(np.float32)


def three_class_frame_labels(
    n_samples: int,
    triggers: list[tuple[int, int]],
    *,
    half_frames: int = LABEL_HALF_FRAMES,
) -> np.ndarray:
    """
    Per-frame labels of shape [N, 3] = (none, left, right).

    At each trigger frame n2 of class c in {1, 2}: (0,1,0) or (0,0,1).
    ±half_frames are Gaussian-blended back to (1,0,0).
    """
    labels = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    labels[:, CLASS_NONE] = 1.0
    if half_frames <= 0:
        sigma = 1.0
    else:
        sigma = half_frames / 3.0

    for n2, class_id in triggers:
        if class_id not in (CLASS_LEFT, CLASS_RIGHT):
            continue
        lo = max(0, n2 - half_frames)
        hi = min(n_samples - 1, n2 + half_frames)
        for n in range(lo, hi + 1):
            weight = float(np.exp(-0.5 * ((n - n2) / sigma) ** 2))
            labels[n, CLASS_NONE] = 1.0 - weight
            labels[n, CLASS_LEFT] = weight if class_id == CLASS_LEFT else 0.0
            labels[n, CLASS_RIGHT] = weight if class_id == CLASS_RIGHT else 0.0
    return labels


def binary_peak_to_three_class(
    peak: np.ndarray,
    class_id: int = CLASS_LEFT,
) -> np.ndarray:
    """Convert a 1-D peak in [0, 1] to (none, left, right) labels."""
    weight = np.clip(peak.astype(np.float32), 0.0, 1.0)
    labels = np.zeros((len(weight), NUM_CLASSES), dtype=np.float32)
    labels[:, CLASS_NONE] = 1.0 - weight
    if class_id in (CLASS_LEFT, CLASS_RIGHT):
        labels[:, class_id] = weight
    return labels


def estimate_first_tap_frame(imu: np.ndarray, t2_frame: int) -> int | None:
    """
    Estimate the first tap as the strongest acc-energy delta before t2.

    Used to filter recorded training windows by inter-tap interval.
    """
    if t2_frame <= 0:
        return None

    acc_delta = acc_energy_delta(imu[:, :3])
    if t2_frame >= len(acc_delta):
        return None

    t1_frame = int(np.argmax(acc_delta[:t2_frame]))
    if t1_frame <= 0 or t1_frame >= t2_frame:
        return None
    return t1_frame
