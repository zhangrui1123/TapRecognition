"""
Online (causal) double-tap detector with persistent GRU state.

Suitable for real-time IMU streams: one sample in, one probability out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from .labels import confirm_prior_tap
from .model_factory import build_model
from .physics import IMUSimulator


@dataclass
class DetectorConfig:
    threshold_on: float = 0.50
    threshold_off: float = 0.3
    consecutive_on: int = 1
    look_ahead_sec: float = 0.12
    refractory_sec: float = 0.50
    highpass_cutoff: float = 0.5
    sample_rate: float = 100.0
    ema_alpha: float = 0.01
    require_prior_tap: bool = False
    prior_tap_mad_k: float = 5.0


@dataclass
class DetectionEvent:
    frame_index: int
    time_sec: float
    probability: float
    class_id: int = 0


def pick_tap_events(
    probs: np.ndarray,
    *,
    threshold: float = 0.50,
    look_ahead_sec: float = 0.12,
    refractory_sec: float = 0.50,
    sample_rate: float = 100.0,
    consecutive_on: int = 1,
    acc: np.ndarray | None = None,
    require_prior_tap: bool = False,
    prior_tap_mad_k: float = 5.0,
) -> list[tuple[int, int, float]]:
    """
    One report per physical double-tap.

    On first frames over threshold, wait up to look_ahead_sec,
    emit the strongest left/right peak, then ignore refractory_sec so
    ring-down cannot fire again.
    """
    n = len(probs)
    look = max(1, int(round(look_ahead_sec * sample_rate)))
    lock = max(1, int(round(refractory_sec * sample_rate)))
    events: list[tuple[int, int, float]] = []
    i = 0
    streak = 0
    while i < n:
        event_p = float(probs[i, 1:].max())
        if event_p < threshold:
            streak = 0
            i += 1
            continue
        streak += 1
        if streak < consecutive_on:
            i += 1
            continue
        start = i - consecutive_on + 1
        end = min(n, start + look)
        local = probs[start:end, 1:]
        rel = int(np.argmax(local.max(axis=1)))
        peak = start + rel
        class_id = int(np.argmax(probs[peak, 1:])) + 1
        peak_p = float(probs[peak, class_id])
        if require_prior_tap and acc is not None:
            if not confirm_prior_tap(
                acc,
                peak,
                sample_rate=sample_rate,
                mad_k=prior_tap_mad_k,
            ):
                i += 1
                streak = 0
                continue
        events.append((peak, class_id, peak_p))
        i = peak + lock
        streak = 0
    return events


class TrainingHighPass:
    """2nd-order Butterworth high-pass matching dataset.py (scipy lfilter)."""

    B = (0.9780304792065597, -1.9560609584131194, 0.9780304792065597)
    A1 = -1.9555782403150352
    A2 = 0.9565436765112031

    def __init__(self, channels: int = 6):
        self._x1 = np.zeros(channels, dtype=np.float32)
        self._x2 = np.zeros(channels, dtype=np.float32)
        self._y1 = np.zeros(channels, dtype=np.float32)
        self._y2 = np.zeros(channels, dtype=np.float32)

    def filter(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b0, b1, b2 = self.B
        for i in range(len(x)):
            yn = (
                b0 * x[i] + b1 * self._x1[i] + b2 * self._x2[i]
                - self.A1 * self._y1[i] - self.A2 * self._y2[i]
            )
            self._x2[i] = self._x1[i]
            self._x1[i] = x[i]
            self._y2[i] = self._y1[i]
            self._y1[i] = yn
            y[i] = yn
        return y

    def reset(self) -> None:
        self._x1.fill(0)
        self._x2.fill(0)
        self._y1.fill(0)
        self._y2.fill(0)


class CausalHighPass:
    """Per-channel one-pole high-pass (causal, for streaming)."""

    def __init__(self, cutoff: float, sample_rate: float, channels: int = 6):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff)
        self.alpha = rc / (rc + dt)
        self.prev_x = np.zeros(channels, dtype=np.float32)
        self.prev_y = np.zeros(channels, dtype=np.float32)

    def filter(self, x: np.ndarray) -> np.ndarray:
        y = self.alpha * (self.prev_y + x - self.prev_x)
        self.prev_x = x.copy()
        self.prev_y = y.copy()
        return y


class RunningNormalizer:
    """Causal EMA z-score normalization."""

    def __init__(self, channels: int = 6, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = np.zeros(channels, dtype=np.float32)
        self.var = np.ones(channels, dtype=np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        diff = x - self.mean
        self.var = (1 - self.alpha) * self.var + self.alpha * (diff ** 2)
        return diff / (np.sqrt(self.var) + 1e-6)


class OnlineDoubleTapDetector:
    """
    Streaming detector wrapping a causal recurrent model with preprocessing and hysteresis.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DetectorConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.config = config or DetectorConfig()
        self.device = torch.device(device)

        self._h: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        self._cnn_buffer: torch.Tensor | None = None
        self._hp = TrainingHighPass()
        self._acc_hp: list[np.ndarray] = []

        self._frame_idx = 0
        self._consecutive = 0
        self._pending_start: int | None = None
        self._pending_best: tuple[int, int, float] | None = None
        self._refractory_until = 0.0
        self.events: list[DetectionEvent] = []

    def reset(self) -> None:
        self._h = None
        self._cnn_buffer = None
        self._hp = TrainingHighPass()
        self._acc_hp = []
        self._frame_idx = 0
        self._consecutive = 0
        self._pending_start = None
        self._pending_best = None
        self._refractory_until = 0.0
        self.events.clear()

    def _emit_pending(self) -> bool:
        if self._pending_best is None:
            self._pending_start = None
            return False
        frame, class_id, p_event = self._pending_best
        if self.config.require_prior_tap and self._acc_hp:
            acc = np.stack(self._acc_hp, axis=0)
            if not confirm_prior_tap(
                acc,
                frame,
                sample_rate=self.config.sample_rate,
                mad_k=self.config.prior_tap_mad_k,
            ):
                self._pending_start = None
                self._pending_best = None
                self._consecutive = 0
                return False
        t = frame / self.config.sample_rate
        self.events.append(DetectionEvent(frame, t, p_event, class_id))
        self._refractory_until = t + self.config.refractory_sec
        self._pending_start = None
        self._pending_best = None
        self._consecutive = 0
        return True

    @torch.no_grad()
    def process_sample(self, imu_sample: np.ndarray) -> tuple[float, bool]:
        """
        Process one IMU sample [6].

        Returns:
            (event probability, triggered_this_step)
        """
        x = self._hp.filter(imu_sample.astype(np.float32))
        self._acc_hp.append(x[:3].copy())

        xt = torch.from_numpy(x).view(1, 1, -1).to(self.device)
        prob, self._h, self._cnn_buffer = self.model.step(
            xt, self._h, self._cnn_buffer
        )
        p_vec = prob[0, 0].cpu().numpy()
        p_event = float(p_vec[1:].max())
        class_id = int(p_vec[1:].argmax()) + 1
        t = self._frame_idx / self.config.sample_rate
        triggered = False

        if self._pending_start is not None:
            if self._pending_best is None or p_event > self._pending_best[2]:
                self._pending_best = (self._frame_idx, class_id, p_event)
            if t >= self._pending_start + self.config.look_ahead_sec:
                triggered = self._emit_pending()
        elif t >= self._refractory_until:
            if p_event >= self.config.threshold_on:
                self._consecutive += 1
            else:
                self._consecutive = 0
            if self._consecutive >= self.config.consecutive_on:
                start_t = (
                    self._frame_idx - self.config.consecutive_on + 1
                ) / self.config.sample_rate
                self._pending_start = start_t
                self._pending_best = (self._frame_idx, class_id, p_event)

        self._frame_idx += 1
        return p_event, triggered

    def process_stream(self, imu: np.ndarray) -> tuple[np.ndarray, list[DetectionEvent]]:
        """Process full array [T, 6] sample-by-sample."""
        self.reset()
        probs = np.zeros(len(imu), dtype=np.float32)
        for i, sample in enumerate(imu):
            probs[i], _ = self.process_sample(sample)
        if self._pending_best is not None:
            self._emit_pending()
        return probs, list(self.events)


def load_detector(checkpoint_path: str, device: str = "cpu") -> OnlineDoubleTapDetector:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    if "dilations" in cfg:
        cfg["dilations"] = tuple(cfg["dilations"])
    model = build_model(cfg, ckpt.get("model_type")).to(device)
    model.load_state_dict(ckpt["model_state"])
    return OnlineDoubleTapDetector(model, device=device)


def _demo_inference(checkpoint: str) -> None:
    import matplotlib.pyplot as plt

    sim = IMUSimulator(sample_rate=100, window_samples=128, seed=99)
    detector = load_detector(checkpoint)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax_idx, kind in enumerate(["double", "single"]):
        imu, labels, meta = sim.generate_window(kind=kind)
        imu_hp = sim.highpass(imu)

        probs, events = detector.process_stream(imu_hp)
        t = np.arange(len(imu)) / sim.sample_rate

        ax = axes[ax_idx]
        ax.plot(t, imu_hp[:, 0], alpha=0.5, label="ax (high-pass)")
        ax2 = ax.twinx()
        ax2.plot(t, probs, "r-", linewidth=2, label="P(double-tap)")
        ax2.axhline(0.7, color="gray", linestyle="--", alpha=0.5)
        for ev in events:
            ax2.axvline(ev.time_sec, color="green", linestyle=":", alpha=0.8)
        ax.set_title(f"{kind} — label={meta.label}, detections={len(events)}")
        ax.set_ylabel("accel")
        ax2.set_ylabel("probability")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    axes[1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig("inference_demo.png", dpi=120)
    print("Saved inference_demo.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online IMU double-tap inference")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    detector = load_detector(args.checkpoint, args.device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Receptive field: {detector.model.receptive_field} samples")
    _demo_inference(args.checkpoint)


if __name__ == "__main__":
    main()
