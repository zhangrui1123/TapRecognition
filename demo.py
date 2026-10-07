"""
End-to-end demo: mathematical model visualization → training → recorded-file inference.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from tap_recognition.config import LabelConfig, TrainConfig
from train import train
from tap_recognition.dataset import (
    RecordedIMUDataset,
    load_full_recording,
    load_trigger_events,
    load_trigger_frames,
)
from tap_recognition.inference import OnlineDoubleTapDetector
from tap_recognition.labels import estimate_first_tap_frame, three_class_frame_labels
from tap_recognition.model import CausalCNNGRU
from tap_recognition.physics import IMUSimulator
from train import evaluate


def plot_physics_model(out_path: str = "physics_model.png") -> None:
    """Visualize the mathematical tap model from docs/mathematical_model.md."""
    sim = IMUSimulator(sample_rate=200, window_samples=200, seed=7)
    t_axis = np.arange(200) / sim.sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    kinds = [
        ("double", "Double-tap: two damped responses"),
        ("single", "Single tap (negative class)"),
        ("background", "Background only"),
    ]
    for ax, (kind, title) in zip(axes, kinds):
        y, labels, meta = sim.generate_window(kind=kind)
        y_hp = sim.highpass(y)
        ax.plot(t_axis, y_hp[:, 0], label="ax")
        ax.plot(t_axis, y_hp[:, 3], label="gx", alpha=0.7)
        for tt in meta.tap_times:
            ax.axvline(tt, color="red", linestyle="--", alpha=0.6)
        if labels.max() > 0:
            onset = np.argmax(labels > 0.5) / sim.sample_rate
            ax.axvline(onset, color="green", linestyle=":", label="label onset")
        ax.set_title(title)
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("IMU Double-Tap Physical Model (high-pass filtered)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


IMU_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
IMU_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728", "#8c564b"]


def _mark_triggers(
    ax: plt.Axes,
    trigger_frames: list[int],
    pred_frames: list[int],
    t1_frames: list[int] | None = None,
    time_axis: np.ndarray | None = None,
) -> None:
    x = time_axis if time_axis is not None else trigger_frames
    for frame in trigger_frames:
        xv = time_axis[frame] if time_axis is not None else frame
        ax.axvline(xv, color="#cc3333", linestyle="--", alpha=0.65, linewidth=0.9)
    if t1_frames:
        for frame in t1_frames:
            if frame is not None:
                xv = time_axis[frame] if time_axis is not None else frame
                ax.axvline(xv, color="#e69138", linestyle=":", alpha=0.7, linewidth=0.9)
    for frame in pred_frames:
        xv = time_axis[frame] if time_axis is not None else frame
        ax.axvline(xv, color="#33aa33", linestyle="-", alpha=0.75, linewidth=1.0)


def plot_detailed_pipeline(
    cfg: TrainConfig,
    out_path: str = "pipeline_detailed.png",
) -> None:
    """Draw the full data → train → deploy pipeline."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color="#e8f0fe", edge="#3366cc", fontsize=8):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color, edgecolor=edge, linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    ax.text(7, 9.6, "IMU Double-Tap Recognition — Detailed Pipeline", ha="center", fontsize=13, fontweight="bold")

    # --- Data collection ---
    ax.text(1.2, 8.85, "1. Data collection", fontsize=10, fontweight="bold", color="#333")
    box(0.3, 7.5, 2.4, 0.9, "IMU CSV\n6-ch @ 100 Hz", "#fff3e0", "#e69138")
    box(3.1, 7.5, 2.6, 0.9, "auto_label_imu.py\nACC peak pairing", "#fff3e0", "#e69138")
    box(6.1, 7.5, 2.5, 0.9, "Sidecar .txt\ntrigger frame n₂", "#fff3e0", "#e69138")
    arrow(2.7, 7.95, 3.1, 7.95)
    arrow(5.7, 7.95, 6.1, 7.95)

    # --- Preprocess & windowing ---
    ax.text(1.2, 6.85, "2. Preprocess & windowing", fontsize=10, fontweight="bold", color="#333")
    box(0.3, 5.5, 2.2, 0.9, "High-pass\n(gravity removal)", "#e8f5e9", "#2e7d32")
    box(2.8, 5.5, 2.8, 0.9, f"{cfg.data.window_samples}-frame window\n3 s segment / context={cfg.data.trigger_context_samples}", "#e8f5e9", "#2e7d32")
    box(5.9, 5.5, 2.6, 0.9, f"Δt filter\n[{cfg.data.dt_min:.2f}, {cfg.data.dt_max:.2f}] s", "#e8f5e9", "#2e7d32")
    box(8.8, 5.5, 2.4, 0.9, f"Gaussian labels\nσ={cfg.labels.sigma}s", "#e8f5e9", "#2e7d32")
    arrow(2.5, 5.95, 2.8, 5.95)
    arrow(5.6, 5.95, 5.9, 5.95)
    arrow(8.5, 5.95, 8.8, 5.95)
    arrow(7.35, 7.5, 1.4, 6.4)

    # --- Model ---
    ax.text(1.2, 4.85, "3. Causal CNN + GRU (per frame)", fontsize=10, fontweight="bold", color="#333")
    model_boxes = [
        (0.3, 3.5, 1.5, 0.9, "Linear\n6→32"),
        (2.1, 3.5, 1.7, 0.9, "Causal Conv\nk=5, d=1,2,4"),
        (4.1, 3.5, 1.3, 0.9, "GRU\nh=64"),
        (5.7, 3.5, 1.5, 0.9, "MLP head\n→ logit"),
        (7.5, 3.5, 1.5, 0.9, "σ(logit)\nP(tap)"),
    ]
    for x, y, w, h, text in model_boxes:
        box(x, y, w, h, text, "#e8f0fe", "#3366cc")
    for i in range(len(model_boxes) - 1):
        arrow(model_boxes[i][0] + model_boxes[i][2], 3.95,
              model_boxes[i + 1][0], 3.95)
    ax.text(4.5, 3.15, f"Receptive field R={1 + sum((cfg.model.kernel_size - 1) * d for d in cfg.model.dilations)} frames (causal, left-pad)", ha="center", fontsize=8, color="#555")

    # --- Training ---
    ax.text(1.2, 2.65, "4. Training", fontsize=10, fontweight="bold", color="#333")
    box(0.3, 1.3, 2.6, 0.9, "BCE loss\nframe labels yₙ", "#f3e5f5", "#7b1fa2")
    box(3.2, 1.3, 2.4, 0.9, "Window acc\nmax(P)>0.5", "#f3e5f5", "#7b1fa2")
    box(5.9, 1.3, 2.2, 0.9, "Early stop\nbest.pt", "#f3e5f5", "#7b1fa2")
    arrow(9.5, 5.5, 1.6, 2.2)
    arrow(4.5, 3.5, 1.6, 2.2)

    # --- Deploy ---
    ax.text(8.2, 2.65, "5. Deploy (streaming)", fontsize=10, fontweight="bold", color="#333")
    box(8.0, 1.3, 2.2, 0.9, "Causal HP\n+ EMA norm", "#fce4ec", "#c62828")
    box(10.5, 1.3, 1.8, 0.9, "NN step\n[1,1,6]→[1,1,1]\ncarry h, buf", "#fce4ec", "#c62828")
    box(12.2, 1.3, 1.5, 0.9, "Hysteresis\ntrigger", "#fce4ec", "#c62828")
    arrow(8.25, 3.5, 9.1, 2.2)
    arrow(10.2, 1.75, 10.5, 1.75)
    arrow(12.0, 1.75, 12.2, 1.75)
    box(8.0, 0.15, 5.7, 0.75, "Export: best.pt → ONNX → MindSpore Lite .ms (HarmonyOS)", "#fffde7", "#f9a825", fontsize=7.5)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_architecture(out_path: str = "architecture.png") -> None:
    """Compact network schematic."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        (0.3, 0.8, 1.2, 0.7, "IMU 6-ch"),
        (1.9, 0.8, 1.5, 0.7, "Causal CNN"),
        (3.8, 0.8, 1.0, 0.7, "GRU"),
        (5.2, 0.8, 1.2, 0.7, "σ(logit)"),
        (6.8, 0.8, 1.2, 0.7, "Hysteresis"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#e8f0fe", edgecolor="#3366cc")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][0], 1.15), xytext=(boxes[i][0] + boxes[i][2], 1.15),
                    arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(5, 2.1, "Causal CNN + GRU — see pipeline_detailed.png for full flow", ha="center", fontsize=10)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def run_training(cfg: TrainConfig, force: bool = False) -> Path:
    ckpt = Path(cfg.out_dir) / "best.pt"
    if ckpt.exists() and not force:
        print("Checkpoint exists, skipping training (use --force-train to retrain)")
        return ckpt

    print("Training causal CNN + GRU...")
    train(cfg)
    return ckpt


def export_to_mindspore(
    checkpoint: Path,
    cfg: TrainConfig,
    onnx_path: Path = Path("checkpoints/tap_recognition.onnx"),
    ms_output: Path = Path("checkpoints/tap_recognition"),
    ms_lite_root: str = r"D:\mindspore-lite-2.9.0-win-x64",
    mingw_bin: str = "",
) -> Path:
    """Export PyTorch checkpoint to ONNX, then convert to MindSpore Lite (.ms)."""
    print("Exporting checkpoint to ONNX for MindSpore Lite...")
    export_cmd = [
        sys.executable,
        "tools/export_step_for_harmony.py",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(onnx_path),
        "--window-samples",
        str(cfg.data.window_samples),
        "--input-dim",
        str(cfg.model.input_dim),
    ]
    subprocess.run(export_cmd, check=True)

    ms_path = ms_output.with_suffix(".ms")
    print("Converting ONNX to MindSpore Lite (.ms)...")
    convert_cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools/convert_to_ms.ps1",
        "-OnnxFile",
        str(onnx_path),
        "-OutputFile",
        str(ms_output),
        "-MsLiteRoot",
        ms_lite_root,
    ]
    if mingw_bin:
        convert_cmd.extend(["-MingwBin", mingw_bin])
    subprocess.run(convert_cmd, check=True)

    if not ms_path.exists():
        raise FileNotFoundError(f"MindSpore export failed: {ms_path} not created")
    print(f"Saved {ms_path} ({ms_path.stat().st_size // 1024} KB)")
    return ms_path


def event_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """P(left or right second-tap) = max of class-1/2 softmax."""
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 1:].max(dim=-1).values


def build_full_stream_true_labels(
    n_samples: int,
    trigger_events: list[tuple[int, int]],
    half_frames: int,
) -> np.ndarray:
    """Per-frame event weight 1 - P(none) from 3-class Gaussian labels."""
    labels = three_class_frame_labels(n_samples, trigger_events, half_frames=half_frames)
    return 1.0 - labels[:, 0]


def match_triggers(
    true_frames: list[int],
    pred_frames: list[int],
    tolerance: int,
) -> tuple[int, int, int]:
    """Return (matched, missed true, false positives) within frame tolerance."""
    matched = 0
    used_pred: set[int] = set()
    for true_frame in true_frames:
        best_idx = None
        best_dist = tolerance + 1
        for idx, pred_frame in enumerate(pred_frames):
            if idx in used_pred:
                continue
            dist = abs(pred_frame - true_frame)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            matched += 1
            used_pred.add(best_idx)
    missed = len(true_frames) - matched
    false_pos = len(pred_frames) - len(used_pred)
    return matched, missed, false_pos


def predict_recording_probs_causal(
    imu: np.ndarray,
    model: CausalCNNGRU,
    device: str = "cpu",
) -> np.ndarray:
    """Per-frame probability from one causal forward pass (GRU state carried, no pooling)."""
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(imu.astype(np.float32)).unsqueeze(0).to(device)
        logits, _ = model(xt)
        return event_prob_from_logits(logits).cpu().numpy()[0]


def predict_recording_probs_windowed(
    imu: np.ndarray,
    model: CausalCNNGRU,
    window_samples: int = 64,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Per-frame probability envelope using isolated training-style windows.

    Takes the MAX over all overlapping 64-frame windows. This matches window-level
    val metrics but inflates the displayed baseline (~0.15–0.2 on negatives) because
    any window alignment that briefly raises P(t) will dominate. Prefer
    predict_recording_probs_causal for full-recording plots.
    """
    total = len(imu)
    probs = np.zeros(total, dtype=np.float32)
    if total < window_samples:
        return probs

    starts = list(range(0, total - window_samples + 1))
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            windows = np.stack(
                [imu[s : s + window_samples] for s in batch_starts],
                axis=0,
            )
            xt = torch.from_numpy(windows).to(device)
            logits, _ = model(xt)
            batch_probs = event_prob_from_logits(logits).cpu().numpy()
            for start, window_probs in zip(batch_starts, batch_probs):
                for j, value in enumerate(window_probs):
                    idx = start + j
                    if value > probs[idx]:
                        probs[idx] = value
    return probs


@torch.no_grad()
def evaluate_file_windows_from_dataset(
    ds: RecordedIMUDataset,
    source_file: str,
    model: CausalCNNGRU,
    threshold: float,
    device: str = "cpu",
) -> dict[str, int | float | list[int]]:
    """Evaluate pre-built training/val windows belonging to one source file."""
    correct = 0
    tp = fp = fn = 0
    detected_triggers: list[int] = []
    n_windows = 0

    for idx, meta in enumerate(ds.window_meta):
        if meta.source_file != source_file:
            continue

        n_windows += 1
        batch = ds[idx]
        imu = batch["imu"].unsqueeze(0).to(device)
        logits, _ = model(imu)
        probs = event_prob_from_logits(logits).cpu().numpy()[0]
        window_pred = bool(probs.max() > threshold)
        window_label = int(batch["window_label"].item() > 0)

        correct += int(window_pred == window_label)

        if window_label == 1:
            if window_pred:
                tp += 1
                if meta.trigger_frame is not None:
                    detected_triggers.append(meta.trigger_frame)
            else:
                fn += 1
        elif window_pred:
            fp += 1

    if n_windows == 0:
        return {
            "window_acc": 0.0,
            "n_windows": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "detected_triggers": [],
            "false_positive_windows": 0,
        }

    return {
        "window_acc": correct / n_windows,
        "n_windows": n_windows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": n_windows - tp - fp - fn,
        "detected_triggers": sorted(set(detected_triggers)),
        "false_positive_windows": fp,
    }


def plot_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    title: str,
    out_path: Path,
    threshold: float,
) -> None:
    """Plot window-level confusion matrix (rows=true, cols=predicted)."""
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred negative", "Pred double-tap"])
    ax.set_yticklabels(["True negative", "True double-tap"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        row_total = cm[i].sum()
        for j in range(2):
            count = cm[i, j]
            if row_total == 0:
                rate_label = "n/a"
            elif i == j:
                rate = 100.0 * count / row_total
                rate_label = f"recall {rate:.1f}%"
            else:
                rate = 100.0 * count / row_total
                rate_label = f"error {rate:.1f}%"
            color = "white" if count > cm.max() * 0.55 else "black"
            ax.text(
                j, i, f"{count}\n({rate_label})",
                ha="center", va="center", color=color, fontsize=10,
            )

    acc = (tp + tn) / total if total else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0.0
    ax.set_title(
        f"{title}\nthreshold={threshold}, acc={acc:.3f}, "
        f"neg recall={neg_recall:.3f}, double-tap recall={pos_recall:.3f}, n={total}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved {out_path}")


def _sum_confusion(metrics_list: list[dict]) -> tuple[int, int, int, int]:
    tp = sum(int(m.get("tp", 0)) for m in metrics_list)
    fp = sum(int(m.get("fp", 0)) for m in metrics_list)
    fn = sum(int(m.get("fn", 0)) for m in metrics_list)
    tn = sum(int(m.get("tn", 0)) for m in metrics_list)
    return tp, fp, fn, tn


def _first_example_window(
    dataset: RecordedIMUDataset,
    source_file: str,
    prefer_positive: bool = True,
) -> tuple[int, dict] | None:
    """Pick one training window from the dataset for detail plotting."""
    positive: tuple[int, dict] | None = None
    negative: tuple[int, dict] | None = None
    for idx, meta in enumerate(dataset.window_meta):
        if meta.source_file != source_file:
            continue
        batch = dataset[idx]
        item = {"meta": meta, "batch": batch, "idx": idx}
        if int(batch["window_label"].item()) > 0 and positive is None:
            positive = (idx, item)
        elif int(batch["window_label"].item()) == 0 and negative is None:
            negative = (idx, item)
    if prefer_positive and positive is not None:
        return positive
    return positive or negative


def _window_bounds(trigger_frame: int, n_samples: int, ctx: int, win: int) -> tuple[int, int]:
    start = trigger_frame - ctx
    end = start + win
    if start < 0:
        start = 0
        end = win
    if end > n_samples:
        end = n_samples
        start = max(0, end - win)
    return start, end


@torch.no_grad()
def _window_model_output(
    model: CausalCNNGRU,
    imu_window: np.ndarray,
    device: str,
) -> np.ndarray:
    xt = torch.from_numpy(imu_window).unsqueeze(0).to(device)
    logits, _ = model(xt)
    return event_prob_from_logits(logits).cpu().numpy()[0]


def plot_recorded_file(
    csv_path: Path,
    model: CausalCNNGRU,
    cfg: TrainConfig,
    dataset: RecordedIMUDataset,
    out_path: Path,
    device: str = "cpu",
    zoom_half_sec: float = 1.5,
) -> dict[str, int | float | str]:
    """Visualize input time series, training labels, predictions, and a window detail."""
    imu_raw, sample_rate = load_full_recording(csv_path)
    label_path = csv_path.with_suffix(".txt")
    trigger_events = load_trigger_events(label_path)
    trigger_frames = [frame for frame, _cls in trigger_events]
    n = len(imu_raw)
    t_sec = np.arange(n) / sample_rate

    sim = IMUSimulator(sample_rate=sample_rate, window_samples=cfg.data.window_samples)
    imu_hp = sim.highpass(imu_raw) if cfg.data.highpass else imu_raw.astype(np.float32)

    threshold = cfg.training.prediction_threshold
    window_stats = evaluate_file_windows_from_dataset(
        dataset, csv_path.name, model, threshold, device=device
    )
    probs = predict_recording_probs_causal(imu_hp, model, device=device)
    pred_frames = window_stats["detected_triggers"]

    true_labels = build_full_stream_true_labels(n, trigger_events, cfg.labels.half_frames)

    t1_frames: list[int | None] = []
    valid_triggers: list[int] = []
    for tf in trigger_frames:
        t1 = estimate_first_tap_frame(imu_hp, tf)
        t1_frames.append(t1)
        start, end = _window_bounds(tf, n, cfg.data.trigger_context_samples, cfg.data.window_samples)
        t2_in = tf - start
        if t1 is not None:
            dt = (t2_in - t1) / sample_rate
            if cfg.data.dt_min <= dt <= cfg.data.dt_max:
                valid_triggers.append(tf)

    acc_mag = np.linalg.norm(imu_raw[:, :3], axis=1)
    gyro_mag = np.linalg.norm(imu_raw[:, 3:], axis=1)
    acc_hp_mag = np.linalg.norm(imu_hp[:, :3], axis=1)
    gyro_hp_mag = np.linalg.norm(imu_hp[:, 3:], axis=1)

    example = _first_example_window(dataset, csv_path.name)
    zoom_center = example[1]["meta"].trigger_frame if example and example[1]["meta"].trigger_frame else (
        trigger_frames[0] if trigger_frames else n // 2
    )
    z0 = max(0, int(zoom_center - zoom_half_sec * sample_rate))
    z1 = min(n, int(zoom_center + zoom_half_sec * sample_rate))
    sl = slice(z0, z1)

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1.1, 1.4], width_ratios=[1.6, 1], hspace=0.38, wspace=0.28)

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.plot(t_sec, acc_mag, color="#3366cc", lw=0.7, label="|a| raw")
    ax_raw.plot(t_sec, gyro_mag, color="#cc6633", lw=0.7, alpha=0.8, label="|ω| raw")
    _mark_triggers(ax_raw, trigger_frames, pred_frames, t1_frames, t_sec)
    ax_raw.set_ylabel("amplitude")
    ax_raw.set_title("A. Raw input time series (full recording)")
    ax_raw.legend(loc="upper right", fontsize=7)

    ax_hp = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_hp.plot(t_sec, acc_hp_mag, color="#3366cc", lw=0.7, label="|a| HP")
    ax_hp.plot(t_sec, gyro_hp_mag, color="#cc6633", lw=0.7, alpha=0.8, label="|ω| HP")
    _mark_triggers(ax_hp, trigger_frames, pred_frames, t1_frames, t_sec)
    ax_hp.set_ylabel("amplitude")
    ax_hp.set_title("B. Preprocessed input (high-pass, training/inference)")
    ax_hp.legend(loc="upper right", fontsize=7)

    ax_lbl = fig.add_subplot(gs[2, 0], sharex=ax_raw)
    ax_lbl.fill_between(t_sec, 0, true_labels, color="#cc3333", alpha=0.35, label="Gaussian yₙ")
    for tf in trigger_frames:
        ax_lbl.axvline(t_sec[tf], color="#cc3333", ls="--", alpha=0.5, lw=0.8)
    ax_lbl.set_ylim(-0.05, 1.05)
    ax_lbl.set_ylabel("label")
    ax_lbl.set_title("C. Training labels (from label_imu triggers → Gaussian)")
    ax_lbl.legend(loc="upper right", fontsize=7)

    ax_pred = fig.add_subplot(gs[3, 0], sharex=ax_raw)
    ax_pred.plot(t_sec, probs, color="#3366cc", lw=1.0, label="P(double-tap)")
    ax_pred.axhline(threshold, color="gray", ls="--", alpha=0.5)
    _mark_triggers(ax_pred, trigger_frames, pred_frames, None, t_sec)
    ax_pred.set_ylim(-0.05, 1.05)
    ax_pred.set_ylabel("probability")
    ax_pred.set_xlabel("time (s)")
    ax_pred.set_title(f"D. Model output — causal stream (threshold={threshold})")
    ax_pred.legend(loc="upper right", fontsize=7)

    ax_zoom = fig.add_subplot(gs[4, 0])
    tz = t_sec[sl]
    for ch in range(6):
        ax_zoom.plot(tz, imu_hp[sl, ch], color=IMU_COLORS[ch], lw=0.9, alpha=0.85, label=IMU_NAMES[ch])
    for f in trigger_frames:
        if z0 <= f < z1:
            ax_zoom.axvline(t_sec[f], color="#cc3333", ls="--", alpha=0.55, lw=0.9)
    for f in pred_frames:
        if z0 <= f < z1:
            ax_zoom.axvline(t_sec[f], color="#33aa33", ls="-", alpha=0.75, lw=1.0)
    ax_zoom.set_xlabel("time (s)")
    ax_zoom.set_ylabel("HP amplitude")
    ax_zoom.set_title(f"E. Zoomed input ({zoom_half_sec:.1f}s around example) — 6 channels")
    ax_zoom.legend(loc="upper right", fontsize=6, ncol=2)

    # --- Right column: training window detail ---
    ax_win_in = fig.add_subplot(gs[0:2, 1])
    ax_win_lbl = fig.add_subplot(gs[2, 1])
    ax_win_pred = fig.add_subplot(gs[3, 1])
    ax_win_txt = fig.add_subplot(gs[4, 1])
    ax_win_txt.axis("off")

    if example is not None:
        ex_idx, ex = example
        batch = ex["batch"]
        meta = ex["meta"]
        imu_w = batch["imu"].numpy()
        labels_w = batch["frame_labels"].numpy()
        win_label = int(batch["window_label"].item())
        tw = np.arange(len(imu_w)) / sample_rate
        win_probs = _window_model_output(model, imu_w, device)

        offset = 0.6
        for ch in range(6):
            ax_win_in.plot(tw, imu_w[:, ch] + ch * offset, color=IMU_COLORS[ch], lw=1.2)
        ax_win_in.set_yticks([ch * offset for ch in range(6)])
        ax_win_in.set_yticklabels(IMU_NAMES, fontsize=7)
        ax_win_in.set_title(f"Training window #{ex_idx} — model input (HP, stacked)")
        ax_win_in.set_xlabel("time in window (s)")

        if labels_w.ndim == 2:
            labels_w = 1.0 - labels_w[:, 0]
        ax_win_lbl.fill_between(tw, 0, labels_w, color="#cc3333", alpha=0.4)
        ax_win_lbl.plot(tw, labels_w, color="#cc3333", lw=1.5)
        ax_win_lbl.set_ylim(-0.05, 1.05)
        ax_win_lbl.set_ylabel("yₙ")
        ax_win_lbl.set_title("Frame labels (Gaussian soft targets)")
        ax_win_lbl.grid(True, alpha=0.2)

        ax_win_pred.plot(tw, win_probs, color="#3366cc", lw=1.5, label="P(tap)")
        ax_win_pred.plot(tw, labels_w, color="#cc3333", ls="--", alpha=0.6, label="yₙ")
        ax_win_pred.axhline(threshold, color="gray", ls=":", alpha=0.6)
        ax_win_pred.set_ylim(-0.05, 1.05)
        ax_win_pred.set_ylabel("prob")
        ax_win_pred.set_xlabel("time in window (s)")
        ax_win_pred.set_title(f"Window prediction (label={'pos' if win_label else 'neg'}, max P={win_probs.max():.2f})")
        ax_win_pred.legend(fontsize=7)

        detail_lines = [
            f"Window length: {cfg.data.window_samples} frames @ {sample_rate:.0f} Hz",
            f"Window class: {'positive (double-tap)' if win_label else 'negative'}",
        ]
        if meta.trigger_frame is not None:
            detail_lines.append(f"Trigger n₂ = frame {meta.trigger_frame} (global)")
            ws, _ = _window_bounds(meta.trigger_frame, n, cfg.data.trigger_context_samples, cfg.data.window_samples)
            t2_local = meta.trigger_frame - ws
            t1_local = estimate_first_tap_frame(imu_hp[ws:ws + cfg.data.window_samples], t2_local)
            if t1_local is not None:
                dt = (t2_local - t1_local) / sample_rate
                detail_lines.append(f"Est. n₁ = frame {ws + t1_local}, Δt = {dt:.3f} s")
        detail_lines.append(f"Δt filter: [{cfg.data.dt_min}, {cfg.data.dt_max}] s")
        detail_lines.append(f"Valid triggers kept: {len(valid_triggers)}/{len(trigger_frames)}")
        ax_win_txt.text(0.02, 0.95, "\n".join(detail_lines), va="top", fontsize=8, family="monospace",
                        transform=ax_win_txt.transAxes, bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))
    else:
        ax_win_in.text(0.5, 0.5, "No training windows\nfor this file", ha="center", va="center")
        ax_win_in.set_title("Training window detail")

    matched, missed, _ = match_triggers(trigger_frames, pred_frames, cfg.data.trigger_context_samples)
    false_pos = int(window_stats["fp"])
    window_acc = float(window_stats["window_acc"])
    summary = (
        f"window_acc={window_acc:.3f} ({window_stats['n_windows']} windows) | "
        f"triggers: {len(trigger_frames)} labeled, {len(valid_triggers)} Δt-valid, "
        f"{len(pred_frames)} detected, matched={matched}, missed={missed}, fp={false_pos}"
    )
    fig.suptitle(f"{csv_path.name}\n{summary}", fontsize=11, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()

    return {
        "file": csv_path.name,
        "window_acc": window_acc,
        "n_windows": int(window_stats["n_windows"]),
        "true_triggers": len(trigger_frames),
        "pred_triggers": len(pred_frames),
        "matched": matched,
        "missed": missed,
        "false_positives": false_pos,
    }


def run_recorded_comparison(
    checkpoint: Path,
    cfg: TrainConfig,
    data_dirs: list[Path],
    out_dir: Path,
    device: str = "cpu",
) -> list[dict[str, int | float | str]]:
    """Plot every CSV in data_dirs and compare predictions to label_imu sidecars."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CausalCNNGRU(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if "val_metrics" in ckpt:
        stored = ckpt["val_metrics"]
        print(
            f"Checkpoint val metrics: window_acc={stored['window_acc']:.3f} "
            f"frame_acc={stored['frame_acc']:.3f}"
        )

    from torch.utils.data import DataLoader

    label_params = cfg.labels.to_label_params()
    data_kwargs = dict(
        window_samples=cfg.data.window_samples,
        label_params=label_params,
        highpass=cfg.data.highpass,
        negative_windows_per_file=cfg.data.negative_windows_per_file,
        trigger_context_samples=cfg.data.trigger_context_samples,
        exclusion_margin=cfg.data.exclusion_margin,
        dt_min=cfg.data.dt_min,
        dt_max=cfg.data.dt_max,
    )
    train_ds = RecordedIMUDataset(cfg.data.train_dir, seed=cfg.seed, **data_kwargs)
    val_ds = RecordedIMUDataset(cfg.data.val_dir, seed=cfg.seed + 1, **data_kwargs)
    datasets_by_dir = {
        Path(cfg.data.train_dir): train_ds,
        Path(cfg.data.val_dir): val_ds,
    }

    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False)
    threshold = cfg.training.prediction_threshold
    device_t = torch.device(device)

    val_metrics = evaluate(model, val_loader, device_t, threshold)
    train_metrics = evaluate(model, train_loader, device_t, threshold)
    print(
        f"Recomputed val metrics (training windows): "
        f"window_acc={val_metrics['window_acc']:.3f} "
        f"frame_acc={val_metrics['frame_acc']:.3f} "
        f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f}"
    )
    print(
        f"Recomputed train metrics: "
        f"window_acc={train_metrics['window_acc']:.3f} "
        f"P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f}"
    )

    plot_confusion_matrix(
        val_metrics["tp"], val_metrics["fp"], val_metrics["fn"], val_metrics["tn"],
        "Validation set — window confusion matrix",
        out_dir / "confusion_matrix_val.png",
        threshold,
    )
    plot_confusion_matrix(
        train_metrics["tp"], train_metrics["fp"], train_metrics["fn"], train_metrics["tn"],
        "Training set — window confusion matrix",
        out_dir / "confusion_matrix_train.png",
        threshold,
    )

    results: list[dict[str, int | float | str]] = []

    csv_files: list[Path] = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"Skipping missing directory: {data_dir}")
            continue
        csv_files.extend(sorted(data_dir.glob("*.csv")))

    if not csv_files:
        print("No CSV recordings found for comparison demo.")
        return results

    print(f"\nComparing predictions vs label_imu for {len(csv_files)} files...")
    for csv_path in csv_files:
        out_path = out_dir / f"{csv_path.parent.name}_{csv_path.stem}.png"
        dataset = datasets_by_dir[csv_path.parent]
        stats = plot_recorded_file(
            csv_path,
            model,
            cfg,
            dataset,
            out_path,
            device=device,
        )
        results.append(stats)
        print(
            f"  {stats['file']}: window_acc={stats['window_acc']:.3f} "
            f"({stats['n_windows']} windows) triggers={stats['true_triggers']}/"
            f"{stats['pred_triggers']} matched={stats['matched']} "
            f"missed={stats['missed']} fp_windows={stats['false_positives']} "
            f"→ {out_path.name}"
        )

    file_metrics = [
        evaluate_file_windows_from_dataset(
            datasets_by_dir[csv_path.parent], csv_path.name, model, threshold, device
        )
        for csv_path in csv_files
    ]
    all_tp, all_fp, all_fn, all_tn = _sum_confusion(file_metrics)
    plot_confusion_matrix(
        all_tp, all_fp, all_fn, all_tn,
        "All recordings (per-file windows aggregated)",
        out_dir / "confusion_matrix_all.png",
        threshold,
    )

    total_windows = sum(int(r["n_windows"]) for r in results)
    weighted_acc = sum(
        float(r["window_acc"]) * int(r["n_windows"]) for r in results
    ) / max(total_windows, 1)
    total_matched = sum(int(r["matched"]) for r in results)
    total_missed = sum(int(r["missed"]) for r in results)
    total_fp = sum(int(r["false_positives"]) for r in results)
    print(
        f"\nOverall file window_acc (weighted): {weighted_acc:.3f} | "
        f"trigger matched={total_matched} missed={total_missed} fp_windows={total_fp}"
    )
    print(
        f"Confusion (all windows): TN={all_tn} FP={all_fp} FN={all_fn} TP={all_tp}"
    )
    return results


def run_online_demo(checkpoint: Path) -> None:
    """Sample-by-sample demo: imu [1,1,6] + hidden states -> prob [1,1,1]."""
    from tap_recognition.inference import DetectorConfig, OnlineDoubleTapDetector

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CausalCNNGRU(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg = ckpt["model_config"]
    cnn_channels = cfg.get("cnn_channels", 32)
    gru_hidden = cfg.get("gru_hidden", 64)
    gru_layers = cfg.get("gru_layers", 1)
    receptive_field = model.receptive_field

    det_cfg = DetectorConfig()
    detector = OnlineDoubleTapDetector(model, config=det_cfg)

    sim = IMUSimulator(sample_rate=100, window_samples=100, seed=123)
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.55, 1.0, 1.0, 1.0], hspace=0.35)
    ax_info = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i]) for i in range(1, 4)]
    test_cases = ["double", "single", "background"]

    ax_info.axis("off")
    io_text = (
        "Streaming step I/O (each new IMU sample):\n"
        f"  in:  imu [1, 1, 6]  +  cnn_buffer [1, {cnn_channels}, {receptive_field}]  "
        f"+  h [{gru_layers}, 1, {gru_hidden}]\n"
        "  out: prob [1, 1, 1]  +  updated cnn_buffer  +  updated h\n"
        "  post: hysteresis on prob scalar -> trigger event"
    )
    ax_info.text(
        0.02, 0.5, io_text, va="center", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999"),
    )

    for ax, kind in zip(axes, test_cases):
        imu, labels, meta = sim.generate_window(kind=kind)
        imu_hp = sim.highpass(imu)

        detector.reset()
        probs = np.zeros(len(imu), dtype=np.float32)
        h_norms = np.zeros(len(imu), dtype=np.float32)
        buf_norms = np.zeros(len(imu), dtype=np.float32)

        for i, sample in enumerate(imu_hp):
            probs[i], _ = detector.process_sample(sample)
            if detector._h is not None:
                h_norms[i] = float(detector._h.norm().item())
            if detector._cnn_buffer is not None:
                buf_norms[i] = float(detector._cnn_buffer.norm().item())

        events = list(detector.events)
        t = np.arange(len(imu)) / sim.sample_rate

        ax.plot(t, np.linalg.norm(imu_hp[:, :3], axis=1), label="|a| norm", color="#1f77b4")
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, labels, alpha=0.12, color="blue", label="ground truth")
        ax2.plot(t, probs, "r-", linewidth=2, label="prob [1,1,1]")
        ax2.plot(t, h_norms / max(h_norms.max(), 1e-6), "--", color="#888", alpha=0.7, label="||h|| (normed)")
        ax2.plot(t, buf_norms / max(buf_norms.max(), 1e-6), ":", color="#666", alpha=0.7, label="||cnn_buf|| (normed)")
        ax2.axhline(det_cfg.threshold_on, color="gray", linestyle="--", alpha=0.4)
        for ev in events:
            ax2.axvline(ev.time_sec, color="green", linewidth=1.5, linestyle=":")
        ax.set_title(f"{kind}: gt={meta.label}, detected={len(events)}")
        ax.set_ylabel("|a|")
        ax2.set_ylabel("prob / state norm")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Online Inference — one sample per step", fontsize=12, y=0.98)
    plt.savefig("online_demo.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved online_demo.png")


def verify_causality() -> None:
    """Ensure output at t does not change when future inputs are modified."""
    model = CausalCNNGRU()
    model.eval()
    x = torch.randn(1, 32, 6)
    with torch.no_grad():
        logits_full, _ = model(x)
        p_t = torch.softmax(logits_full[0, 15], dim=-1)[0].item()

        x_modified = x.clone()
        x_modified[0, 16:, :] = torch.randn(16, 6) * 100
        logits_mod, _ = model(x_modified)
        p_t_mod = torch.softmax(logits_mod[0, 15], dim=-1)[0].item()

    assert abs(p_t - p_t_mod) < 1e-5, "Causality violated!"
    print(f"Causality check passed (p@t={p_t:.6f}, unchanged after future perturbation)")


def main() -> None:
    parser = argparse.ArgumentParser(description="IMU double-tap recognition demo")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--recorded-out-dir",
        default="demo_outputs",
        help="Directory for per-file prediction vs label_imu plots",
    )
    parser.add_argument(
        "--skip-export-ms",
        action="store_true",
        help="Skip PyTorch checkpoint -> MindSpore Lite (.ms) export",
    )
    parser.add_argument(
        "--onnx-output",
        default="checkpoints/tap_recognition.onnx",
        help="ONNX intermediate path for MindSpore conversion",
    )
    parser.add_argument(
        "--ms-output",
        default="checkpoints/tap_recognition",
        help="MindSpore Lite output prefix (produces <prefix>.ms)",
    )
    parser.add_argument(
        "--ms-lite-root",
        default=r"D:\mindspore-lite-2.9.0-win-x64",
        help="Extracted MindSpore Lite Windows SDK root",
    )
    parser.add_argument(
        "--mingw-bin",
        default="",
        help="MinGW bin dir for converter DLLs (default: auto-detect)",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    data_dirs = [Path(cfg.data.train_dir), Path(cfg.data.val_dir)]

    print("=" * 60, flush=True)
    print("IMU Double-Tap Recognition — Full Demo", flush=True)
    print("=" * 60, flush=True)

    if not args.skip_synthetic:
        plot_physics_model()
        plot_architecture()
        verify_causality()

    plot_detailed_pipeline(cfg, "pipeline_detailed.png")

    ckpt = Path(args.checkpoint)
    if not args.skip_train:
        ckpt = run_training(cfg, force=args.force_train)
    elif not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Train first or pass --checkpoint."
        )

    run_recorded_comparison(
        ckpt,
        cfg,
        data_dirs,
        Path(args.recorded_out_dir),
    )

    if not args.skip_synthetic:
        run_online_demo(ckpt)

    ms_path: Path | None = None
    if not args.skip_export_ms:
        mingw_bin = args.mingw_bin
        if not mingw_bin:
            bundled = Path("tools/mingw730/mingw64/bin")
            if (bundled / "libgcc_s_seh-1.dll").exists():
                mingw_bin = str(bundled)
        ms_path = export_to_mindspore(
            ckpt,
            cfg,
            onnx_path=Path(args.onnx_output),
            ms_output=Path(args.ms_output),
            ms_lite_root=args.ms_lite_root,
            mingw_bin=mingw_bin,
        )

    print("\nDemo complete. Outputs:")
    print("  docs/mathematical_model.md  — full mathematical model")
    if not args.skip_synthetic:
        print("  physics_model.png           — synthetic IMU signals")
        print("  architecture.png            — network schematic")
        print("  online_demo.png             — streaming inference on synthetic data")
    print("  pipeline_detailed.png       — full data/train/deploy pipeline")
    print(f"  {args.recorded_out_dir}/              — per-file input/label/prediction plots")
    print(f"  {args.recorded_out_dir}/confusion_matrix_*.png — window confusion matrices")
    print(f"  {args.checkpoint}         — trained weights")
    if ms_path is not None:
        print(f"  {args.onnx_output}       — ONNX for MindSpore Lite")
        print(f"  {ms_path}     — MindSpore Lite model for HarmonyOS")


if __name__ == "__main__":
    main()
