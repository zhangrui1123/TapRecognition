"""Training script for causal CNN + GRU/LSTM double-tap detectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from tap_recognition.config import (
    EarlyStoppingConfig,
    LSTMModelConfig,
    ModelConfig,
    TrainConfig,
)
from tap_recognition.dataset import IMUDoubleTapDataset, RecordedIMUDataset
from tap_recognition.model_factory import build_model
from tap_recognition.model_lstm import CausalCNNLSTM


def freeze_except_last_layer(model: nn.Module) -> int:
    """Freeze all weights except the final Linear in ``model.head``."""
    for param in model.parameters():
        param.requires_grad = False
    last = model.head[-1]
    for param in last.parameters():
        param.requires_grad = True
    return sum(p.numel() for p in last.parameters())


def _augment_positive_windows(
    imu: torch.Tensor,
    window_label: torch.Tensor,
) -> torch.Tensor:
    """Randomly scale positive windows so lighter taps still train."""
    pos = window_label > 0
    if not bool(pos.any()):
        return imu
    scale = imu.new_empty(imu.size(0), 1, 1).uniform_(0.70, 1.10)
    scale = torch.where(pos.view(-1, 1, 1), scale, torch.ones_like(scale))
    return imu * scale


def frame_soft_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    event_weight: float = 10.0,
    window_label: torch.Tensor | None = None,
    neg_window_weight: float = 2.0,
) -> torch.Tensor:
    """Soft CE with extra weight on event frames and negative windows."""
    logp = F.log_softmax(logits, dim=-1)
    per_frame = -(labels * logp).sum(dim=-1)
    weights = 1.0 + event_weight * (1.0 - labels[..., 0])
    per_sample = (per_frame * weights).mean(dim=-1)
    if window_label is not None:
        scale = torch.where(
            window_label == 0,
            per_sample.new_tensor(neg_window_weight),
            per_sample.new_tensor(1.0),
        )
        return (per_sample * scale).mean()
    return per_sample.mean()


def window_class_from_probs(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """probs [B, T, 3] -> predicted window class 0/1/2."""
    event = probs[..., 1:]
    event_t = event.max(dim=-1).values
    t_star = event_t.argmax(dim=1)
    batch = torch.arange(probs.size(0), device=probs.device)
    cls = event[batch, t_star].argmax(dim=-1) + 1
    fired = event_t.max(dim=1).values > threshold
    return torch.where(fired, cls, torch.zeros_like(cls))


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _split_dirs(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _recorded_dataset(
    dirs: list[str],
    cfg: TrainConfig,
    seed: int,
) -> Dataset:
    data = cfg.data
    labels = cfg.labels.to_label_params()
    parts: list[Dataset] = []
    for i, data_dir in enumerate(dirs):
        print(f"  {data_dir}")
        parts.append(
            RecordedIMUDataset(
                data_dir,
                window_samples=data.window_samples,
                label_params=labels,
                highpass=data.highpass,
                negative_windows_per_file=data.negative_windows_per_file,
                trigger_context_samples=data.trigger_context_samples,
                exclusion_margin=data.exclusion_margin,
                dt_min=data.dt_min,
                dt_max=data.dt_max,
                seed=seed + i,
                session_exclusions_file=data.session_exclusions_file,
            )
        )
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def _count_positive_windows(ds: Dataset) -> int:
    if hasattr(ds, "samples"):
        return sum(1 for _w, _y, cls in ds.samples if cls > 0)
    return sum(1 for item in ds if int(item["window_label"]) > 0)


def build_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    data = cfg.data

    if data.mode == "recorded":
        print("Train dirs:")
        train_ds = _recorded_dataset(_split_dirs(data.train_dir), cfg, cfg.seed)
        print("Val dirs:")
        val_ds = _recorded_dataset(_split_dirs(data.val_dir), cfg, cfg.seed + 1)
        val_pos = _count_positive_windows(val_ds)
        if val_pos == 0:
            n_val = max(1, int(data.val_split * len(train_ds)))
            n_train = len(train_ds) - n_val
            print(
                f"Validation dir has no positive windows; "
                f"splitting train windows {n_train}/{n_val}"
            )
            return random_split(
                train_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(cfg.seed),
            )
        return train_ds, val_ds

    full_ds = IMUDoubleTapDataset(
        num_samples=data.synthetic_samples,
        window_samples=data.window_samples,
        sample_rate=data.sample_rate,
        seed=cfg.seed,
        highpass=data.highpass,
    )
    n_val = max(1, int(data.val_split * len(full_ds)))
    n_train = len(full_ds) - n_val
    return random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    threshold: float,
    event_weight: float,
    freeze_except_last: bool = False,
    neg_window_weight: float = 2.0,
) -> dict[str, float]:
    if freeze_except_last:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    correct_windows = 0
    total_windows = 0

    for batch in loader:
        imu = batch["imu"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        window_label = batch["window_label"].to(device)
        imu = _augment_positive_windows(imu, window_label)

        logits, _ = model(imu)
        loss = frame_soft_ce(
            logits,
            frame_labels,
            event_weight,
            window_label=window_label,
            neg_window_weight=neg_window_weight,
        )
        probs = torch.softmax(logits, dim=-1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * imu.size(0)
        pred = probs.argmax(dim=-1)
        true_cls = frame_labels.argmax(dim=-1)
        correct_frames += (pred == true_cls).sum().item()
        total_frames += true_cls.numel()

        window_pred = window_class_from_probs(probs, threshold)
        window_true = batch["window_label"].to(device)
        correct_windows += (window_pred == window_true).sum().item()
        total_windows += imu.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "frame_acc": correct_frames / total_frames,
        "window_acc": correct_windows / total_windows,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    event_weight: float = 10.0,
    neg_window_weight: float = 2.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    correct_windows = 0
    total_windows = 0
    tp = fp = fn = tn = 0

    for batch in loader:
        imu = batch["imu"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        window_labels = batch["window_label"].to(device)

        logits, _ = model(imu)
        probs = torch.softmax(logits, dim=-1)
        loss = frame_soft_ce(
            logits,
            frame_labels,
            event_weight,
            window_label=window_labels,
            neg_window_weight=neg_window_weight,
        )
        total_loss += loss.item() * imu.size(0)

        pred = probs.argmax(dim=-1)
        true_cls = frame_labels.argmax(dim=-1)
        correct_frames += (pred == true_cls).sum().item()
        total_frames += true_cls.numel()

        window_pred = window_class_from_probs(probs, threshold)
        correct_windows += (window_pred == window_labels).sum().item()
        total_windows += imu.size(0)

        pos_pred = (window_pred > 0).long()
        pos_true = (window_labels > 0).long()
        for wp, wl in zip(pos_pred, pos_true):
            if wp == 1 and wl == 1:
                tp += 1
            elif wp == 1 and wl == 0:
                fp += 1
            elif wp == 0 and wl == 1:
                fn += 1
            else:
                tn += 1

    n = len(loader.dataset)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "loss": total_loss / n,
        "frame_acc": correct_frames / total_frames,
        "window_acc": correct_windows / total_windows,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class EarlyStopping:
    """Stop training when the monitored validation metric stops improving."""

    def __init__(self, config: EarlyStoppingConfig):
        self.enabled = config.enabled
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor
        if self.monitor not in {"val_window_acc", "val_loss"}:
            raise ValueError(
                f"Unsupported early_stopping.monitor: {self.monitor!r}"
            )
        self.mode = "max" if self.monitor == "val_window_acc" else "min"
        self.metric_key = "window_acc" if self.monitor == "val_window_acc" else "loss"
        self.best: float | None = None
        self.counter = 0

    def step(self, metrics: dict[str, float]) -> bool:
        if not self.enabled:
            return False

        current = metrics[self.metric_key]
        if self.best is None:
            self.best = current
            return False

        if self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train(cfg: TrainConfig) -> float:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    print(f"Loading {cfg.data.mode} datasets...")
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model_kwargs = cfg.model.to_model_kwargs()
    if cfg.init_checkpoint:
        ckpt = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
        loaded = dict(ckpt.get("model_config", model_kwargs))
        if "dilations" in loaded:
            loaded["dilations"] = tuple(loaded["dilations"])
        model_kwargs = loaded
        model = build_model(model_kwargs, ckpt.get("model_type")).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Initialized weights from {cfg.init_checkpoint}")
    else:
        model = build_model(model_kwargs).to(device)

    model_type = "lstm" if isinstance(model, CausalCNNLSTM) else "gru"
    config_class = LSTMModelConfig if model_type == "lstm" else ModelConfig
    cfg.model = config_class(**model_kwargs)

    if cfg.freeze_except_last:
        n_last = freeze_except_last_layer(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"Frozen backbone; training last layer only ({n_last} params)")
    else:
        trainable = list(model.parameters())

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    early_stop = EarlyStopping(cfg.training.early_stopping)
    stopped_epoch = cfg.training.epochs

    es = cfg.training.early_stopping
    n_train_pos = sum(
        1 for item in train_ds if int(item["window_label"]) > 0
    )
    n_val_pos = sum(1 for item in val_ds if int(item["window_label"]) > 0)
    print(
        f"Training on {device}, {len(train_ds)} train "
        f"({n_train_pos} pos / {len(train_ds) - n_train_pos} neg) / "
        f"{len(val_ds)} val ({n_val_pos} pos / {len(val_ds) - n_val_pos} neg)"
    )
    print(
        f"Sequence length: {cfg.data.window_samples} frames "
        f"({cfg.data.window_samples / cfg.data.sample_rate:.2f} s)"
    )
    print(f"Model receptive field: {model.receptive_field} samples")
    if es.enabled:
        print(
            f"Early stopping: monitor={es.monitor}, patience={es.patience}, "
            f"min_delta={es.min_delta}"
        )

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.training.grad_clip,
            cfg.training.prediction_threshold,
            cfg.training.event_loss_weight,
            freeze_except_last=cfg.freeze_except_last,
            neg_window_weight=cfg.training.neg_window_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            cfg.training.prediction_threshold,
            cfg.training.event_loss_weight,
            neg_window_weight=cfg.training.neg_window_weight,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['window_acc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['window_acc']:.3f} "
            f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f}"
        )

        if val_metrics["window_acc"] > best_val_acc:
            best_val_acc = val_metrics["window_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": model_type,
                    "model_config": model_kwargs,
                    "train_config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

        if early_stop.step(val_metrics):
            stopped_epoch = epoch
            print(
                f"Early stopping at epoch {epoch} "
                f"(no {es.monitor} improvement for {es.patience} epochs)"
            )
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": model_type,
            "model_config": model_kwargs,
            "train_config": cfg.to_dict(),
            "epoch": stopped_epoch,
        },
        out_dir / "last.pt",
    )
    print(f"Done. Best val window acc: {best_val_acc:.3f} (stopped at epoch {stopped_epoch})")
    print(f"Checkpoints saved to {out_dir}/")
    return best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or fine-tune the tap detector")
    parser.add_argument("--init-checkpoint", default="", help="Load weights before training")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--recurrent-type", choices=("gru", "lstm"), default="gru",
        help="Recurrent architecture for training from scratch (default: gru)",
    )
    parser.add_argument(
        "--recurrent-hidden", type=int, default=None,
        help="Hidden size for the selected architecture (default: 64)",
    )
    parser.add_argument(
        "--recurrent-layers", type=int, default=None,
        help="Recurrent layer count (default: 1)",
    )
    parser.add_argument(
        "--freeze-except-last",
        action="store_true",
        help="Train only the final Linear layer",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Training CSV dir, or comma-separated dirs",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Validation CSV dir, or comma-separated dirs",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.recurrent_type == "lstm":
        cfg.model = LSTMModelConfig()
    if args.recurrent_hidden is not None:
        setattr(cfg.model, f"{args.recurrent_type}_hidden", args.recurrent_hidden)
    if args.recurrent_layers is not None:
        setattr(cfg.model, f"{args.recurrent_type}_layers", args.recurrent_layers)
    if args.init_checkpoint:
        cfg.init_checkpoint = args.init_checkpoint
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.train_dir:
        cfg.data.train_dir = args.train_dir
    if args.val_dir:
        cfg.data.val_dir = args.val_dir
    cfg.freeze_except_last = args.freeze_except_last
    train(cfg)


if __name__ == "__main__":
    main()
