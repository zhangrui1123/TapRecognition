"""Training and model hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .recording import LabelParams


@dataclass
class DataConfig:
    mode: str = "recorded"
    train_dir: str = "data/train_data"
    val_dir: str = "data/valid_data"
    window_samples: int = 300
    sample_rate: float = 100.0
    highpass: bool = True
    negative_windows_per_file: int = 20
    trigger_context_samples: int = 200
    exclusion_margin: int = 200
    dt_min: float = 0.12
    dt_max: float = 0.45
    session_exclusions_file: str = "data/session_exclusions.csv"
    synthetic_samples: int = 8000
    val_split: float = 0.15


@dataclass
class LabelConfig:
    sigma: float = 0.04
    peak_offset: float = 0.05
    half_frames: int = 10

    def to_label_params(self) -> LabelParams:
        return LabelParams(
            sigma=self.sigma,
            peak_offset=self.peak_offset,
            half_frames=self.half_frames,
        )


@dataclass
class ModelConfig:
    input_dim: int = 6
    num_classes: int = 3
    cnn_channels: int = 32
    gru_hidden: int = 64
    gru_layers: int = 1
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "cnn_channels": self.cnn_channels,
            "gru_hidden": self.gru_hidden,
            "gru_layers": self.gru_layers,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }


@dataclass
class LSTMModelConfig:
    input_dim: int = 6
    num_classes: int = 3
    cnn_channels: int = 32
    lstm_hidden: int = 64
    lstm_layers: int = 1
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "cnn_channels": self.cnn_channels,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_window_acc"


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    prediction_threshold: float = 0.50
    event_loss_weight: float = 15.0
    neg_window_weight: float = 1.5
    num_workers: int = 0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    out_dir: str = "checkpoints"
    init_checkpoint: str | None = None
    freeze_except_last: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig | LSTMModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"]["dilations"] = list(self.model.dilations)
        return payload
