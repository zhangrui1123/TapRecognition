"""Construct GRU or LSTM models from training/checkpoint configuration."""

from __future__ import annotations

from typing import Any

from .model import CausalCNNGRU
from .model_lstm import CausalCNNLSTM


def build_model(
    model_config: dict[str, Any], model_type: str | None = None
) -> CausalCNNGRU | CausalCNNLSTM:
    """Keep legacy GRU checkpoints usable without architecture metadata."""
    config = dict(model_config)
    if "dilations" in config:
        config["dilations"] = tuple(config["dilations"])
    if model_type is None:
        model_type = "lstm" if "lstm_hidden" in config or "lstm_layers" in config else "gru"
    if model_type == "gru":
        return CausalCNNGRU(**config)
    if model_type == "lstm":
        return CausalCNNLSTM(**config)
    raise ValueError(f"Unsupported model_type: {model_type!r}")
