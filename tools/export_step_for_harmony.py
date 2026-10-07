"""Export TapRecognition checkpoint to step-based ONNX for streaming HarmonyOS inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tap_recognition.model import CausalCNNGRU
from tap_recognition.model_factory import build_model
from tap_recognition.model_lstm import CausalCNNLSTM


class StreamingStep(nn.Module):
    """Single-sample streaming forward: (1,1,6) + states -> softmax + updated states."""

    def __init__(self, model: CausalCNNGRU):
        super().__init__()
        self.model = model

    def forward(
        self,
        imu: torch.Tensor,
        cnn_buffer: torch.Tensor,
        h_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob, h_out, buf_out = self.model.step(imu, h_in, cnn_buffer)
        return prob, h_out, buf_out


class StreamingLSTMStep(nn.Module):
    """One sample with both LSTM states and packed causal CNN context."""

    def __init__(self, model: CausalCNNLSTM):
        super().__init__()
        self.model = model

    def forward(
        self,
        imu: torch.Tensor,
        cnn_buffer: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prob, (h_out, c_out), buf_out = self.model.step(imu, (h_in, c_in), cnn_buffer)
        return prob, h_out, c_out, buf_out


def export_step_onnx(checkpoint_path: Path, output_path: Path, opset: int = 14) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = dict(checkpoint["model_config"])
    model = build_model(cfg, checkpoint.get("model_type"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_dim = cfg.get("input_dim", 6)
    cnn_channels = cfg.get("cnn_channels", 32)
    receptive_field = model.receptive_field

    imu = torch.randn(1, 1, input_dim)
    cnn_buffer = torch.zeros(1, cnn_channels, receptive_field)
    if isinstance(model, CausalCNNLSTM):
        state_shape = (cfg["lstm_layers"], 1, cfg["lstm_hidden"])
        wrapper = StreamingLSTMStep(model).eval()
        inputs = (imu, cnn_buffer, torch.zeros(state_shape), torch.zeros(state_shape))
        input_names = ["imu", "cnn_buffer", "h_in", "c_in"]
        output_names = ["prob", "h_out", "c_out", "cnn_buffer_out"]
    else:
        assert isinstance(model, CausalCNNGRU)
        state_shape = (cfg.get("gru_layers", 1), 1, cfg.get("gru_hidden", 64))
        wrapper = StreamingStep(model).eval()
        inputs = (imu, cnn_buffer, torch.zeros(state_shape))
        input_names = ["imu", "cnn_buffer", "h_in"]
        output_names = ["prob", "h_out", "cnn_buffer_out"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset,
        dynamo=False,  # Keep the requested opset for the MindSpore Lite converter.
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    print(
        f"Streaming I/O: imu [1,1,{input_dim}] + cnn_buffer [1,{cnn_channels},{receptive_field}] "
        f"+ {', '.join(input_names[2:])} {list(state_shape)} -> prob [1,1,{cfg.get('num_classes', 3)}]"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export step ONNX for HarmonyOS streaming")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="checkpoints/tap_step.onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    path = export_step_onnx(Path(args.checkpoint), Path(args.output), opset=args.opset)
    print(f"Exported step ONNX: {path}")


if __name__ == "__main__":
    main()
