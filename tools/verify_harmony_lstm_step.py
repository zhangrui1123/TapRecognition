"""Verify a CNN/LSTM streaming export against full PyTorch forward().

Use the same arguments as tools/verify_harmony_step.py, with the LSTM .ms.
Run with LD_LIBRARY_PATH including the MindSpore Lite converter/lib and runtime/lib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tap_recognition.dataset import load_full_recording
from tap_recognition.model_factory import build_model
from tap_recognition.physics import IMUSimulator
from verify_harmony_step import MindSporeStep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "ms", "runtime-lib", "recording"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=360)
    args = parser.parse_args()
    assert args.frames > 29, "Verify beyond the full CNN receptive field"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    assert ckpt["model_type"] == "lstm"
    assert (cfg["input_dim"], cfg["num_classes"], cfg["cnn_channels"],
            cfg["lstm_hidden"]) == (6, 3, 32, 64)
    layers = cfg["lstm_layers"]
    assert layers in (1, 2)
    state_shape = (layers, 1, 64)
    model = build_model(cfg, "lstm").eval()
    model.load_state_dict(ckpt["model_state"])
    raw, fs = load_full_recording(args.recording)
    assert abs(fs - ckpt["train_config"]["data"]["sample_rate"]) < 0.5
    samples = IMUSimulator(sample_rate=fs).highpass(raw)[:args.frames]
    assert len(samples) == args.frames

    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)
    assert not any(node.op_type in ("GRU", "LSTM") for node in graph.graph.node)
    input_names = ("imu", "cnn_buffer", "h_in", "c_in")
    output_names = ("prob", "h_out", "c_out", "cnn_buffer_out")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == list(input_names)
    assert [x.name for x in session.get_outputs()] == list(output_names)

    ms = MindSporeStep(args.runtime_lib, args.ms, input_names,
                       ((1, 1, 3), state_shape, state_shape, (1, 32, 29)), output_names)
    torch.set_num_threads(1)
    h_onnx = np.zeros(state_shape, dtype=np.float32)
    c_onnx = h_onnx.copy()
    b_onnx = np.zeros((1, 32, 29), dtype=np.float32)
    h_ms, c_ms, b_ms = h_onnx.copy(), c_onnx.copy(), b_onnx.copy()
    errors = {label: 0.0 for label in ("torch_prob", "torch_h", "torch_c",
              "onnx_prob", "onnx_h", "onnx_c", "onnx_buffer",
              "ms_prob", "ms_h", "ms_c", "ms_buffer")}

    with torch.inference_mode():
        logits, (h_final, c_final) = model(torch.from_numpy(samples).unsqueeze(0))
        expected = logits.softmax(-1).numpy()
        state = (torch.zeros(state_shape), torch.zeros(state_shape))
        buffer = torch.zeros(1, 32, 29)
        for frame in range(len(samples)):
            imu = samples[frame:frame+1][None, ...]
            prob, state, buffer = model.step(torch.from_numpy(imu), state, buffer)
            onnx_prob, h_onnx, c_onnx, b_onnx = session.run(None, dict(zip(
                input_names, (imu, b_onnx, h_onnx, c_onnx))))
            ms_prob, h_ms, c_ms, b_ms = ms.step(imu, b_ms, h_ms, c_ms)
            reference = expected[:, frame:frame+1]
            for name, actual, target in (
                ("torch_prob", prob.numpy(), reference),
                ("onnx_prob", onnx_prob, reference),
                ("ms_prob", ms_prob, reference),
                ("onnx_h", h_onnx, state[0].numpy()),
                ("onnx_c", c_onnx, state[1].numpy()),
                ("onnx_buffer", b_onnx, buffer.numpy()),
                ("ms_h", h_ms, state[0].numpy()),
                ("ms_c", c_ms, state[1].numpy()),
                ("ms_buffer", b_ms, buffer.numpy()),
            ):
                errors[name] = max(errors[name], float(np.max(np.abs(actual - target))))
        errors["torch_h"] = float(np.max(np.abs(state[0].numpy() - h_final.numpy())))
        errors["torch_c"] = float(np.max(np.abs(state[1].numpy() - c_final.numpy())))

    for name, error in errors.items():
        print(f"{name}: {error:.8g}")
        assert error < (2e-3 if name.startswith("ms_") else 1e-4), f"{name} diverged by frame {frame}"
    print(f"PASS: {len(samples)} frames, all {layers} LSTM layer(s)' hidden and cell states fed back independently")


if __name__ == "__main__":
    main()
