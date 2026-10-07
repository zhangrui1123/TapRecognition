"""Compare a streaming Harmony export against PyTorch on real preprocessed IMU.

Example (MindSpore Lite 2.9 device-side runtime):
  LD_LIBRARY_PATH=/path/to/lite/tools/converter/lib:/path/to/lite/runtime/lib \
    python tools/verify_harmony_step.py --checkpoint best.pt --onnx tap_step.onnx \
    --ms tap_step.ms --runtime-lib /path/to/lite/runtime/lib/libmindspore-lite.so \
    --recording data/valid_data/recording.csv
"""

from __future__ import annotations

import argparse
import ctypes as C
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tap_recognition.dataset import load_full_recording
from tap_recognition.model import CausalCNNGRU
from tap_recognition.physics import IMUSimulator


class TensorArray(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]


class MindSporeStep:
    """Small ctypes wrapper over the device-side MindSpore Lite 2.9 C API."""

    def __init__(
        self,
        lib_path: Path,
        model_path: Path,
        input_names: tuple[str, ...] = ("imu", "cnn_buffer", "h_in"),
        output_shapes: tuple[tuple[int, ...], ...] = ((1, 1, 3), (1, 1, 64), (1, 32, 29)),
        output_names: tuple[str, ...] = ("prob", "h_out", "cnn_buffer_out"),
    ):
        lib = C.CDLL(str(lib_path))
        self.lib = lib
        lib.MSContextCreate.restype = C.c_void_p
        lib.MSDeviceInfoCreate.argtypes = [C.c_int]
        lib.MSDeviceInfoCreate.restype = C.c_void_p
        lib.MSContextAddDeviceInfo.argtypes = [C.c_void_p, C.c_void_p]
        lib.MSContextSetThreadNum.argtypes = [C.c_void_p, C.c_int]
        lib.MSModelCreate.restype = C.c_void_p
        lib.MSModelBuildFromFile.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p]
        lib.MSModelBuildFromFile.restype = C.c_int
        lib.MSModelGetInputs.argtypes = [C.c_void_p]
        lib.MSModelGetInputs.restype = TensorArray
        lib.MSModelPredict.argtypes = [C.c_void_p, TensorArray, C.POINTER(TensorArray), C.c_void_p, C.c_void_p]
        lib.MSModelPredict.restype = C.c_int
        lib.MSTensorGetName.argtypes = [C.c_void_p]
        lib.MSTensorGetName.restype = C.c_char_p
        lib.MSTensorGetMutableData.argtypes = [C.c_void_p]
        lib.MSTensorGetMutableData.restype = C.c_void_p
        lib.MSTensorGetData.argtypes = [C.c_void_p]
        lib.MSTensorGetData.restype = C.c_void_p
        lib.MSTensorGetDataSize.argtypes = [C.c_void_p]
        lib.MSTensorGetDataSize.restype = C.c_size_t
        self.context = C.c_void_p(lib.MSContextCreate())
        lib.MSContextSetThreadNum(self.context, 1)
        lib.MSContextAddDeviceInfo(self.context, lib.MSDeviceInfoCreate(0))  # CPU
        self.model = C.c_void_p(lib.MSModelCreate())
        status = lib.MSModelBuildFromFile(self.model, str(model_path).encode(), 0, self.context)
        if status != 0:
            raise RuntimeError(f"MSModelBuildFromFile failed: {status}")
        self.inputs = lib.MSModelGetInputs(self.model)
        names = [lib.MSTensorGetName(self.inputs.handle_list[i]).decode() for i in range(self.inputs.handle_num)]
        assert names == list(input_names), names
        self.output_names = output_names
        self.output_shapes = output_shapes

    def step(self, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        assert len(arrays) == self.inputs.handle_num
        for i, array in enumerate(arrays):
            array = np.ascontiguousarray(array, dtype=np.float32)
            tensor = self.inputs.handle_list[i]
            assert self.lib.MSTensorGetDataSize(tensor) == array.nbytes
            C.memmove(self.lib.MSTensorGetMutableData(tensor), array.ctypes.data, array.nbytes)
        outputs = TensorArray()
        status = self.lib.MSModelPredict(self.model, self.inputs, C.byref(outputs), None, None)
        if status != 0:
            raise RuntimeError(f"MSModelPredict failed: {status}")
        assert outputs.handle_num == len(self.output_names), outputs.handle_num
        names = [self.lib.MSTensorGetName(outputs.handle_list[i]).decode() for i in range(outputs.handle_num)]
        assert names == list(self.output_names), names
        result = []
        for i, shape in enumerate(self.output_shapes):
            tensor = outputs.handle_list[i]
            n = int(np.prod(shape))
            assert self.lib.MSTensorGetDataSize(tensor) == n * 4
            ptr = C.cast(self.lib.MSTensorGetData(tensor), C.POINTER(C.c_float))
            result.append(np.ctypeslib.as_array(ptr, shape=(n,)).copy().reshape(shape))
        return tuple(result)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "ms", "runtime-lib", "recording"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=360)
    args = parser.parse_args()
    assert args.frames > 29, "Verify a sequence longer than the CNN receptive field"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    assert (cfg["input_dim"], cfg["num_classes"], cfg["cnn_channels"],
            cfg["gru_hidden"], cfg["gru_layers"]) == (6, 3, 32, 64, 1)
    model = CausalCNNGRU(**ckpt["model_config"]).eval()
    model.load_state_dict(ckpt["model_state"])
    raw, fs = load_full_recording(args.recording)
    assert abs(fs - ckpt["train_config"]["data"]["sample_rate"]) < 0.5
    samples = IMUSimulator(sample_rate=fs).highpass(raw)[:args.frames]
    assert len(samples) == args.frames

    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)
    assert not any(node.op_type == "GRU" for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == ["imu", "cnn_buffer", "h_in"]
    assert [x.name for x in session.get_outputs()] == ["prob", "h_out", "cnn_buffer_out"]

    torch.set_num_threads(1)
    x = torch.from_numpy(samples).unsqueeze(0)
    ms = MindSporeStep(args.runtime_lib, args.ms)
    max_errors = {"torch_step_prob": 0.0, "onnx_prob": 0.0, "onnx_h": 0.0,
                  "onnx_buffer": 0.0, "ms_prob": 0.0, "ms_h": 0.0, "ms_buffer": 0.0}
    h_onnx = np.zeros((1, 1, 64), dtype=np.float32)
    b_onnx = np.zeros((1, 32, 29), dtype=np.float32)
    h_ms, b_ms = h_onnx.copy(), b_onnx.copy()
    with torch.inference_mode():
        expected_logits, expected_h = model(x)
        expected = expected_logits.softmax(-1).numpy()
        h_torch = torch.zeros((1, 1, 64))
        b_torch = torch.zeros((1, 32, 29))
        for i in range(len(samples)):
            imu = samples[i:i+1][None, ...]
            p_torch, h_torch, b_torch = model.step(torch.from_numpy(imu), h_torch, b_torch)
            onnx_p, h_onnx, b_onnx = session.run(None, {"imu": imu, "cnn_buffer": b_onnx, "h_in": h_onnx})
            ms_p, h_ms, b_ms = ms.step(imu, b_ms, h_ms)
            for label, actual, reference in (
                ("torch_step_prob", p_torch.numpy(), expected[:, i:i+1]),
                ("onnx_prob", onnx_p, expected[:, i:i+1]),
                ("onnx_h", h_onnx, h_torch.numpy()),
                ("onnx_buffer", b_onnx, b_torch.numpy()),
                ("ms_prob", ms_p, expected[:, i:i+1]),
                ("ms_h", h_ms, h_torch.numpy()),
                ("ms_buffer", b_ms, b_torch.numpy()),
            ):
                max_errors[label] = max(max_errors[label], float(np.max(np.abs(actual - reference))))
        max_errors["final_torch_h"] = float(np.max(np.abs(h_torch.numpy() - expected_h.numpy())))
    # MindSpore Lite 2.9's C-API destroy path can double-free this model on
    # Linux; this one-shot verifier releases its allocations at process exit.
    for name, error in max_errors.items():
        print(f"{name}: {error:.8g}")
        assert error < (2e-3 if name.startswith("ms_") else 1e-4), f"{name} diverged at frame {i}"
    print(f"PASS: {len(samples)} causal frames, with independent ONNX and MS state feedback")


if __name__ == "__main__":
    main()
