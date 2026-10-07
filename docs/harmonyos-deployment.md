# HarmonyOS Streaming Export

Training checkpoints are PyTorch state dictionaries plus metadata. HarmonyOS deployment requires a streaming graph that accepts one IMU frame and returns the updated model state.

## Export A Streaming ONNX Graph

Install the additional exporter dependency when needed:

```bash
pip install onnx
```

Export a checkpoint with:

```bash
./.venv/bin/python tools/export_step_for_harmony.py \
  --checkpoint checkpoints/best.pt \
  --output checkpoints/tap_step.onnx \
  --opset 14
```

The exporter loads GRU or LSTM checkpoints through `model_factory` and writes a one-sample ONNX graph.

### GRU I/O

| Tensor | Shape |
| --- | --- |
| `imu` | `[1, 1, 6]` |
| `cnn_buffer` | `[1, cnn_channels, receptive_field]` |
| `h_in` | `[gru_layers, 1, gru_hidden]` |
| `prob` | `[1, 1, 3]` |
| `h_out` | `[gru_layers, 1, gru_hidden]` |
| `cnn_buffer_out` | `[1, cnn_channels, receptive_field]` |

### LSTM I/O

The LSTM graph additionally receives and returns a cell state:

| Tensor | Shape |
| --- | --- |
| `imu` | `[1, 1, 6]` |
| `cnn_buffer` | `[1, cnn_channels, receptive_field]` |
| `h_in`, `c_in` | `[lstm_layers, 1, lstm_hidden]` |
| `prob` | `[1, 1, 3]` |
| `h_out`, `c_out` | `[lstm_layers, 1, lstm_hidden]` |
| `cnn_buffer_out` | `[1, cnn_channels, receptive_field]` |

The application must feed every output state back into the next inference call. Reset all state only when beginning a new independent stream.

## Convert With MindSpore Lite

Use the official MindSpore Lite converter for the target platform to convert the ONNX graph to `.ms`. The converter is external to this repository; install the version required by the HarmonyOS application and provide the expected runtime libraries there.

The previous static-window export description is not the current deployment path. Use the step export above so the device-side graph receives the same recurrent and CNN history used by Python streaming inference.

## Verify The Converted Model

Use the verification tool matching the recurrent head:

```bash
# GRU export
./.venv/bin/python tools/verify_harmony_step.py --help

# LSTM export
./.venv/bin/python tools/verify_harmony_lstm_step.py --help
```

These tools compare PyTorch streaming output, ONNX output, and MindSpore Lite runtime output on a recording. Run verification whenever the model, exporter, converter, or device runtime changes.

## Deployment Configuration

The model graph alone is not the detector. Deploy the same non-neural settings used during evaluation:

- high-pass preprocessing;
- sample rate;
- left/right thresholds;
- look-ahead duration;
- consecutive-on rule;
- prior-tap confirmation settings when enabled;
- refractory interval;
- state reset policy.

For the current selected experiment, keep the LSTM64-delay-8 checkpoint and its frozen `0.50 / 0.50` threshold pair together with this configuration.
