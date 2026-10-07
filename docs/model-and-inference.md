# Model And Online Inference

## Input And Output

Each IMU frame contains six values:

```text
[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

The model produces three logits per frame:

```text
0: none
1: left double tap
2: right double tap
```

Input recording preprocessing includes high-pass filtering before inference. Training and deployment must use compatible preprocessing.

## Causal CNN Front End

Both recurrent variants share the same `CausalConv1d` stack:

- input projection from six to 32 channels;
- three causal convolution blocks;
- kernel size 5 and dilations `(1, 2, 4)`;
- left-only padding;
- BatchNorm1d and GELU in each block;
- receptive field of 29 samples.

Left-only padding ensures each convolution output at time `t` depends on input samples at or before `t`. The model remains causal through the recurrent head because the GRU and LSTM are unidirectional.

### BatchNorm Caveat

The current CNN retains BatchNorm. In evaluation it uses frozen running statistics; during training its statistics span the current batch and training window. This is a known training-time causality question and was intentionally not changed without a controlled performance experiment. All retained reported results use this architecture.

## Recurrent Heads

### GRU

`tap_recognition/model.py` defines `CausalCNNGRU`, the original architecture. Its default head is one GRU layer with 64 hidden units.

### LSTM

`tap_recognition/model_lstm.py` defines `CausalCNNLSTM`. It uses the same CNN and classifier head with a configurable unidirectional LSTM:

- `lstm_hidden` selects hidden width;
- `lstm_layers` selects layer count;
- recurrent inter-layer dropout is disabled;
- external dropout remains before the recurrent layer and in the classifier head.

`tap_recognition/model_factory.py` reconstructs either model from checkpoint metadata. Older checkpoints without a `model_type` continue to load as GRU models.

## Streaming State

`model.step()` processes one frame at a time and returns class probabilities plus the state required for the next frame.

| Model | Recurrent state | CNN state |
| --- | --- | --- |
| GRU | Hidden state `[layers, batch, hidden]` | Packed per-convolution history. |
| LSTM | Hidden and cell states, each `[layers, batch, hidden]` | Packed per-convolution history. |

The CNN history must be retained separately for each convolution block. Replaying only the original projected input would not reproduce the left-padding behavior at the beginning of a stream.

Reset the online detector between independent recordings or sessions. Do not reset it between consecutive samples of the same stream.

## Online Event Postprocessing

`tap_recognition/inference.py` provides `OnlineDoubleTapDetector`.

For every sample it:

1. Applies causal high-pass filtering.
2. Runs `model.step()` with recurrent and CNN state.
3. Selects the higher left/right probability as the event score and class.
4. Starts a pending alarm once the configured threshold and consecutive-frame rule are satisfied.
5. Keeps the best score during the configured look-ahead interval.
6. Optionally confirms that a plausible first tap precedes the proposed second tap.
7. Emits one event and enforces the refractory period.

The final chosen experiment used a 12-frame look-ahead policy and symmetric left/right thresholds of 0.50. These values are operating-point settings, not model weights; save and deploy them with the selected checkpoint.

## Checkpoint Contract

A training checkpoint contains at least:

- `model_state`;
- `model_type` for LSTM checkpoints;
- `model_config`;
- `train_config`;
- selected epoch and validation metrics for `best.pt`;
- `positive_label_delay_frames` for notebook experiments;
- `amplitude_augmentation` for LSTM notebook experiments.

Preserve this metadata. Recreating a model from weights without its architecture, delay, preprocessing, policy, and thresholds is not a reproducible deployment.

## Local Inference Demonstration

Run the online inference demonstration with a saved checkpoint:

```bash
./.venv/bin/python -m tap_recognition.inference \
  --checkpoint checkpoints/best.pt --device cpu
```

The module loads the checkpoint through the model factory and writes `inference_demo.png` for synthetic double-tap and single-tap streams. `demo.py` provides a broader end-to-end visualization workflow.
