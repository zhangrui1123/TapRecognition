# IMU Double-Tap Recognition

Detect left and right double taps on a laptop or PC chassis from a six-axis IMU stream: three accelerometer channels and three gyroscope channels. The detector is designed for online use: every prediction uses only current and previous samples.

The repository contains the complete iteration workflow: collect and label recordings, train GRU or LSTM models, evaluate full-recording behavior, compare saved runs, and export a streaming model for HarmonyOS.

## Current Experiment Result

The current selected experiment is a causal CNN with a one-layer, 64-unit LSTM trained with an 8-frame delayed target (`lstm64d8`). Its chosen operating point uses a 12-frame look-ahead policy with left/right thresholds of `0.50 / 0.50`.

This is a validation-set selection, not an independent test result. The detailed evidence, retained metrics, and limitations are in [`summary/README.md`](summary/README.md).

## Quick Start

Create and activate a Python environment, then install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebooks with the `.venv` Python kernel. The standard experiment loop is:

1. Prepare and auto-label recordings as described in [`docs/data-collection-and-labeling.md`](docs/data-collection-and-labeling.md).
2. Train one candidate in [`main_notebooks/training_gru.ipynb`](main_notebooks/training_gru.ipynb) or [`main_notebooks/training_lstm.ipynb`](main_notebooks/training_lstm.ipynb).
3. Set `RUN_DIR` and run [`main_notebooks/evaluate.ipynb`](main_notebooks/evaluate.ipynb) to create the full metric and threshold exports for that run.
4. Add the exported run to `RUNS` in [`main_notebooks/compare_models.ipynb`](main_notebooks/compare_models.ipynb) and compare compatible experiments side by side.

The notebooks do not replace each other: training creates one candidate, evaluation measures it, and comparison reads multiple completed evaluations. See [`main_notebooks/README.md`](main_notebooks/README.md) for the complete notebook workflow.

## Data Collection And Labeling

Recordings are divided into consecutive collection segments. The newer strict collection mode asks the participant to perform the double tap inside a designated time window within every segment. The auto-labeler receives the same window with:

```bash
./.venv/bin/python tools/auto_label_imu.py \
  data_02_10_26/train_data data_02_10_26/valid_data \
  --strict-spike-window-ms <START_MS> <END_MS>
```

For `imuStrict_*` recordings, this option is required. It requires the first detected tap to occur after `START_MS` and the second detected tap before `END_MS`, separately for every segment. This prevents incidental movement outside the intended collection interval from becoming the label.

Regular `imu_*` recordings retain unrestricted pairing. Full data layout, label formats, review, and exclusions are documented in [`docs/data-collection-and-labeling.md`](docs/data-collection-and-labeling.md).

## Architecture

```text
IMU sample stream [T, 6]
        |
        v
high-pass preprocessing
        |
        v
causal CNN: 32 channels, kernel 5, dilations 1/2/4
        |
        v
GRU or unidirectional LSTM recurrent head
        |
        v
per-frame logits: none / left / right
        |
        v
threshold, look-ahead, prior-tap check, refractory postprocessing
```

The CNN is left-padded and the recurrent head is unidirectional. The model supports both the original GRU and configurable LSTM heads. Read [`docs/model-and-inference.md`](docs/model-and-inference.md) for streaming state, target labels, checkpoint metadata, and known limitations.

The physical IMU and chassis-vibration formulation is in [`docs/mathematical_model.md`](docs/mathematical_model.md).

## Repository Map

```text
TapRecognition/
├── data_02_10_26/          # Current train/validation recording set
├── docs/                   # Focused technical documentation
├── main_notebooks/         # Training, evaluation, and comparison notebooks
├── notebooks_analysis/     # Read-only data, label, and streaming analyses
├── tap_recognition/        # Dataset, labels, models, and online inference
├── tools/                  # Auto-labeling, export, split, and verification tools
├── train.py                # CLI training entry point for GRU or LSTM
├── demo.py                 # End-to-end demonstration and visualization
├── summary/                # Detailed internal experiment record
└── requirements.txt
```

## Command-Line Entry Points

The notebooks are the preferred path for controlled delayed-target experiments. The shared script remains useful for regular training:

```bash
# GRU
./.venv/bin/python train.py --out-dir checkpoints_gru

# One-layer LSTM with 64 hidden units
./.venv/bin/python train.py \
  --recurrent-type lstm --recurrent-hidden 64 --recurrent-layers 1 \
  --out-dir checkpoints_lstm64

# Run the streaming inference demonstration
./.venv/bin/python -m tap_recognition.inference --checkpoint checkpoints/best.pt
```

The script's configuration defaults are defined in `tap_recognition/config.py`. Checkpoints store model type and architecture metadata so `tap_recognition.model_factory.build_model()` can load GRU, LSTM, and legacy GRU checkpoints.

## Further Documentation

| Document | Contents |
| --- | --- |
| [`docs/data-collection-and-labeling.md`](docs/data-collection-and-labeling.md) | Recording format, strict collection window, auto-labeling, audits, and exclusions. |
| [`docs/training-and-evaluation.md`](docs/training-and-evaluation.md) | Controlled experiment workflow, delayed targets, metrics, threshold selection, and external testing. |
| [`docs/model-and-inference.md`](docs/model-and-inference.md) | CNN/GRU/LSTM architecture, causal streaming state, checkpoints, and online postprocessing. |
| [`docs/harmonyos-deployment.md`](docs/harmonyos-deployment.md) | Streaming ONNX export and MindSpore Lite conversion requirements. |
| [`main_notebooks/README.md`](main_notebooks/README.md) | Detailed guide to training, evaluation, and comparison notebooks. |
| [`notebooks_analysis/README.md`](notebooks_analysis/README.md) | Detailed guide to dataset, label, and state-analysis notebooks. |
| [`summary/README.md`](summary/README.md) | Detailed working record of the completed labeling, delay, augmentation, and LSTM experiments. |

## Limitations

- Reported model-selection metrics use the validation recordings and threshold tuning; they are not independent test estimates.
- The selected model and final thresholds should be frozen before running an external test.
- The CNN currently retains BatchNorm. Its training-time normalization behavior is a future controlled experiment, not a claimed solved causality issue.
- Retained experiment CSVs do not replace preserving the selected checkpoint for deployment.
