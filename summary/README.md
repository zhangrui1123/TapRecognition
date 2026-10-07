# Experiment Record: Labeling, Delay, and LSTM Work

This is a working record for preparing the project README and presentation. It is intentionally detailed rather than polished for a public audience.

## Scope And Evidence

This record was assembled by reading the current source code, notebooks, and retained CSV exports only. No notebook cells, training jobs, evaluations, or tests were run while creating it.

The work described here covers repository changes and experiments after the original detector baseline. Demo-specific changes are intentionally outside this record and should be documented separately.

The retained comparison artifacts are under `testing_checkpoints/`:

| Directory | Meaning |
| --- | --- |
| `baseline_old_data/` | Original baseline evaluated on the earlier dataset. Historical reference only. |
| `baseline/` | `(4)baseline`: GRU with no target delay on `data_02_10_26`. |
| `delay8/` | `(4)delay8`: GRU with 8-frame target delay on `data_02_10_26`. |
| `lstm64d8/` | `(4)lstm64d8`: one-layer 64-unit LSTM with 8-frame target delay. Final selected model. |
| `lstm64-64d8/` | `(4)lstm64-64d8`: two-layer 64-unit LSTM with 8-frame target delay. |

Only the retained evaluation exports should be treated as reproducible numeric evidence here. The old model-weight files were removed. In particular, there is no retained delay-10 result, so this document makes no claim that a 10-frame experiment was completed.

## Detector Context

The detector receives six IMU channels per frame: three accelerometer axes and three gyroscope axes. It emits one of three frame classes:

| Class | Meaning |
| --- | --- |
| `0` | No double tap |
| `1` | Left double tap |
| `2` | Right double tap |

The common front end is a causal CNN:

- An input projection maps six input channels to 32 channels.
- Three causal convolution blocks use kernel size 5 with dilations `(1, 2, 4)`.
- Each block currently uses `BatchNorm1d` and GELU.
- The causal CNN receptive field is 29 frames.
- Dropout is 0.1 before the recurrent layer and in the classification head.

The original recurrent head is a one-layer, 64-unit GRU. The new alternatives replace only that recurrent head with a unidirectional LSTM while retaining the CNN front end and the same three-class head.

### Current BatchNorm Decision

The CNN still uses `BatchNorm1d`. Its training-time statistics span the batch and the training-window time axis, which is a known causality/training-inference-mismatch question. A LayerNorm replacement was considered but was not adopted because it would change the learned amplitude representation and had not been evaluated. The reported results and final selection therefore use the existing BatchNorm architecture.

## Data Quality And Auto-Labeling

### Segment-Aware Auto-Labeling

Recordings are collected as consecutive approximately three-second segments. Positive recordings are expected to contain one double tap per segment. The auto-labeler detects a first and second accelerometer-energy peak, `n1` and `n2`, subject to the allowed inter-tap interval of 0.12 to 0.45 seconds.

The labeling tool, `tools/auto_label_imu.py`, now supports:

```text
--strict-spike-window-ms START_MS END_MS
```

For `imuStrict_*` recordings this option is mandatory. It constrains every segment independently:

- The first tap must occur strictly after `START_MS`.
- The second tap must occur strictly before `END_MS`.
- Candidate pairs outside the designated collection window are rejected.

This prevents a strong unintended movement elsewhere in a segment from being selected instead of the intended double tap. If robust peak thresholding is dominated by an early accidental movement, the algorithm falls back to ranking all local peaks inside the requested window, rather than discarding a weaker intended pair.

Regular `imu_*` recordings retain unrestricted pairing; passing the strict option for them emits a warning and does not change their behavior.

### Audit And Exclusions

The label audit was not limited to agreement with the auto-labeler. The workflow added visual review and manual exclusions:

- `notebooks_analysis/01_label_audit_metrics.ipynb` checks coverage, side/class integrity, segment containment, dense-label validity, automatic-pair consistency, negative-control pairs, sample rate, and event placement.
- `notebooks_analysis/02_visual_label_review.ipynb` provides recording- and segment-level plots of accelerometer, gyroscope, energy, stored `n2`, recalculated `n2`, and dense labels for human decisions.
- `data/session_exclusions.csv` records manually rejected segments, including incorrect labels, uncertain cases, taps too close to the end of a segment, and segments where the double tap was not found.
- `notebooks_analysis/04_recording_inventory_split_balance.ipynb` reads filenames only to show participant, gesture, and train/validation balance before labeling.

The stricter auto-labeling window and human review were intended to improve label precision before asking the model to learn a temporal target.

## Delayed Gaussian Frame Targets

### Original Problem

The dense supervision is a three-class Gaussian centered at the annotated second-tap frame, `n2`, with a default half-width of 10 frames. At 100 Hz, the original target therefore assigned non-zero double-tap probability to frames before `n2`.

For a causal detector, those early frames can be contradictory: the decisive second impact has not happened yet, so the model cannot yet have observed all information needed to assert a double tap. The model can reduce training loss by learning an early proxy rather than detecting the completed double tap.

### Change

The training notebooks add `shift_event_targets_right(frame_labels, shift)`.

For an 8-frame delay:

- Only the generated soft left/right frame targets move eight samples later.
- The remaining probability mass is assigned back to the `none` class.
- The original window label and annotated event frame remain unchanged.
- Event matching and latency remain referenced to the original annotation, not the delayed training target.

At 100 Hz, 8 frames equals 80 ms. A Gaussian that previously covered approximately `n2 - 10` through `n2 + 10` is shifted to approximately `n2 - 2` through `n2 + 18`. This removes almost all of the pre-second-tap supervision while allowing the causal model time to observe the second impact.

The delayed target is saved in checkpoint metadata as `positive_label_delay_frames`, and `evaluate.ipynb` reads that value to apply the matching delayed labels for frame/window evaluation.

### Result

The 8-frame delay was the largest successful experimental change. On the retained `data_02_10_26` validation export, the GRU window recall at the automated maximum-F1 point increased from 81.97% without delay to 93.68% with delay. The delayed targets also enabled the strongest LSTM result described below.

## LSTM Support

The repository now supports a causal CNN plus LSTM model alongside the original CNN plus GRU.

### Implementation

`tap_recognition/model_lstm.py` adds `CausalCNNLSTM`:

- A unidirectional `nn.LSTM` replaces the GRU.
- `lstm_hidden` and `lstm_layers` are configurable.
- The LSTM is batch-first, has biases, has no projection, and is not bidirectional.
- Recurrent inter-layer dropout is zero; the existing external dropout remains in the same positions as the GRU model.
- `forward()` resets hidden and cell state unless an explicit state is supplied.
- `step()` maintains both hidden state and cell state, as well as the packed per-convolution history required for sample-by-sample causal CNN inference.
- Explicit one-step LSTM gates are used in streaming inference to avoid an ONNX LSTM operator whose converted recurrent state did not match PyTorch in MindSpore Lite.

Related integration changes:

- `LSTMModelConfig` in `tap_recognition/config.py` serializes LSTM architecture settings.
- `tap_recognition/model_factory.py` builds either GRU or LSTM models from checkpoint metadata, while inferring old GRU checkpoints when no model type exists.
- `train.py` supports `--recurrent-type`, `--recurrent-hidden`, and `--recurrent-layers`.
- Checkpoints save `model_type`, model configuration, target delay, and the LSTM notebook's amplitude-augmentation setting.
- `tap_recognition/inference.py`, `tools/export_step_for_harmony.py`, and `tools/verify_harmony_lstm_step.py` support LSTM loading, ONNX streaming export, and hidden/cell-state verification.
- `tests/test_model_lstm.py` covers forward/step equivalence, hidden/cell reset behavior, checkpoint round trips, shared evaluation, online-detector state management, and CLI model selection.

### Architectures Evaluated

The retained current-data experiment family used seed 101, `data_02_10_26`, the same causal CNN front end, and the same validation/export procedure.

| ID | Recurrent head | Delay | Layers | Hidden size |
| --- | --- | ---: | ---: | ---: |
| `(4)baseline` | GRU | 0 frames | 1 | 64 |
| `(4)delay8` | GRU | 8 frames | 1 | 64 |
| `(4)lstm64d8` | LSTM | 8 frames | 1 | 64 |
| `(4)lstm64-64d8` | LSTM | 8 frames | 2 | 64 per layer |

## Amplitude Augmentation Ablation

During training, `_augment_positive_windows()` uniformly scales all six IMU channels of each positive window by a random factor in `[0.70, 1.10]`. Negative windows are unchanged. The intent is to expose the model to lighter taps without changing the tap's temporal structure or inter-axis relationship.

Both augmented and non-augmented LSTM variants were tried in the earlier experiment cycle. The result was small and inconsistent:

- For the one-layer LSTM, augmentation traded a little precision for better recall.
- For the two-layer LSTM, aggregate differences were negligible and error locations changed rather than producing a consistent winner.

This did not justify treating augmentation as a proven improvement or changing the default experiment strategy. The retained final LSTM notebook configuration keeps `AMPLITUDE_AUGMENTATION = True` to match the GRU training procedure, but augmentation is not the reason the model was selected.

## Evaluation Protocol

`main_notebooks/evaluate.ipynb` evaluates a saved checkpoint without retraining. It:

- Loads model architecture and target delay from checkpoint metadata.
- Computes window, frame, and matched-event metrics.
- Replays continuous recordings and evaluates both instant and `lookahead_12f` alarm policies.
- Sweeps left/right thresholds from 0.30 to 0.80 in increments of 0.05.
- Uses a 0.5-second refractory interval, 10-frame label half-width, 1-frame early allowance, 10-frame timely budget, and a maximum lateness of 30 frames.
- Exports selection tables, threshold-pair sweeps, overall summaries, per-recording metrics, and per-recording-type metrics to `<run>/threshold_comparison/`.

`main_notebooks/compare_models.ipynb` loads those exports and rejects a comparison when validation fingerprints, preprocessing fingerprints, matching settings, threshold grid, or evaluation constraints differ.

All retained new-data model exports share:

| Field | Value |
| --- | --- |
| Evaluation design | `tuned_validation` |
| Validation event count | 526 |
| Validation duration | 43.7248 minutes |
| Negative-only duration | 17.1682 minutes |
| Validation fingerprint | `3901f797b614397900a7dd0e2a0c25fea106c4517bde24879ae5f88f92749d58` |
| Preprocessing fingerprint | `40ddabf9aec9de1a21a8932f8b48cd0a24853560f2ce44153f53f3d9e68c0fdc` |

These are model-selection results from the validation recordings, not independent test-set estimates. Thresholds were tuned on the same validation data used to report the scores. Any future external-test evaluation must keep the final model and thresholds frozen.

## Controlled Model Comparison

### Window Metrics At Each Model's Automated Maximum-F1 Point

The following rows are the `max_f1` selections under the `lookahead_12f` policy. They are useful for architecture comparison, but their threshold pair is independently tuned for each model and is not the final deployment threshold.

| Model | Thresholds L/R | Window TP | Window FP | Window FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU, delay 0 | 0.75 / 0.80 | 441 | 6 | 97 | 98.66% | 81.97% | 89.54% |
| GRU, delay 8 | 0.75 / 0.75 | 504 | 8 | 34 | 98.44% | 93.68% | 96.00% |
| LSTM64, delay 8 | 0.55 / 0.55 | 519 | 10 | 19 | 98.11% | **96.47%** | **97.28%** |
| LSTM64-64, delay 8 | 0.75 / 0.60 | 504 | 15 | 34 | 97.11% | 93.68% | 95.36% |

The one-layer LSTM64 with delayed targets achieved the highest window recall and F1. Adding a second 64-unit LSTM layer did not improve the result enough to justify its added complexity.

### Final Common Operating Point: 0.50 / 0.50

The final selected deployment configuration is:

```text
Model:       (4)lstm64d8
Architecture: causal CNN + one-layer LSTM, 64 hidden units
Target delay: 8 frames (80 ms at 100 Hz)
Policy:      lookahead_12f
Thresholds:  left = 0.50, right = 0.50
```

At the same fixed `0.50 / 0.50` threshold pair, the event-matching sweep gives the following comparison. These are matched-event metrics, not the window metrics in the preceding table.

| Model | TP | FP | FN | Precision | Recall | F1 | Macro recall | FP/min | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU, delay 0 | 443 | 195 | 83 | 69.44% | 84.22% | 76.12% | 84.32% | 4.46 | 80 ms |
| GRU, delay 8 | 484 | 171 | 42 | 73.89% | 92.02% | 81.96% | 92.03% | 3.91 | 140 ms |
| LSTM64, delay 8 | **498** | **46** | **28** | **91.54%** | **94.68%** | **93.08%** | **94.68%** | **1.05** | 150 ms |
| LSTM64-64, delay 8 | 495 | 71 | 31 | 87.46% | 94.11% | 90.66% | 94.15% | 1.62 | 150 ms |

The single-layer LSTM was selected because it has the strongest balanced result at the shared final threshold pair:

- Highest overall matched-event recall: 498 of 526 events, or 94.68%.
- Strong and balanced side recall: 94.92% left and 94.44% right.
- Highest matched-event precision and F1 among the four candidates.
- Far fewer false alarms than the GRU variants and the two-layer LSTM: 46 total, or 1.05 per minute.
- A symmetric 0.50/0.50 configuration is simpler to explain and deploy than side-specific thresholds.

The threshold sweep's automated `max_f1` row for this LSTM is a separate result: it selected 0.55/0.55, with 497 TP, 40 FP, 29 FN, 92.55% precision, 94.49% recall, and 93.51% F1. The chosen 0.50/0.50 point accepts six additional false alarms for one extra matched event and slightly higher recall. This distinction must be preserved in any final report: the final threshold was a deliberate operating-point choice, not the automatic maximum-F1 pair.

## Why The Final Model Is LSTM64D8

The final model selection is supported by two levels of evidence:

1. With each model allowed its own maximum-F1 threshold pair, the delayed one-layer LSTM has the best window recall and window F1.
2. At the final shared 0.50/0.50 operating point, it also has the best matched-event precision, recall, F1, and false-alarm rate.

The central finding is not merely that an LSTM is better than a GRU. The largest observed gain comes from making the training target more compatible with causal detection by delaying the Gaussian label. The LSTM adds a further improvement after that change. A second LSTM layer does not add a reliable benefit.

## Notebook And Tool Map

### Main Experiment Notebooks

| File | Role | Current Status |
| --- | --- | --- |
| `main_notebooks/training_gru.ipynb` | Train the causal CNN + GRU baseline; exposes `LABEL_DELAY_FRAMES`. | Active training notebook. |
| `main_notebooks/training_lstm.ipynb` | Train LSTM variants; exposes hidden size, layer count, delay, and amplitude augmentation. | Active training notebook. |
| `main_notebooks/evaluate.ipynb` | Evaluate one saved run, perform threshold-pair sweeps, and export comparable CSVs. | Canonical evaluation notebook. |
| `main_notebooks/compare_models.ipynb` | Compare the current four retained runs and validate compatible metadata. | Canonical comparison notebook. |
| `main_notebooks/compare_8frames.ipynb` | Historical saved-run comparison focused on delayed-target experiments. | Reference/exploration notebook. |
| `main_notebooks/compare_GRUvsLSTM.ipynb` | Historical saved-run GRU/LSTM comparison with example replay. | Reference/exploration notebook. |
| `main_notebooks/baseline_evaluation_delayed_labels.ipynb` | Earlier delayed-label evaluation notebook. | Historical predecessor of the canonical evaluator. |
| Frozen external evaluation workflow | Evaluate a held-out directory without changing model or thresholds. | Prepare a dedicated script or notebook before reporting generalization. |

### Analysis And Labeling Tools

| File | Role |
| --- | --- |
| `tools/auto_label_imu.py` | Segment-aware auto-labeling, including the strict spike window for `imuStrict_*` files. |
| `tools/plot_labels.py` | Plot energy, tap markers, and dense labels for a recording. |
| `notebooks_analysis/01_label_audit_metrics.ipynb` | Read-only label coverage, integrity, consistency, and negative-control audit. |
| `notebooks_analysis/02_visual_label_review.ipynb` | Read-only visual inspection and human-review workflow. |
| `notebooks_analysis/03_state_reset_vs_continuous.ipynb` | Analysis of reset versus continuous inference and streaming equivalence. |
| `notebooks_analysis/04_recording_inventory_split_balance.ipynb` | Filename-only inventory by participant, gesture, and split. |
| `notebooks_analysis/state_reset_analysis.py` | Reusable helpers for continuous/replay analysis. |

## Important Limitations And Next Formalization Steps

- The final numerical claims are validation-set claims. They are not independent external-test performance estimates.
- The final model's exported weights are no longer retained in `testing_checkpoints/`; only its evaluation CSVs remain. Preserve future selected checkpoints, their exact metadata, and the final threshold configuration together.
- Do not compare `baseline_old_data/` absolute scores to the new-data runs. It uses a different validation and preprocessing fingerprint and contains 446 events rather than 526.
- The BatchNorm causality question remains open and was intentionally not changed. It should be a separately controlled architecture experiment, not mixed into the current result story.
- The delay-10 idea has no retained artifact. It remains a future controlled experiment, not a conclusion.
- Before reporting the final model as generalizable, run a frozen-configuration evaluation on truly unseen recordings with LSTM64D8 and the frozen 0.50/0.50 thresholds.
- The public project README can be derived from this document by reducing the detailed audit history, retaining the final model and validation caveat, and adding the separate demo changes.
