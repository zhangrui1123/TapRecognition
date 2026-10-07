# Data, Label, And Streaming Analysis Notebooks

These notebooks inspect existing recordings, labels, checkpoints, and exports. They are not the normal training workflow; use [`../main_notebooks/README.md`](../main_notebooks/README.md) to train and evaluate model candidates.

Open notebooks with the repository `.venv` Python kernel. `pandas` is included in the root `requirements.txt` because notebooks 01, 02, and 04 use it.

## 01. Label Audit Metrics

[`01_label_audit_metrics.ipynb`](01_label_audit_metrics.ipynb) is a read-only inventory and consistency audit of existing CSV recordings and their sparse/dense sidecars. It does not create or modify labels.

It checks:

- expected versus written event counts for positive and negative recordings;
- label class, frame bounds, and segment containment;
- sparse/dense sidecar consistency;
- stored second-tap frame versus the current auto-labeling heuristic;
- inter-tap interval and peak-strength distributions;
- accepted-pair candidates in negative recordings;
- sample rate, segment duration, and event position within each segment.

Agreement with the auto-labeling heuristic means only that a label is consistent with that heuristic. It is not human confirmation of a physical double tap. Send suspicious or uncertain segments to notebook 02.

## 02. Visual Label Review

[`02_visual_label_review.ipynb`](02_visual_label_review.ipynb) is a read-only human review aid. It shows a full-recording overview plus a detailed three-second segment view containing accelerometer, gyroscope, auto-labeling energy, sparse markers, and dense labels.

For each reviewed positive segment, record one decision in the notebook's `review_decisions` cell:

| Decision | Meaning |
| --- | --- |
| `correct` | Two credible impacts; the stored second-tap label is defensible. |
| `uncertain` | Noise, vibration, or multiple candidates prevent a defensible judgment. |
| `incorrect` | The label is missing a tap or points to unrelated movement. |
| `too_close_end` | There is not enough trailing context for the training window. |

Move the resulting exclusions, including their reason, into [`../data/session_exclusions.csv`](../data/session_exclusions.csv). Do not overwrite labels simply because the current heuristic produces a different peak; make an explicit reviewed-data decision.

## 03. State Reset Versus Continuous Inference

[`03_state_reset_vs_continuous.ipynb`](03_state_reset_vs_continuous.ipynb) evaluates an existing checkpoint without retraining. Set `CHECKPOINT` near the beginning of the notebook before running it.

It provides:

- reproduction of the checkpoint's 300-frame dataset metrics;
- a paired 2x2 boundary ablation over original, unpadded recording frames;
- separate CNN-history and recurrent-state carry/reset effects;
- batch, chunked, and single-frame streaming-equivalence checks;
- matched-event metrics, probability differences, bootstrap intervals, temporal plots, and threshold/tolerance sensitivity.

The four continuous-stream arms are:

| Arm | CNN history at segment boundary | Recurrent state at segment boundary |
| --- | --- | --- |
| `reset_both` | Reset | Reset |
| `carry_gru_only` | Reset | Carry |
| `carry_cnn_only` | Carry | Reset |
| `continuous` | Carry | Carry |

The main comparison is `continuous - reset_both`. State is always reset between independent recordings. The notebook evaluates causal `forward()` and `model.step()` equivalence in evaluation mode; neither result is a training run or a deployment benchmark by itself.

It writes CSV tables and a hash manifest to `notebooks_analysis/results/state_reset_<checkpoint-directory>_<checkpoint-stem>/`. The reusable numerical helpers live in [`state_reset_analysis.py`](state_reset_analysis.py).

## 04. Recording Inventory And Split Balance

[`04_recording_inventory_split_balance.ipynb`](04_recording_inventory_split_balance.ipynb) reads CSV filenames only. It does not load IMU samples, require sidecars, or modify files.

Use it before auto-labeling or training to inspect train/validation coverage by participant, gesture, recording family, and split. It recognizes both regular names such as `imu_<person>_<gesture>_<timestamp>.csv` and strict-window names such as `imuStrict_<person>_<gesture>_<timestamp>.csv`.

This notebook identifies imbalance and participant overlap but does not itself enforce a split. Use [`../tools/materialize_person_split.py`](../tools/materialize_person_split.py) when materializing a participant-level split from a manifest.

## Recommended Sequence

1. Run notebook 04 when new recordings arrive to inspect naming, participant allocation, and gesture balance.
2. Auto-label with the strict window that was used during collection; see [`../docs/data-collection-and-labeling.md`](../docs/data-collection-and-labeling.md).
3. Run notebook 01 to identify coverage or integrity anomalies.
4. Review anomalous and sampled positive segments in notebook 02, then update `session_exclusions.csv` deliberately.
5. Train/evaluate candidates through `main_notebooks`.
6. Use notebook 03 only when investigating streaming-boundary behavior of a saved checkpoint.
