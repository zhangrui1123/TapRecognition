# Data Collection And Labeling

This document describes the recording conventions and labeling workflow used by the detector. It applies to recorded-data experiments such as `data_02_10_26`.

## Recording Layout

A recording is a CSV containing timestamped IMU samples and segment metadata. The detector consumes six channels per sample:

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

Recordings are collected as consecutive approximately three-second segments. A segment is the unit used by the auto-labeler, review workflow, and expected-event checks.

The current dataset layout is:

```text
data_02_10_26/
├── train_data/
│   ├── <recording>.csv
│   ├── <recording>.txt
│   └── <recording>.labels.txt
└── valid_data/
    ├── <recording>.csv
    ├── <recording>.txt
    └── <recording>.labels.txt
```

The sparse sidecar `<recording>.txt` contains one `frame, class` row per annotated second tap. The dense `<recording>.labels.txt` sidecar contains `p_none`, `p_left`, and `p_right` soft labels.

Filename conventions determine the class:

| Filename contains | Class |
| --- | --- |
| `knock_twice_left` | Left double tap (`1`) |
| `knock_twice_right` | Right double tap (`2`) |
| `knock_twice` without a side suffix | Left double tap (`1`) |
| No double-tap name | Negative recording |

The permitted first-to-second tap interval is 0.12 to 0.45 seconds.

## Strict Collection Window

The newer collection demo can instruct the participant to perform the double tap inside a designated time range within each segment. This makes the intended pair easier to distinguish from incidental handling movement, earlier vibration, or late motion.

Use the same collection window when auto-labeling the recordings:

```bash
./.venv/bin/python tools/auto_label_imu.py \
  data_02_10_26/train_data data_02_10_26/valid_data \
  --strict-spike-window-ms <START_MS> <END_MS>
```

For every segment of an `imuStrict_*` recording, the auto-labeler accepts a pair only when:

- the first peak occurs strictly after `START_MS`;
- the second peak occurs strictly before `END_MS`;
- the pair's inter-tap interval is within the allowed range.

`imuStrict_*` files are rejected when no strict window is supplied. This is deliberate: a strict recording without its collection window cannot be reliably auto-labeled. Regular `imu_*` files keep unrestricted peak pairing; the tool warns if a strict window was passed for them.

The values are relative to the beginning of each segment, not to the beginning of the entire recording. Choose the exact values from the collection-demo instruction shown to the participant. Do not substitute arbitrary values later, because that changes the label definition.

## Auto-Labeling Behavior

`tools/auto_label_imu.py` detects accelerometer-energy peaks, pairs an earlier peak `n1` with a later peak `n2`, and writes `n2` as the labeled second-tap frame. It processes every recorded segment independently.

The strict window improves robustness in two ways:

- Pairs outside the instructed time range are excluded even when they are stronger than the intended tap.
- If a strong accidental movement raises the global robust threshold, the tool ranks all local peaks inside the requested strict window rather than hiding a weaker intended pair.

The tool also writes the dense three-class labels used by model training whenever it finds at least one event.

## Audit Before Training

Auto-label agreement is not a human correctness guarantee. Run the audit and visual review before treating a label set as training-ready.

| Resource | Purpose |
| --- | --- |
| [`notebooks_analysis/01_label_audit_metrics.ipynb`](../notebooks_analysis/01_label_audit_metrics.ipynb) | Read-only coverage, integrity, pair-consistency, negative-control, and recording-geometry checks. |
| [`notebooks_analysis/02_visual_label_review.ipynb`](../notebooks_analysis/02_visual_label_review.ipynb) | Full-recording and segment-level plots for human judgment of `n1`, stored `n2`, recalculated `n2`, IMU signal, and dense labels. |
| [`tools/plot_labels.py`](../tools/plot_labels.py) | Plot one recording's energy, marker positions, and soft labels. |
| [`data/session_exclusions.csv`](../data/session_exclusions.csv) | Segments excluded after manual review. |

Manual-review decisions should distinguish at least:

- `correct`: two credible impacts and a correct second-tap marker;
- `uncertain`: noise or multiple candidates prevent a defensible decision;
- `incorrect`: missing tap or a marker on unrelated movement;
- `too_close_end`: insufficient trailing context for a reliable training window.

Keep exclusions under version control with their reason. Exclusions are part of the dataset definition, not a post-hoc visualization choice.

## Split Hygiene

Keep train and validation recordings in separate directories before training. Check participant, gesture, and recording balance with [`notebooks_analysis/04_recording_inventory_split_balance.ipynb`](../notebooks_analysis/04_recording_inventory_split_balance.ipynb).

`tools/materialize_person_split.py` can materialize a checked participant-level split from a manifest. It verifies that a participant does not occur in both splits and can enforce a particular held-out participant configuration. Use a participant-separated external set when estimating generalization beyond the current validation recordings.
