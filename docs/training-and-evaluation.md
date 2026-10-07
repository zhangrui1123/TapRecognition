# Training And Evaluation Workflow

This project evaluates detector changes as controlled experiments. Train one candidate, evaluate it on full recordings, export its metrics, then compare compatible completed runs. Do not compare notebook training accuracy alone.

## Experiment Controls

For a fair architecture or target comparison, keep these settings fixed unless they are the experiment variable:

- train and validation recording directories;
- seed;
- high-pass preprocessing;
- session exclusions;
- window size and negative-window sampling;
- optimizer, learning rate, epoch limit, and early stopping;
- target delay and amplitude augmentation, except when explicitly being ablated;
- evaluation policy, threshold grid, matching tolerance, and refractory time.

Run resource-heavy experiments one at a time on this machine. Name the output directory clearly and preserve `best.pt`, `last.pt`, `history.csv`, and `threshold_comparison/` together.

## 1. Train A Candidate

Open the repository with the `.venv` kernel and run notebook cells in order.

| Notebook | Use it for | Main controls |
| --- | --- | --- |
| [`main_notebooks/training_gru.ipynb`](../main_notebooks/training_gru.ipynb) | Causal CNN + GRU experiments. | `LABEL_DELAY_FRAMES`, seed, dataset paths, training configuration. |
| [`main_notebooks/training_lstm.ipynb`](../main_notebooks/training_lstm.ipynb) | Causal CNN + LSTM experiments. | `LSTM_HIDDEN_SIZE`, `LSTM_LAYERS`, `LABEL_DELAY_FRAMES`, `AMPLITUDE_AUGMENTATION`, seed, and dataset paths. |

Both notebooks train on 300-frame windows and save checkpoint metadata needed to reconstruct the model. The LSTM notebook also stores `model_type="lstm"` and its augmentation setting.

### Delayed Targets

The original Gaussian label assigns positive mass around the second-tap frame, including frames before the second impact. A causal detector cannot yet observe the completed double tap at those earlier frames.

The training notebooks can move only the soft left/right frame targets later with `LABEL_DELAY_FRAMES`. An 8-frame delay at 100 Hz equals 80 ms. The original sparse event annotation, window label, and event-matching reference frame remain unchanged.

This must be applied consistently in both training and validation loss calculations. `evaluate.ipynb` reads `positive_label_delay_frames` from the checkpoint so its frame/window metrics use the same supervision definition.

## 2. Evaluate The Saved Run

Set `RUN_DIR` at the top of [`main_notebooks/evaluate.ipynb`](../main_notebooks/evaluate.ipynb) to the completed output directory. The notebook never retrains the model.

It reports:

- training history and selected checkpoint epoch;
- window and frame metrics;
- continuous-recording matched-event precision, recall, F1, latency, and false alarms per minute;
- left/right metrics and per-recording/per-recording-type breakdowns;
- instant and look-ahead alarm policies;
- a left/right threshold-pair sweep.

The notebook exports all comparison inputs to:

```text
<RUN_DIR>/threshold_comparison/
```

Important files include:

| File | Contents |
| --- | --- |
| `selected_thresholds.csv` | Chosen threshold pairs for each selection mode and policy. |
| `pair_sweep.csv` | Every evaluated left/right threshold pair with matched-event metrics. |
| `comparison_summary.csv` | Window, frame, and event summaries by side. |
| `recording_metrics_all_modes.csv` | Per-recording results. |
| `type_metrics_all_modes.csv` | Results grouped by recording type. |

## 3. Compare Completed Runs

Edit `RUNS`, `SELECTION_MODE`, and `POLICY` in [`main_notebooks/compare_models.ipynb`](../main_notebooks/compare_models.ipynb). Point each entry at a distinct `threshold_comparison/` directory.

The notebook reads exports only. It does not retrain, re-sweep, or re-tune models. Before comparing scores, it verifies that the candidates share:

- validation and preprocessing fingerprints;
- selection/evaluation split and evaluation design;
- threshold grid and selection constraints;
- matching tolerance, timing budget, refractory duration, and label width;
- recording and recording-type coverage.

This prevents a visually attractive but invalid comparison between models evaluated on different data or rules.

`compare_8frames.ipynb` and `compare_GRUvsLSTM.ipynb` are historical comparison notebooks. Use `compare_models.ipynb` for the current canonical comparison.

## Threshold Selection And Reporting

The evaluator supports multiple threshold-selection objectives, including maximum F1, recall targets, false-alarm budgets, and timely-recall budgets. They represent different operating points; do not call them interchangeable.

The final current selection is LSTM64 with an 8-frame target delay. The final documented operating point is left/right `0.50 / 0.50` with `lookahead_12f`. The automatic maximum-F1 pair is a separate result and must be reported separately from the manually selected deployment point.

All current experiment metrics use `tuned_validation`: the same validation recordings were used to select thresholds and to report scores. Treat those values as model-selection evidence, not an independent generalization estimate.

## 4. Freeze And Test Externally

After selecting a model and operating point:

1. Preserve the exact checkpoint, model metadata, preprocessing settings, policy, thresholds, and refractory time.
2. Do not change those values for the external data.
3. Evaluate truly unseen recordings with a dedicated frozen-configuration evaluation script or notebook.
4. Report that external result separately from validation selection metrics.

The current repository does not retain a completed delay-10 artifact. Do not describe a delay-10 result until it has a preserved checkpoint and evaluation export.
