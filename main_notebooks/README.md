# Training, Evaluation, And Comparison Notebooks

Use these notebooks for controlled recorded-data experiments. Open them with the repository `.venv` Python kernel and run cells in order. The expected sequence is train one run, evaluate that saved run, then compare its exports with other evaluated runs.

Do not train multiple candidates simultaneously on this machine. Preserve every completed run's checkpoints, history, and evaluation export under a distinct directory.

## Canonical Workflow

```text
training_gru.ipynb or training_lstm.ipynb
                    |
                    v
          <run directory>/best.pt
          <run directory>/last.pt
          <run directory>/history.csv
                    |
                    v
              evaluate.ipynb
                    |
                    v
  <run directory>/threshold_comparison/*.csv
                    |
                    v
            compare_models.ipynb
```

All compared runs must use compatible data, preprocessing, matching rules, and threshold grid. `compare_models.ipynb` verifies those requirements before it combines results.

## Train A GRU

[`training_gru.ipynb`](training_gru.ipynb) trains the original causal CNN plus GRU architecture.

Set before training:

- seed;
- train and validation directories;
- output directory or run name;
- `LABEL_DELAY_FRAMES`;
- standard training settings in `TrainConfig` and `TrainingConfig`.

The notebook applies positive-window amplitude augmentation and writes the target delay into checkpoint metadata. Use `LABEL_DELAY_FRAMES = 0` for an unshifted baseline and a positive value only for a deliberate delayed-target experiment.

## Train An LSTM

[`training_lstm.ipynb`](training_lstm.ipynb) changes only the recurrent head. It keeps the causal CNN and classifier head aligned with the GRU experiment.

Set before training:

- `LSTM_HIDDEN_SIZE`;
- `LSTM_LAYERS`;
- `LABEL_DELAY_FRAMES`;
- `AMPLITUDE_AUGMENTATION`;
- seed, data directories, output directory, and training settings.

For a controlled GRU/LSTM comparison, keep all non-architecture settings identical. The LSTM checkpoint records `model_type="lstm"`, layer count, delay, and augmentation setting.

## Evaluate One Saved Run

[`evaluate.ipynb`](evaluate.ipynb) performs full-recording evaluation; it does not retrain or overwrite the selected checkpoint.

Set `RUN_DIR` to the completed run directory. The notebook reconstructs the correct GRU or LSTM model from checkpoint metadata, reads the target delay, and produces:

- training-history plots;
- window, frame, and matched-event metrics;
- instant and look-ahead policies;
- left/right threshold-pair sweeps;
- latency, early/late detection, false-alarm, and recording-type analysis;
- CSV exports in `RUN_DIR/threshold_comparison/`.

Evaluation data is used for model selection and threshold selection. Its results are not independent test estimates.

## Compare Evaluated Runs

[`compare_models.ipynb`](compare_models.ipynb) reads completed `threshold_comparison/` exports from multiple runs.

Edit:

- `RUNS`: names and paths to each run's export directory;
- `SELECTION_MODE`: for example, `max_f1`;
- `POLICY`: for example, `lookahead_12f`.

The notebook shows operating points, window/frame/event metrics, per-type results, per-recording behavior, and saved-checkpoint examples. It does not retrain or change thresholds. It rejects incompatible exports instead of silently comparing different datasets or rules.

## Historical Notebooks

| Notebook | Purpose |
| --- | --- |
| [`compare_8frames.ipynb`](compare_8frames.ipynb) | Earlier saved-run comparison focused on delayed-target experiments. |
| [`compare_GRUvsLSTM.ipynb`](compare_GRUvsLSTM.ipynb) | Earlier GRU/LSTM comparison with example inspection. |

Use `compare_models.ipynb` for new comparisons. Historical notebooks remain useful for understanding the experiment evolution but should not replace the canonical evaluation/export flow.

## Related Documentation

- [`../docs/training-and-evaluation.md`](../docs/training-and-evaluation.md): controlled experiment rules, exported files, thresholds, and external testing.
- [`../docs/data-collection-and-labeling.md`](../docs/data-collection-and-labeling.md): recording layout and strict auto-labeling window.
- [`../notebooks_analysis/README.md`](../notebooks_analysis/README.md): read-only data, label, and state-analysis notebooks.
