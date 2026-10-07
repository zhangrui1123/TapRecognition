"""Auditable inference/metric helpers for 03_state_reset_vs_continuous.ipynb.

No training or changes to the production model. All comparisons use eval mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

from tap_recognition.dataset import load_session_exclusions, load_trigger_events
from tap_recognition.inference import pick_tap_events
from tap_recognition.labels import three_class_frame_labels
from tap_recognition.physics import IMUSimulator
from tap_recognition.recording import IMU_COLUMNS, estimate_sample_rate_hz
from train import frame_soft_ce, window_class_from_probs


# The middle two arms isolate GRU state while holding CNN behavior constant.
MODES = {
    "reset_both": (False, False),
    "carry_gru_only": (False, True),
    "carry_cnn_only": (True, False),
    "continuous": (True, True),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Recording:
    split: str
    path: Path
    user: str
    action: str
    fs: float
    time: np.ndarray
    imu: np.ndarray
    labels: np.ndarray
    events: list[tuple[int, int]]
    bounds: list[tuple[int, int]]


def load_recordings(root: Path, data_config: dict, half_frames: int):
    """Load only non-excluded sessions while preserving their original frame order."""
    recordings, audit, manifest = [], [], []
    registry_value = data_config.get("session_exclusions_file")
    registry = Path(registry_value) if registry_value else None
    if registry is not None and not registry.is_absolute():
        registry = root / registry
    exclusions = load_session_exclusions(registry) if registry is not None else []
    excluded_by_file: dict[str, set[int]] = {}
    for exclusion in exclusions:
        excluded_by_file.setdefault(exclusion.source_file, set()).add(
            exclusion.segment_index
        )

    for split, key in [("train", "train_dir"), ("valid", "val_dir")]:
        for directory in data_config[key].split(","):
            paths = sorted((root / directory.strip()).glob("*.csv"))
            assert paths, f"No recordings in {directory}"
            for path in paths:
                if path.name.endswith(".labels.csv"):
                    continue
                df = pd.read_csv(path)
                timestamps = df.timestamp_ms.to_numpy(dtype=np.float64)
                dt = np.diff(timestamps)
                fs = estimate_sample_rate_hz(timestamps)
                # Fail rather than silently join gaps, duplicate or reversed timestamps.
                assert np.all(dt > 0), f"Non-monotonic time: {path}"
                assert np.allclose(dt, 1000 / fs, atol=0.01, rtol=0), path
                seg = df.segment_index.to_numpy()
                starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
                ends = np.r_[starts[1:], len(df)]
                segment_ids = seg[starts].astype(int)
                indexed_bounds = list(zip(segment_ids.tolist(), starts.tolist(), ends.tolist()))
                assert len(np.unique(segment_ids)) == len(indexed_bounds), path
                excluded_ids = excluded_by_file.get(path.name, set())
                unknown = excluded_ids - set(segment_ids.tolist())
                assert not unknown, f"{path.name}: unknown excluded segment(s) {sorted(unknown)}"
                included_bounds = [
                    (start, end)
                    for segment_id, start, end in indexed_bounds
                    if segment_id not in excluded_ids
                ]
                if not included_bounds:
                    raise ValueError(f"{path.name}: all segments are excluded")
                label_path = path.with_suffix(".txt")
                assert label_path.exists(), f"Missing labels are NOT negatives: {path}"
                events = load_trigger_events(label_path)
                assert len(set(events)) == len(events), path
                assert all(0 <= f < len(df) and c in (1, 2) for f, c in events), path
                expected = 1 if "knock_twice_left" in path.name.lower() else (
                    2 if "knock_twice_right" in path.name.lower() else 0
                )
                for segment_id, start, end in indexed_bounds:
                    if segment_id in excluded_ids:
                        continue
                    local = [(f, c) for f, c in events if start <= f < end]
                    assert len(local) <= int(expected > 0), (path, start, local)
                    assert all(c == expected for _, c in local), path
                imu = df[IMU_COLUMNS].to_numpy(dtype=np.float32)
                assert np.isfinite(imu).all(), path
                if data_config["highpass"]:
                    imu = IMUSimulator(sample_rate=fs).highpass(imu)
                included_events = [
                    (frame, cls)
                    for frame, cls in events
                    if any(start <= frame < end for start, end in included_bounds)
                ]
                excluded_event_count = len(events) - len(included_events)
                filtered_imu_parts = []
                filtered_time_parts = []
                remapped_events = []
                bounds = []
                cursor = 0
                for start, end in included_bounds:
                    filtered_imu_parts.append(imu[start:end])
                    filtered_time_parts.append((timestamps[start:end] - timestamps[0]) / 1000)
                    bounds.append((cursor, cursor + end - start))
                    remapped_events.extend(
                        (frame - start + cursor, cls)
                        for frame, cls in included_events
                        if start <= frame < end
                    )
                    cursor += end - start
                filtered_imu = np.concatenate(filtered_imu_parts, axis=0)
                filtered_time = np.concatenate(filtered_time_parts)
                labels = three_class_frame_labels(
                    len(filtered_imu), remapped_events, half_frames=half_frames
                )
                recordings.append(Recording(
                    split, path, str(df.user_name.iloc[0]), str(df.action_label.iloc[0]),
                    fs, filtered_time, filtered_imu, labels, remapped_events, bounds,
                ))
                lengths = np.asarray([end - start for start, end in included_bounds])
                audit.append(dict(
                    split=split, file=path.name, user=str(df.user_name.iloc[0]),
                    action=str(df.action_label.iloc[0]), frames=int(lengths.sum()), fs=fs,
                    minutes=int(lengths.sum()) / fs / 60, segments=len(indexed_bounds),
                    included_segments=len(included_bounds),
                    excluded_segments=len(indexed_bounds) - len(included_bounds),
                    excluded_segment_ids=",".join(map(str, sorted(excluded_ids))),
                    events=len(included_events), excluded_events=excluded_event_count,
                    segment_min=int(lengths.min()), segment_max=int(lengths.max()),
                    non_300_segments=int((lengths != 300).sum()),
                    dropped_by_300_fit=int(np.maximum(lengths - 300, 0).sum()),
                    padded_by_300_fit=int(np.maximum(300 - lengths, 0).sum()),
                    min_dt_ms=float(dt.min()), max_dt_ms=float(dt.max()),
                ))
                for source in (path, label_path):
                    manifest.append(dict(path=str(source.relative_to(root)), sha256=sha256(source)))
    if registry is not None and registry.is_file():
        manifest.append(dict(path=str(registry.relative_to(root)), sha256=sha256(registry)))
    keys = [(r.split, r.path.name) for r in recordings]
    assert len(set(keys)) == len(keys), "Duplicate input files"
    return recordings, pd.DataFrame(audit), manifest


def cnn_features(model, x):
    """x [1,T,6] -> [1,T,C], using the checkpoint's frozen BatchNorm statistics."""
    return model.dropout(model.cnn(model.input_proj(x).transpose(1, 2)).transpose(1, 2))


@torch.inference_mode()
def predict_modes(model, recording):
    """2x2 state ablation on exactly the same, unmodified frames.

    Continuous CNN features can be computed once because the CNN is causal in
    eval mode. Recurrent chunks carry state only within this call/recording.
    """
    assert not model.training
    x = torch.from_numpy(recording.imu).unsqueeze(0)
    full_features = cnn_features(model, x)
    reset_features = [cnn_features(model, x[:, a:b]) for a, b in recording.bounds]
    recurrent = model.lstm if hasattr(model, "lstm") else model.gru
    logits_by_mode, hidden_by_mode = {}, {}
    for mode, (carry_cnn, carry_gru) in MODES.items():
        state, parts, states = None, [], []
        for i, (a, b) in enumerate(recording.bounds):
            features = full_features[:, a:b] if carry_cnn else reset_features[i]
            output, state = recurrent(features, state if carry_gru else None)
            parts.append(model.head(output))
            hidden = state[0] if isinstance(state, tuple) else state
            states.append(hidden[0, 0].cpu().numpy().copy())
        logits_by_mode[mode] = torch.cat(parts, dim=1)[0]
        hidden_by_mode[mode] = np.stack(states)

    reference, _ = model(x)
    torch.testing.assert_close(logits_by_mode["continuous"], reference[0], atol=2e-5, rtol=2e-5)
    for mode in MODES:
        first_end = recording.bounds[0][1]
        torch.testing.assert_close(
            logits_by_mode[mode][:first_end], reference[0, :first_end], atol=2e-5, rtol=2e-5
        )
    # Independent direct forward() for each raw reset segment.
    direct_reset = torch.cat([model(x[:, a:b])[0] for a, b in recording.bounds], dim=1)[0]
    torch.testing.assert_close(logits_by_mode["reset_both"], direct_reset, atol=2e-5, rtol=2e-5)
    return logits_by_mode, hidden_by_mode


@torch.inference_mode()
def overlap_stream_logits(model, x, chunk_sizes):
    """Independent correct streaming reference, without the repository's step().

    Keep real raw-input history (not fictitious zero history). Recompute at most
    receptive_field-1 frames for the CNN, then feed ONLY new recurrent features.
    Works for single frames as well as arbitrary chunk sizes.
    """
    assert not model.training and sum(chunk_sizes) == x.shape[1]
    context, state, parts, start = None, None, [], 0
    keep = model.receptive_field - 1
    recurrent = model.lstm if hasattr(model, "lstm") else model.gru
    for size in chunk_sizes:
        current = x[:, start:start + size]
        buffered = current if context is None else torch.cat([context, current], dim=1)
        features = cnn_features(model, buffered)[:, -size:]
        output, state = recurrent(features, state)
        parts.append(model.head(output))
        context = buffered[:, -keep:] if keep else buffered[:, :0]
        start += size
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def numerical_checks(model, recordings):
    """Real positive + negative signals: causality, chunking, and legacy diagnostic."""
    selected = [next(r for r in recordings if r.events), next(r for r in recordings if not r.events)]
    rows = []
    for rec in selected:
        x = torch.from_numpy(rec.imu[:640]).unsqueeze(0)
        reference, _ = model(x)
        for size in (1, 7, 53, 300):
            sizes = [min(size, x.shape[1] - a) for a in range(0, x.shape[1], size)]
            actual = overlap_stream_logits(model, x, sizes)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            rows.append(dict(file=rec.path.name, check=f"correct streaming, chunk={size}",
                             max_logit_error=float((actual - reference).abs().max()), passed=True))
        changed = x.clone()
        changed[:, 320:] += 50
        changed_logits, _ = model(changed)
        torch.testing.assert_close(reference[:, :320], changed_logits[:, :320], atol=2e-5, rtol=2e-5)
        rows.append(dict(file=rec.path.name, check="future perturbation", passed=True,
                         max_logit_error=float((reference[:, :320] - changed_logits[:, :320]).abs().max())))
        # A diagnostic, NOT the continuous arm: step() currently replays CNN history into GRU.
        h = buffer = None
        legacy = []
        for i in range(x.shape[1]):
            prob, h, buffer = model.step(x[:, i], h, buffer)
            legacy.append(prob)
        legacy = torch.cat(legacy, dim=1)
        error = (legacy - reference.softmax(-1)).abs()
        rows.append(dict(file=rec.path.name, check="existing step() diagnostic (not an equivalence assertion)",
                         max_probability_error=float(error.max()), mean_probability_error=float(error.mean())))
    return pd.DataFrame(rows)


def match_events(truth, predicted, tolerance_frames, class_aware=True):
    """Maximum-cardinality 1:1 matching, then minimum timing error.

    Dummy columns permit unmatched truths. A wrong class counts as FP + FN in
    class-aware metrics; duplicates cannot share the same truth event.
    """
    n, m = len(truth), len(predicted)
    if n == 0 or m == 0:
        return []
    unmatched = float(n + 1)
    cost = np.full((n, m + n), unmatched)
    cost[:, :m] = 2 * unmatched
    for i, (frame, cls) in enumerate(truth):
        for j, (pred_frame, pred_cls, _prob) in enumerate(predicted):
            distance = abs(frame - pred_frame)
            if distance <= tolerance_frames + 1e-9 and (not class_aware or cls == pred_cls):
                cost[i, j] = distance / (tolerance_frames + 1)
    row, col = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(row, col) if j < m and cost[i, j] < 1]


def check_event_matching():
    """Regression cases for the metric, including an ambiguous greedy-match trap."""
    assert match_events([], [(10, 1, .9)], 10) == []
    assert match_events([(10, 1)], [], 10) == []
    assert len(match_events([(10, 1)], [(10, 1, .9), (11, 1, .8)], 2)) == 1
    assert not match_events([(10, 1)], [(10, 2, .9)], 2)
    assert len(match_events([(10, 1)], [(10, 2, .9)], 2, False)) == 1
    assert len(match_events([(10, 1)], [(12, 1, .9)], 2)) == 1
    assert not match_events([(10, 1)], [(13, 1, .9)], 2)
    assert len(match_events([(10, 1), (14, 1)], [(12, 1, .9), (16, 1, .9)], 2)) == 2
    return "8 event-matching regression assertions passed"


def event_counts(rec, probs, threshold, tolerance_sec, look_ahead_sec, refractory_sec):
    predicted = pick_tap_events(
        probs, threshold=threshold, sample_rate=rec.fs,
        look_ahead_sec=look_ahead_sec, refractory_sec=refractory_sec,
        consecutive_on=1, require_prior_tap=False,
    )
    matches = match_events(rec.events, predicted, rec.fs * tolerance_sec)
    binary_matches = match_events(rec.events, predicted, rec.fs * tolerance_sec, False)
    errors_ms = [(predicted[j][0] - rec.events[i][0]) / rec.fs * 1000 for i, j in matches]
    matched_pred = {j: i for i, j in matches}
    details = []
    for j, (frame, cls, probability) in enumerate(predicted):
        truth_frame = rec.events[matched_pred[j]][0] if j in matched_pred else None
        details.append(dict(
            frame=frame, time_sec=float(rec.time[frame]), class_id=cls, probability=probability,
            matched=j in matched_pred, matched_true_frame=truth_frame,
            timing_error_ms=(frame - truth_frame) / rec.fs * 1000 if truth_frame is not None else None,
        ))
    return dict(
        event_tp=len(matches), event_fp=len(predicted) - len(matches),
        event_fn=len(rec.events) - len(matches),
        binary_event_tp=len(binary_matches), binary_event_fp=len(predicted) - len(binary_matches),
        binary_event_fn=len(rec.events) - len(binary_matches),
        timing_abs_ms_sum=sum(abs(x) for x in errors_ms), timing_signed_ms_sum=sum(errors_ms),
        negative_fp=len(predicted) if not rec.events else 0,
        negative_minutes=len(rec.imu) / rec.fs / 60 if not rec.events else 0,
    ), details


@torch.inference_mode()
def score_recording(rec, mode, logits, config):
    """Window, frame and event metrics; temporal matches never use label positions as predictions."""
    probs_t = logits.softmax(-1)
    probs = probs_t.numpy()
    target = torch.from_numpy(rec.labels)
    segments = []
    for index, (a, b) in enumerate(rec.bounds):
        truth = next((c for f, c in rec.events if a <= f < b), 0)
        pred = int(window_class_from_probs(probs_t[a:b].unsqueeze(0), config["threshold"])[0])
        loss = frame_soft_ce(logits[a:b].unsqueeze(0), target[a:b].unsqueeze(0),
                             config["event_loss_weight"], torch.tensor([truth]), config["neg_window_weight"])
        segments.append(dict(
            split=rec.split, file=rec.path.name, action=rec.action, user=rec.user, mode=mode,
            segment=index + 1, start=a, end=b, truth=truth, pred=pred, correct=int(truth == pred),
            loss=float(loss), max_event_probability=float(probs[a:b, 1:].max()),
        ))
    ec, event_details = event_counts(rec, probs, config["threshold"], config["tolerance_sec"],
                                     config["look_ahead_sec"], config["refractory_sec"])
    frame_correct = int((probs_t.argmax(-1) == target.argmax(-1)).sum())
    row = dict(split=rec.split, file=rec.path.name, action=rec.action, user=rec.user, mode=mode,
               minutes=len(probs) / rec.fs / 60, frames=len(probs), frame_correct=frame_correct,
               windows=len(segments), window_correct=sum(s["correct"] for s in segments),
               loss_sum=sum(s["loss"] for s in segments), **ec)
    for t in range(3):
        for p in range(3):
            row[f"cm_{t}{p}"] = sum(s["truth"] == t and s["pred"] == p for s in segments)
    return row, segments, event_details


def safe_divide(a, b):
    return a / b if b else float("nan")


def aggregate(group):
    """Micro metrics from additive recording statistics. Undefined metrics stay NaN."""
    totals = group.select_dtypes(include="number").sum()
    tp = sum(totals[f"cm_{t}{p}"] for t in (1, 2) for p in (1, 2))
    fp = totals.cm_01 + totals.cm_02
    fn = totals.cm_10 + totals.cm_20
    result = dict(
        recordings=len(group), minutes=totals.minutes, windows=totals.windows,
        window_acc=totals.window_correct / totals.windows,
        frame_acc=totals.frame_correct / totals.frames, loss=totals.loss_sum / totals.windows,
        window_precision=safe_divide(tp, tp + fp), window_recall=safe_divide(tp, tp + fn),
        window_f1=safe_divide(2 * tp, 2 * tp + fp + fn),
        window_tp=tp, window_fp=fp, window_fn=fn, window_tn=totals.cm_00,
        event_fp_per_min=totals.event_fp / totals.minutes,
        negative_fp_per_min=safe_divide(totals.negative_fp, totals.negative_minutes),
        matched_peak_mae_ms=safe_divide(totals.timing_abs_ms_sum, totals.event_tp),
        matched_peak_bias_ms=safe_divide(totals.timing_signed_ms_sum, totals.event_tp),
    )
    for prefix in ("event", "binary_event"):
        t, p, n = (totals[f"{prefix}_{suffix}"] for suffix in ("tp", "fp", "fn"))
        result.update({
            f"{prefix}_tp": t, f"{prefix}_fp": p, f"{prefix}_fn": n,
            f"{prefix}_precision": safe_divide(t, t + p),
            f"{prefix}_recall": safe_divide(t, t + n),
            f"{prefix}_f1": safe_divide(2 * t, 2 * t + p + n),
        })
    return result


def summarize(file_metrics, group_columns=("split", "mode")):
    rows = []
    for key, group in file_metrics.groupby(list(group_columns), sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(dict(zip(group_columns, key)) | aggregate(group))
    return pd.DataFrame(rows)


def paired_bootstrap(file_metrics, draws=2000, seed=42):
    """Paired, recording-level, action-stratified bootstrap. Never resample frames."""
    rng = np.random.default_rng(seed)
    comparisons = [
        ("reset_both", "continuous"),
        ("reset_both", "carry_gru_only"),
        ("carry_cnn_only", "continuous"),
    ]
    metrics = ["window_acc", "event_f1", "event_recall", "negative_fp_per_min"]
    results = []
    for split, split_rows in file_metrics.groupby("split", sort=False):
        for before, after in comparisons:
            left = split_rows[split_rows["mode"].eq(before)].sort_values("file").reset_index(drop=True)
            right = split_rows[split_rows["mode"].eq(after)].sort_values("file").reset_index(drop=True)
            assert left.file.tolist() == right.file.tolist()
            strata = [g.index.to_numpy() for _, g in left.groupby("action")]
            base_l, base_r = aggregate(left), aggregate(right)
            # Vectorized additive sufficient statistics makes 2,000 draws inexpensive.
            numeric = left.select_dtypes(include="number").columns
            indices = np.concatenate([rng.choice(s, (draws, len(s))) for s in strata], axis=1)
            def bootstrap_metrics(df):
                sums = pd.DataFrame(df[numeric].to_numpy()[indices].sum(axis=1), columns=numeric)
                return {
                    "window_acc": sums.window_correct / sums.windows,
                    "event_f1": 2 * sums.event_tp / (2 * sums.event_tp + sums.event_fp + sums.event_fn),
                    "event_recall": sums.event_tp / (sums.event_tp + sums.event_fn),
                    "negative_fp_per_min": sums.negative_fp / sums.negative_minutes,
                }
            sample_l, sample_r = bootstrap_metrics(left), bootstrap_metrics(right)
            for metric in metrics:
                differences = (sample_r[metric] - sample_l[metric]).to_numpy()
                lo, hi = np.quantile(differences, [.025, .975])
                results.append(dict(split=split, before=before, after=after, metric=metric,
                                    delta=base_r[metric] - base_l[metric], ci_low=lo, ci_high=hi))
    return pd.DataFrame(results)
