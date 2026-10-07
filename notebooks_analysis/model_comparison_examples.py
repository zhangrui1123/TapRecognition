"""Replay validation-selected operating points on training or validation examples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

from tap_recognition.dataset import RecordedIMUDataset, load_session_exclusions, load_trigger_events
from tap_recognition.model_factory import build_model
from tap_recognition.physics import IMUSimulator
from tap_recognition.recording import IMU_COLUMNS, LabelParams, estimate_sample_rate_hz


@dataclass
class ExampleRecording:
    path: Path
    action: str
    user: str
    fs: float
    time: np.ndarray
    raw: np.ndarray
    imu: np.ndarray
    segments: list[tuple[int, int, int]]  # Included (segment_index, start, end).
    events: list[tuple[int, int]]
    windows: dict[int, tuple[np.ndarray, int]]

    @property
    def bounds(self) -> list[tuple[int, int]]:
        bounds = []
        for _, start, end in self.segments:
            if bounds and bounds[-1][1] == start:
                bounds[-1] = (bounds[-1][0], end)
            else:
                bounds.append((start, end))
        return bounds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples(root: Path, data: dict, labels: dict, seed: int, split: str = 'valid'):
    """Load a split's current frames and fitted windows without augmentation."""
    if split not in ('train', 'valid'):
        raise ValueError(f'Unsupported inspection split: {split!r}')
    directory_key = 'train_dir' if split == 'train' else 'val_dir'
    seed_offset = 0 if split == 'train' else 1
    registry = Path(data['session_exclusions_file']) if data.get('session_exclusions_file') else None
    if registry is not None and not registry.is_absolute():
        registry = root / registry
    excluded: dict[str, set[int]] = {}
    for item in load_session_exclusions(registry):
        excluded.setdefault(item.source_file, set()).add(item.segment_index)
    manifest = []
    recordings = {}
    for i, folder in enumerate(data[directory_key].split(',')):
        directory = Path(folder.strip())
        if not directory.is_absolute():
            directory = root / directory
        dataset = RecordedIMUDataset(
            directory, window_samples=data['window_samples'],
            label_params=LabelParams(**labels), highpass=data['highpass'],
            negative_windows_per_file=data['negative_windows_per_file'],
            trigger_context_samples=data['trigger_context_samples'],
            exclusion_margin=data['exclusion_margin'], dt_min=data['dt_min'],
            dt_max=data['dt_max'], seed=seed + seed_offset + i,
            session_exclusions_file=registry,
        )
        cursor = 0
        paths = sorted(p for p in directory.glob('*.csv') if not p.name.endswith('.labels.csv'))
        if not paths:
            raise ValueError(f'No {split} recordings in {directory}')
        for path in paths:
            if path.name in recordings:
                raise ValueError(f'Duplicate {split} filename: {path.name}')
            df = pd.read_csv(path)
            time_ms = df.timestamp_ms.to_numpy(dtype=np.float64)
            fs = estimate_sample_rate_hz(time_ms)
            assert np.allclose(np.diff(time_ms), 1000 / fs, atol=.01, rtol=0), path
            raw = df[IMU_COLUMNS].to_numpy(dtype=np.float32)
            imu = IMUSimulator(sample_rate=fs).highpass(raw) if data['highpass'] else raw
            ids = df.segment_index.to_numpy(dtype=int)
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            ends = np.r_[starts[1:], len(df)]
            assert len(set(ids[starts])) == len(starts), path
            unknown = excluded.get(path.name, set()) - set(ids[starts])
            if unknown:
                raise ValueError(f'{path.name}: unknown excluded segment(s) {sorted(unknown)}')
            segments = [(int(ids[start]), int(start), int(end)) for start, end in zip(starts, ends)
                        if ids[start] not in excluded.get(path.name, set())]
            if not segments:
                raise ValueError(f'{path.name}: all segments excluded')
            label_path = path.with_suffix('.txt')
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            events = load_trigger_events(label_path)
            included_events = [(frame, cls) for frame, cls in events
                               if any(start <= frame < end for _, start, end in segments)]
            windows = {}
            for sid, start, end in segments:
                local = [(frame, cls) for frame, cls in events if start <= frame < end]
                # RecordedIMUDataset skips an unlabeled segment in a labeled recording.
                if events and not local:
                    continue
                if cursor >= len(dataset.samples) or dataset.window_meta[cursor].source_file != path.name:
                    raise ValueError(f'Window order differs from RecordedIMUDataset: {path.name} {sid}')
                x, _, cls = dataset.samples[cursor]
                assert cls == (local[0][1] if local else 0)
                windows[sid] = (x, cls)
                cursor += 1
            recordings[path.name] = ExampleRecording(
                path, str(df.action_label.iloc[0]), str(df.user_name.iloc[0]), fs,
                (time_ms - time_ms[0]) / 1000, raw, imu, segments, included_events, windows,
            )
            for p in (path, label_path):
                manifest.append(dict(path=str(p.relative_to(root)), sha256=_sha256(p)))
        assert cursor == len(dataset.samples), f'Unmatched {split} windows in {directory}'
    if registry is not None and registry.is_file():
        manifest.append(dict(path=str(registry.relative_to(root)), sha256=_sha256(registry)))
    manifest.sort(key=lambda item: item['path'])
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return recordings, digest


def emit_alarms_pair(probs, left_threshold, right_threshold, look, refractory, bounds):
    """Same causal side-specific lookahead/refractory rule as baseline evaluation."""
    thresholds = np.array([left_threshold, right_threshold])
    alarms = []
    for start, end in bounds:
        segment, i, n = probs[start:end], 0, end - start
        while i < n:
            if not np.any(segment[i, 1:] >= thresholds):
                i += 1
                continue
            emission = i + look
            if emission >= n:
                break
            candidate = segment[i:emission + 1, 1:]
            eligible = np.where(candidate >= thresholds, candidate, -np.inf)
            offset, side = np.unravel_index(eligible.argmax(), eligible.shape)
            peak, cls = i + int(offset), int(side) + 1
            alarms.append(dict(alarm_frame=start + emission, peak_frame=start + peak,
                               class_id=cls, probability=float(segment[peak, cls])))
            i = max(emission + 1, peak + max(1, refractory))
    return alarms


def match_alarms(truth, alarms, early_allowance, max_late, class_aware=True):
    """Match across the full recording before attributing errors to segments."""
    n, m = len(truth), len(alarms)
    if not n or not m:
        return []
    cost = np.full((n, m + n), 1e6, dtype=float)
    cost[:, m:] = 1e3
    for a, (frame, cls) in enumerate(truth):
        for b, alarm in enumerate(alarms):
            delta = alarm['alarm_frame'] - frame
            if -early_allowance <= delta <= max_late and (not class_aware or cls == alarm['class_id']):
                cost[a, b] = abs(delta)
    rows, cols = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(rows, cols) if b < m and cost[a, b] < 1e3]


def _segment_at(segments, frame):
    return next((sid for sid, start, end in segments if start <= frame < end), None)


def replay_examples(root: Path, operating_points: pd.DataFrame, split: str = 'valid'):
    """Replay one split at frozen validation-selected thresholds in eval mode.

    Training replay is an in-sample label audit, not a generalization estimate.
    Its current-data hash is not compared with the exported validation hash.
    """
    if split not in ('train', 'valid'):
        raise ValueError(f'Unsupported inspection split: {split!r}')
    choices = operating_points.set_index('model', drop=False)
    checkpoints = {}
    for label, row in choices.iterrows():
        checkpoint_path = root / row['folder'] / '..' / 'best.pt'
        checkpoint_path = checkpoint_path.resolve()
        if _sha256(checkpoint_path) != row['checkpoint_sha256']:
            raise ValueError(f'{label}: best.pt differs from selected-threshold export')
        checkpoints[label] = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    first = next(iter(checkpoints.values()))['train_config']
    recordings, current_hash = load_examples(root, first['data'], first['labels'], first['seed'], split)
    for label, checkpoint in checkpoints.items():
        cfg = checkpoint['train_config']
        if any(cfg[key] != first[key] for key in ('data', 'labels', 'seed')):
            raise ValueError(f'{label}: {split} preprocessing/data config differs from other models')
    if split == 'valid':
        status = pd.DataFrame([dict(model=label, split=split,
                                     export_validation_sha256=row['validation_sha256'],
                                     current_validation_sha256=current_hash,
                                     matches_export=current_hash == row['validation_sha256'])
                               for label, row in choices.iterrows()])
    else:
        status = pd.DataFrame([dict(model=label, split=split, current_training_sha256=current_hash)
                               for label in choices.index])
    segments, alarm_rows, traces = [], [], {}
    looks = {'instant': 0, 'lookahead_12f': 12}
    with torch.inference_mode():
        for label, row in choices.iterrows():
            checkpoint = checkpoints[label]
            config = dict(checkpoint['model_config'])
            config['dilations'] = tuple(config['dilations'])
            model = build_model(config, checkpoint.get('model_type')).cpu().eval()
            model.load_state_dict(checkpoint['model_state'])
            look = looks[row['policy']]
            thresholds = np.array([row['left_threshold'], row['right_threshold']])
            for rec in recordings.values():
                probs = np.zeros((len(rec.imu), 3), dtype=np.float32)
                probs[:, 0] = 1
                for start, end in rec.bounds:
                    logits, _ = model(torch.from_numpy(rec.imu[start:end]).unsqueeze(0))
                    probs[start:end] = logits[0].softmax(-1).numpy()
                traces[(label, rec.path.name)] = probs
                detected = emit_alarms_pair(probs, *thresholds, look,
                                            round(float(row['refractory_sec']) * rec.fs), rec.bounds)
                matched = match_alarms(rec.events, detected, int(row['early_allowance_frames']),
                                       int(row['max_late_frames']))
                matched_by_event = dict(matched)
                matched_by_alarm = {a: t for t, a in matched}
                binary_matches = match_alarms(rec.events, detected, int(row['early_allowance_frames']),
                                              int(row['max_late_frames']), class_aware=False)
                binary_alarms = {a for _, a in binary_matches}
                file_alarms = []
                for index, alarm in enumerate(detected):
                    closest = min(rec.events, key=lambda event: abs(alarm['alarm_frame'] - event[0]),
                                  default=None)
                    nearest = alarm['alarm_frame'] - closest[0] if closest else None
                    truth_i = matched_by_alarm.get(index)
                    detail = dict(
                        model=label, split=split, file=rec.path.name,
                        segment_index=_segment_at(rec.segments, alarm['alarm_frame']),
                        **alarm, matched=truth_i is not None,
                        matched_true_frame=rec.events[truth_i][0] if truth_i is not None else np.nan,
                        latency_frames=alarm['alarm_frame'] - rec.events[truth_i][0]
                        if truth_i is not None else np.nan,
                        nearest_true_frame=closest[0] if closest else np.nan,
                        delta_to_nearest_true=nearest if nearest is not None else np.nan,
                        wrong_side=index in binary_alarms and truth_i is None,
                        too_early=truth_i is None and nearest is not None
                        and -int(row['max_late_frames']) <= nearest
                        < -int(row['early_allowance_frames']),
                    )
                    file_alarms.append(detail)
                    alarm_rows.append(detail)
                for sid, start, end in rec.segments:
                    local_truth = [(idx, frame) for idx, (frame, _) in enumerate(rec.events)
                                   if start <= frame < end]
                    local_alarms = [idx for idx, alarm in enumerate(detected)
                                    if start <= alarm['alarm_frame'] < end]
                    matched_truth = [idx for idx, _ in local_truth if idx in matched_by_event]
                    false_alarms = [idx for idx in local_alarms if idx not in matched_by_alarm]
                    early = sum(bool(file_alarms[idx]['too_early']) for idx in false_alarms)
                    window = rec.windows.get(sid)
                    if window is None:
                        window_true, window_pred, left_score, right_score = np.nan, np.nan, np.nan, np.nan
                    else:
                        x, window_true = window
                        logits, _ = model(torch.from_numpy(x).unsqueeze(0))
                        left_score, right_score = logits[0].softmax(-1)[:, 1:].max(dim=0).values.tolist()
                        eligible = np.where(np.array([left_score, right_score]) >= thresholds,
                                            [left_score, right_score], -np.inf)
                        window_pred = int(eligible.argmax() + 1) if np.isfinite(eligible).any() else 0
                    n_true = len(local_truth)
                    n_tp = len(matched_truth)
                    n_fp = len(false_alarms)
                    deltas = [detected[matched_by_event[idx]]['alarm_frame'] - frame
                              for idx, frame in local_truth if idx in matched_by_event]
                    segments.append(dict(model=label, split=split, file=rec.path.name, user=rec.user,
                                         action=rec.action,
                                         recording_type='knock_twice' if rec.action.startswith('knock_twice') else rec.action,
                                         segment_index=sid, start_frame=start,
                                         end_frame=end, minutes=(end-start)/rec.fs/60,
                                         true_frame=local_truth[0][1] if n_true == 1 else np.nan,
                                         true_class=rec.events[local_truth[0][0]][1] if n_true == 1 else 0,
                                         window_true=window_true, window_pred=window_pred,
                                         window_correct=window_true == window_pred if window is not None else np.nan,
                                         window_fp=int(window_true == 0 and window_pred > 0) if window else np.nan,
                                         window_fn=int(window_true > 0 and window_pred == 0) if window else np.nan,
                                         p_left_max=left_score, p_right_max=right_score,
                                         n_true_events=n_true, n_alarms=len(local_alarms), event_tp=n_tp,
                                         event_fp=n_fp, event_fn=n_true-n_tp, too_early_fp=early,
                                         wrong_side_fp=sum(idx in binary_alarms for idx in false_alarms),
                                         event_fp_per_min=n_fp/((end-start)/rec.fs/60),
                                         latency_frames=deltas[0] if len(deltas) == 1 else np.nan,
                                         mean_latency_frames=float(np.mean(deltas)) if deltas else np.nan))
            del model
    segment_scores = pd.DataFrame(segments)
    alarm_details = pd.DataFrame(alarm_rows, columns=[
        'model', 'split', 'file', 'segment_index', 'alarm_frame', 'peak_frame', 'class_id',
        'probability', 'matched', 'matched_true_frame', 'latency_frames',
        'nearest_true_frame', 'delta_to_nearest_true', 'wrong_side', 'too_early',
    ])
    recording_scores = segment_scores.groupby(['model', 'split', 'file', 'user', 'action', 'recording_type'], as_index=False).agg(
        segments=('segment_index', 'size'), minutes=('minutes', 'sum'),
        n_windows=('window_correct', 'count'), window_correct=('window_correct', 'sum'),
        window_fp=('window_fp', 'sum'), window_fn=('window_fn', 'sum'),
        n_true_events=('n_true_events', 'sum'), n_alarms=('n_alarms', 'sum'),
        event_tp=('event_tp', 'sum'), event_fp=('event_fp', 'sum'), event_fn=('event_fn', 'sum'),
        too_early_fp=('too_early_fp', 'sum'), wrong_side_fp=('wrong_side_fp', 'sum'),
    )
    recording_scores['window_accuracy'] = recording_scores.window_correct / recording_scores.n_windows
    recording_scores['event_fp_per_min'] = recording_scores.event_fp / recording_scores.minutes
    return recordings, traces, segment_scores, recording_scores, alarm_details, status


def comparison_matrix(table: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """One row per example, with a group of columns for each model."""
    result = table.set_index(keys + ['model'])[columns].unstack('model')
    return result.swaplevel(0, 1, axis=1).sort_index(axis=1)


def plot_example(recordings, traces, alarm_details, operating_points, file, segment_index=None):
    """Plot shared IMU data and aligned model probability/alarm rows."""
    rec = recordings[file]
    if segment_index is None:
        start, end = 0, len(rec.imu)
    else:
        found = [(a, b) for sid, a, b in rec.segments if sid == segment_index]
        if len(found) != 1:
            raise ValueError(f'{file}: segment_index {segment_index} is absent or excluded')
        start, end = found[0]
    t = rec.time[start:end]
    models = list(operating_points.model)
    fig, axes = plt.subplots(len(models) + 1, 1, sharex=True,
                             figsize=(16, 2.6 * (len(models) + 1)), layout='constrained')
    # Subtract resting acceleration so impacts are visible alongside gyro energy.
    accel = np.linalg.norm(rec.raw[start:end, :3], axis=1)
    axes[0].plot(t, np.abs(accel - np.median(accel)), color='slategray', lw=.8,
                 label='|acc norm - median|')
    axes[0].plot(t, np.linalg.norm(rec.raw[start:end, 3:], axis=1), color='teal', lw=.8,
                 label='gyro norm')
    axes[0].set(ylabel='IMU energy', title=f'{file}  |  segment {segment_index if segment_index is not None else "whole recording"}')
    axes[0].legend(loc='upper right', fontsize=8)
    for ax, (_, choice) in zip(axes[1:], operating_points.iterrows()):
        model = choice['model']
        p = traces[(model, file)]
        ax.plot(t, p[start:end, 1], color='tab:blue', lw=1, label='P(left)')
        ax.plot(t, p[start:end, 2], color='tab:orange', lw=1, label='P(right)')
        ax.axhline(choice['left_threshold'], color='tab:blue', ls=':', lw=.8)
        ax.axhline(choice['right_threshold'], color='tab:orange', ls=':', lw=.8)
        alarms = alarm_details[(alarm_details.model == model) & (alarm_details.file == file)
                                & alarm_details.alarm_frame.between(start, end-1)]
        for cls, color in ((1, 'tab:blue'), (2, 'tab:orange')):
            side = alarms[alarms.class_id == cls]
            if len(side):
                ax.scatter(rec.time[side.alarm_frame.to_numpy(dtype=int)],
                           np.full(len(side), 1.05), color=color, marker='v', s=42,
                           edgecolor=['black' if matched else 'red' for matched in side.matched],
                           linewidth=.8, label=f'{"left" if cls == 1 else "right"} alarm')
        ax.set(ylabel=model, ylim=(-.03, 1.13))
        ax.legend(loc='upper right', fontsize=8, ncol=4)
    for ax in axes:
        for frame, cls in rec.events:
            if start <= frame < end:
                ax.axvline(rec.time[frame], color='tab:blue' if cls == 1 else 'tab:orange',
                           ls='--', lw=.9, alpha=.7)
        if segment_index is None:
            previous_end = 0
            for _, a, b in rec.segments:
                if a > previous_end:
                    ax.axvspan(rec.time[previous_end], rec.time[a], color='0.8', alpha=.25)
                previous_end = b
            if previous_end < len(rec.time):
                ax.axvspan(rec.time[previous_end], rec.time[-1], color='0.8', alpha=.25)
            for _, a, _ in rec.segments[1:]:
                ax.axvline(rec.time[a], color='0.75', lw=.4)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time from recording start (s); dashed = annotated second tap, v = emitted alarm (red edge = FP)')
    return fig, axes
