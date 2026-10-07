"""Checks for per-example replay alarm semantics."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from notebooks_analysis.model_comparison_examples import (
    emit_alarms_pair,
    load_examples,
    match_alarms,
    replay_examples,
)
from tap_recognition.config import DataConfig, LabelConfig, LSTMModelConfig, ModelConfig, TrainConfig
from tap_recognition.dataset import RecordedIMUDataset
from tap_recognition.model_factory import build_model


class ModelComparisonExamplesTests(unittest.TestCase):
    def test_excluded_gap_resets_lookahead_and_refractory(self):
        probabilities = np.zeros((12, 3), dtype=float)
        probabilities[:, 0] = 1
        probabilities[3, 1] = .9
        probabilities[4, 1] = .99  # Excluded frame must not affect the candidate peak.
        probabilities[7, 2] = .9
        alarms = emit_alarms_pair(probabilities, .8, .8, 1, 50, [(0, 4), (6, 12)])
        self.assertEqual([(alarm['alarm_frame'], alarm['peak_frame'], alarm['class_id'])
                          for alarm in alarms], [(8, 7, 2)])

    def test_strict_timing_and_wrong_side_are_unmatched(self):
        truth = [(100, 1)]
        alarms = [dict(alarm_frame=98, class_id=1),
                  dict(alarm_frame=100, class_id=2)]
        self.assertEqual(match_alarms(truth, alarms, 1, 30), [])
        self.assertEqual(match_alarms(truth, alarms, 1, 30, class_aware=False), [(0, 1)])
        alarms.append(dict(alarm_frame=101, class_id=1))
        self.assertEqual(match_alarms(truth, alarms, 1, 30), [(0, 2)])

    def _write_recordings(self, root):
        """Separate splits with different sides and one excluded training segment."""
        names = {}
        for split, side in [('train', 1), ('valid', 2)]:
            directory = root / split
            directory.mkdir()
            name = f'imu_fixture_knock_twice_{"left" if side == 1 else "right"}_{split}.csv'
            names[split] = name
            frames = np.arange(36)
            pd.DataFrame({
                'timestamp_ms': frames * 10,
                'segment_index': np.repeat([0, 1, 2], 12),
                'user_name': 'fixture',
                'action_label': f'knock_twice_{"left" if side == 1 else "right"}',
                **{channel: np.sin(frames / 3) for channel in
                   ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')},
            }).to_csv(directory / name, index=False)
            (directory / name).with_suffix('.txt').write_text(f'6, {side}\n30, {side}\n')
        registry = root / 'exclusions.csv'
        registry.write_text(f'source_file,segment_index,reason\n{names["train"]},1,fixture\n')
        data = asdict(DataConfig(
            train_dir='train', val_dir='valid', window_samples=12,
            highpass=False, trigger_context_samples=6,
            session_exclusions_file='exclusions.csv',
        ))
        return data, asdict(LabelConfig()), names

    def test_split_loader_uses_correct_files_seeds_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, labels, names = self._write_recordings(root)
            for split, expected_seed, expected_segments, expected_class in (
                ('train', 42, [0, 2], 1), ('valid', 43, [0, 1, 2], 2),
            ):
                with self.subTest(split=split), patch(
                    'notebooks_analysis.model_comparison_examples.RecordedIMUDataset',
                    wraps=RecordedIMUDataset,
                ) as dataset_class:
                    recordings, digest = load_examples(root, data, labels, 42, split=split)
                    self.assertEqual(set(recordings), {names[split]})
                    recording = recordings[names[split]]
                    self.assertEqual([sid for sid, _, _ in recording.segments], expected_segments)
                    self.assertEqual([cls for _, cls in recording.events], [expected_class] * 2)
                    self.assertEqual(set(recording.windows), {0, 2})
                    self.assertEqual(dataset_class.call_args.kwargs['seed'], expected_seed)
                    self.assertEqual(len(digest), 64)
            # The existing no-split call must continue to mean validation.
            recordings, _ = load_examples(root, data, labels, 42)
            self.assertEqual(set(recordings), {names['valid']})

    def test_training_replay_keeps_validation_operating_points_and_split_identity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, labels, names = self._write_recordings(root)
            operating_points = []
            for kind, model_config in (
                ('gru', ModelConfig(cnn_channels=4, gru_hidden=8, kernel_size=3,
                                    dilations=(1,), dropout=0.0)),
                ('lstm', LSTMModelConfig(cnn_channels=4, lstm_hidden=8, kernel_size=3,
                                       dilations=(1,), dropout=0.0)),
            ):
                model = build_model(model_config.to_model_kwargs())
                # Fixed output guarantees alarms; no optimization or training.
                with torch.no_grad():
                    model.head[-1].weight.zero_()
                    model.head[-1].bias.copy_(torch.tensor([0., 8., 0.]))
                cfg = TrainConfig(model=model_config)
                snapshot = {
                    'model_type': kind, 'model_config': model_config.to_model_kwargs(),
                    'model_state': model.state_dict(),
                    'train_config': {**cfg.to_dict(), 'data': data, 'labels': labels},
                }
                directory = root / kind
                directory.mkdir()
                checkpoint_path = directory / 'best.pt'
                torch.save(snapshot, checkpoint_path)
                operating_points.append(dict(
                    model=kind, folder=f'{kind}/threshold_comparison',
                    checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    validation_sha256='historical-validation-hash', policy='instant',
                    left_threshold=.9, right_threshold=.95, refractory_sec=.05,
                    early_allowance_frames=1, max_late_frames=30,
                ))
            choices = pd.DataFrame(operating_points)
            original_choices = choices.copy(deep=True)
            for split in ('train', 'valid'):
                with self.subTest(split=split):
                    recordings, traces, segments, files, alarms, status = replay_examples(
                        root, choices, split=split,
                    )
                    self.assertEqual(set(recordings), {names[split]})
                    for table in (segments, files, alarms, status):
                        self.assertEqual(set(table.split), {split})
                        self.assertEqual(set(table.model), {'gru', 'lstm'})
                    self.assertEqual(set(segments.file), {names[split]})
                    for metric in ('event_tp', 'event_fp', 'event_fn'):
                        self.assertEqual(segments[metric].sum(), files[metric].sum())
                    if split == 'train':
                        self.assertEqual(set(segments.segment_index), {0, 2})
                        self.assertIn('current_training_sha256', status)
                        self.assertNotIn('matches_export', status)
                        self.assertNotIn('export_validation_sha256', status)
                        for probabilities in traces.values():
                            np.testing.assert_array_equal(probabilities[12:24, 0], 1.)
                            np.testing.assert_array_equal(probabilities[12:24, 1:], 0.)
                    else:
                        self.assertEqual(set(segments.segment_index), {0, 1, 2})
                        self.assertIn('current_validation_sha256', status)
                        self.assertTrue((status.matches_export == False).all())
            pd.testing.assert_frame_equal(choices, original_choices)

    def test_invalid_split_is_rejected_before_loading_checkpoints(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported inspection split'):
            replay_examples(Path('.'), pd.DataFrame(), split='test')
        with self.assertRaisesRegex(ValueError, 'Unsupported inspection split'):
            load_examples(Path('.'), {}, {}, 42, split='test')


if __name__ == '__main__':
    unittest.main()
