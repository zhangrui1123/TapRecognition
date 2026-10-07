"""Check the experimental LSTM's causal state and shared training interfaces."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from notebooks_analysis.state_reset_analysis import overlap_stream_logits, predict_modes
from tap_recognition.config import LSTMModelConfig, ModelConfig, TrainConfig
from tap_recognition.inference import OnlineDoubleTapDetector
from tap_recognition.model_factory import build_model
from tap_recognition.model_lstm import CausalCNNLSTM
import train


class LSTMModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_streaming_matches_forward_with_hidden_and_cell_state(self) -> None:
        for hidden, layers, length in ((64, 1, 1), (64, 1, 75), (55, 1, 75), (55, 2, 75)):
            for initial_state in (False, True):
                with self.subTest(hidden=hidden, layers=layers, length=length, initial=initial_state):
                    model = CausalCNNLSTM(lstm_hidden=hidden, lstm_layers=layers).eval()
                    x = torch.randn(2, length, 6)
                    state0 = (
                        torch.randn(layers, 2, hidden), torch.randn(layers, 2, hidden)
                    ) if initial_state else None
                    with torch.no_grad():
                        logits, expected_state = model(x, state0)
                        state, buffer, probs = state0, None, []
                        for t in range(length):
                            # Exercise both accepted single-frame input shapes.
                            frame = x[:, t] if t % 2 else x[:, t:t + 1]
                            probability, state, buffer = model.step(frame, state, buffer)
                            probs.append(probability)
                    torch.testing.assert_close(
                        torch.cat(probs, dim=1), logits.softmax(-1), atol=1e-6, rtol=1e-6
                    )
                    for actual, expected in zip(state, expected_state):
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                    self.assertEqual(buffer.shape, (2, 32, model.receptive_field))

    def test_future_inputs_cannot_change_past_and_chunked_reference_matches(self) -> None:
        model = CausalCNNLSTM().eval()
        x = torch.randn(1, 90, 6)
        changed = x.clone()
        changed[:, 45:] += 20
        with torch.no_grad():
            reference, _ = model(x)
            perturbed, _ = model(changed)
            chunked = overlap_stream_logits(model, x, [1, 7, 29, 53])
        torch.testing.assert_close(reference[:, :45], perturbed[:, :45])
        torch.testing.assert_close(reference, chunked, atol=1e-6, rtol=1e-6)
        recording = SimpleNamespace(imu=x[0].numpy(), bounds=[(0, 45), (45, 90)])
        modes, states = predict_modes(model, recording)
        torch.testing.assert_close(modes['continuous'], reference[0], atol=1e-6, rtol=1e-6)
        self.assertEqual(states['continuous'].shape, (2, model.lstm_hidden))

    def test_checkpoint_round_trip_and_legacy_gru_dispatch(self) -> None:
        for config in (ModelConfig(), LSTMModelConfig(lstm_hidden=55)):
            with self.subTest(config=type(config).__name__):
                cfg = TrainConfig(model=config)
                model = build_model(config.to_model_kwargs()).eval()
                snapshot = {
                    'model_state': model.state_dict(),
                    'model_config': config.to_model_kwargs(),
                    'train_config': cfg.to_dict(),
                }
                if isinstance(config, LSTMModelConfig):
                    snapshot['model_type'] = 'lstm'
                buffer = io.BytesIO()
                torch.save(snapshot, buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, weights_only=True)
                restored = build_model(loaded['model_config'], loaded.get('model_type')).eval()
                restored.load_state_dict(loaded['model_state'])
                self.assertIsInstance(restored, type(model))
                x = torch.randn(1, 40, 6)
                with torch.no_grad():
                    torch.testing.assert_close(restored(x)[0], model(x)[0])
                if isinstance(restored, CausalCNNLSTM):
                    self.assertEqual(loaded['train_config']['model']['lstm_hidden'], 55)

    def test_loss_backward_and_shared_evaluation(self) -> None:
        model = CausalCNNLSTM(lstm_hidden=55)
        labels = torch.zeros(2, 40, 3)
        labels[..., 0] = 1
        labels[0, 20] = torch.tensor([0., 1., 0.])
        x = torch.randn(2, 40, 6)
        window_labels = torch.tensor([1, 0])
        logits, _ = model(x)
        loss = train.frame_soft_ce(logits, labels, window_label=window_labels)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        samples = [dict(imu=x[i], frame_labels=labels[i], window_label=window_labels[i])
                   for i in range(2)]
        metrics = train.evaluate(model, DataLoader(samples, batch_size=2), torch.device('cpu'), .5)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertEqual(sum(metrics[k] for k in ('tp', 'fp', 'fn', 'tn')), 2)

    def test_online_detector_keeps_and_resets_both_lstm_states(self) -> None:
        model = CausalCNNLSTM(lstm_hidden=55).eval()
        detector = OnlineDoubleTapDetector(model)
        detector.process_sample(np.ones(6, dtype=np.float32))
        self.assertIsInstance(detector._h, tuple)
        self.assertEqual([tuple(s.shape) for s in detector._h], [(1, 1, 55), (1, 1, 55)])
        detector.reset()
        self.assertIsNone(detector._h)
        self.assertIsNone(detector._cnn_buffer)

    def test_training_cli_selects_lstm_without_running_training(self) -> None:
        for args, expected_type in (([], ModelConfig), (
            ['--recurrent-type', 'lstm', '--recurrent-hidden', '55', '--recurrent-layers', '2'],
            LSTMModelConfig,
        )):
            with self.subTest(args=args), patch('sys.argv', ['train.py', *args]), \
                    patch.object(train, 'train') as run_training:
                train.main()
                config = run_training.call_args.args[0].model
                self.assertIsInstance(config, expected_type)
                if isinstance(config, LSTMModelConfig):
                    self.assertEqual((config.lstm_hidden, config.lstm_layers), (55, 2))

    def test_notebook_setup_and_delayed_validation_without_training(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / 'main_notebooks/training_lstm.ipynb'
        notebook = json.loads(notebook_path.read_text())
        labels = torch.zeros(40, 3)
        labels[:, 0] = 1
        labels[20] = torch.tensor([0., 1., 0.])
        samples = [dict(imu=torch.randn(40, 6), frame_labels=labels, window_label=torch.tensor(1))]
        namespace = {}
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with patch.object(train, 'build_datasets', return_value=(samples, samples)), \
                    patch.object(Path, 'mkdir'), patch('torch.cuda.is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])
                    source = ''.join(cell['source'])
                    compiled = compile(source, f'{notebook_path.name}:cell{index}', 'exec')
                    if not source.startswith('history = []'):
                        exec(compiled, namespace)
            self.assertIsInstance(namespace['model'], CausalCNNLSTM)
            namespace['LABEL_DELAY_FRAMES'] = 8
            metrics = namespace['evaluate'](namespace['model'], namespace['val_loader'],
                                            torch.device('cpu'), .5)
            self.assertTrue(np.isfinite(metrics['loss']))
        finally:
            torch.use_deterministic_algorithms(was_deterministic)


if __name__ == '__main__':
    unittest.main()
