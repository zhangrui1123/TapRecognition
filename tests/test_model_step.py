"""Streaming inference must reproduce the sequence model used for validation."""

import unittest
from pathlib import Path

import torch

from tap_recognition.model import CausalCNNGRU


class ModelStepTest(unittest.TestCase):
    def test_step_matches_forward_at_every_frame(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        for layers, length in ((1, 1), (1, 8), (1, 75), (3, 75)):
            with self.subTest(layers=layers, length=length):
                model = CausalCNNGRU(gru_layers=layers).eval()
                x = torch.randn(2, length, model.input_dim)
                h0 = torch.randn(layers, 2, model.gru_hidden)

                with torch.no_grad():
                    logits, final_h = model(x, h0)
                    expected = logits.softmax(dim=-1)
                    h, buffer = h0, None
                    actual_frames = []
                    for t in range(length):
                        prob, h, buffer = model.step(x[:, t], h, buffer)
                        actual_frames.append(prob)

                torch.testing.assert_close(
                    torch.cat(actual_frames, dim=1), expected, atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(h, final_h, atol=1e-6, rtol=1e-6)
                self.assertEqual(buffer.shape, (2, 32, model.receptive_field))

    def test_trained_checkpoint_matches_forward(self) -> None:
        torch.set_num_threads(1)
        ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints/best.pt"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CausalCNNGRU(**checkpoint["model_config"]).eval()
        model.load_state_dict(checkpoint["model_state"])
        torch.manual_seed(17)
        x = torch.randn(1, 75, model.input_dim)

        with torch.no_grad():
            logits, final_h = model(x)
            h, buffer = None, None
            probs = []
            for t in range(x.shape[1]):
                prob, h, buffer = model.step(x[:, t : t + 1], h, buffer)
                probs.append(prob)

        torch.testing.assert_close(
            torch.cat(probs, dim=1), logits.softmax(dim=-1), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(h, final_h, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
