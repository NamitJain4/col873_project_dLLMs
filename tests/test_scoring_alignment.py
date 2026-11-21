"""Regression check for Dream's one-token shifted prediction head."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.scoring import masked_prediction_positions


class ScoringAlignmentTests(unittest.TestCase):
    def test_dream_uses_previous_logit_and_llada_uses_current_logit(self):
        targets = torch.tensor([2, 3])
        positions = torch.tensor([2, 3])
        logits = torch.zeros(4, 5)
        logits[1, 2] = 10  # Dream's prediction for target at position 2.
        logits[2, 3] = 10  # Dream's prediction for target at position 3.
        dream = logits[masked_prediction_positions("dream", positions), targets]
        llada = logits[masked_prediction_positions("llada", positions), targets]
        self.assertEqual(dream.tolist(), [10, 10])
        self.assertEqual(llada.tolist(), [0, 0])

    def test_dream_requires_preceding_token(self):
        with self.assertRaises(ValueError):
            masked_prediction_positions("dream", torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
