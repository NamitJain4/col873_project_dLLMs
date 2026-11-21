"""A mask placeholder must never be emitted as a completed token."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator, visible_token_ids


class MaskPreferredModel:
    config = SimpleNamespace(mask_token_id=4)

    def __call__(self, tokens):
        logits = torch.zeros((*tokens.shape, 6))
        logits[..., 4] = 10.0  # The invalid mask would win without filtering.
        logits[..., 3] = 9.0
        return SimpleNamespace(logits=logits)


class EosPreferredModel:
    config = SimpleNamespace(mask_token_id=4)

    def __call__(self, tokens):
        logits = torch.zeros((*tokens.shape, 6))
        logits[..., 5] = 10.0
        return SimpleNamespace(logits=logits)


class LLaDAMaskCompletionTests(unittest.TestCase):
    def test_optional_eos_truncation(self):
        ids = [10, 11, 5, 12, 5]
        self.assertEqual(visible_token_ids(ids, 5, True), [10, 11])
        self.assertEqual(visible_token_ids(ids, 5, False), ids)

    def test_mask_placeholder_cannot_be_selected(self):
        generator = Generator.__new__(Generator)
        generator.model = MaskPreferredModel()
        generator.tokenizer = SimpleNamespace(eos_token_id=5)
        result = generator._llada_generate(torch.tensor([[1]]), length=2, steps=1,
                                            deprioritize_eos=True)
        self.assertEqual(result.tolist(), [[1, 3, 3]])

    def test_filled_eos_positions_stay_excluded(self):
        generator = Generator.__new__(Generator)
        generator.model = EosPreferredModel()
        generator.tokenizer = SimpleNamespace(eos_token_id=5)
        result = generator._llada_generate(torch.tensor([[1]]), length=2, steps=2,
                                            deprioritize_eos=True)
        self.assertEqual(result.tolist(), [[1, 5, 5]])


if __name__ == "__main__":
    unittest.main()
