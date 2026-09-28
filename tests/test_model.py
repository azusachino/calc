import unittest

import torch

from calc.model import LanguageModel


class LanguageModelTests(unittest.TestCase):
    def test_logits_have_one_distribution_per_input_token(self):
        model = LanguageModel(7, 8, 16, 4, 1, 0.0)
        logits, loss = model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
        self.assertEqual(tuple(logits.shape), (1, 3, 7))
        self.assertIsNotNone(loss)

    def test_causal_attention_cannot_see_future_tokens(self):
        torch.manual_seed(0)
        model = LanguageModel(7, 8, 16, 4, 1, 0.0).eval()
        first, _ = model(torch.tensor([[1, 2, 3]]))
        changed_future, _ = model(torch.tensor([[1, 2, 6]]))
        torch.testing.assert_close(first[:, :2], changed_future[:, :2])


if __name__ == "__main__":
    unittest.main()
