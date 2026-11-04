import io
import unittest
from contextlib import redirect_stdout

import torch

from calc.core.model import LanguageModel
from calc.core.tokenizer import CharTokenizer
from calc.core.train import inspect_step


class TrainingInspectionTests(unittest.TestCase):
    def test_inspect_step_reports_shapes_loss_and_gradients(self):
        torch.manual_seed(0)
        tokenizer = CharTokenizer("abc")
        model = LanguageModel(tokenizer.vocab_size, 4, 4, 2, 1, 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.tensor([[0, 1]])
        y = torch.tensor([[1, 2]])
        output = io.StringIO()

        with redirect_stdout(output):
            inspect_step(model, optimizer, x, y, tokenizer)

        report = output.getvalue()
        self.assertIn("input:  shape=(1, 2)", report)
        self.assertIn("target: shape=(1, 2)", report)
        self.assertIn("logits: shape=(1, 2, 3)", report)
        self.assertIn("loss:", report)
        self.assertIn("gradient token_embedding.weight:", report)
        self.assertIn("gradient output.weight:", report)
        self.assertIn("one optimizer step applied; checkpoint not written", report)


if __name__ == "__main__":
    unittest.main()
