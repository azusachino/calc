"""Tests for Online Softmax and tiled FlashAttention-2 forward pass."""

import unittest
import torch
import torch.nn.functional as F
from calc.math.flash_attention import (
    attention_hbm_bytes_comparison,
    online_softmax_1d,
    standard_attention_reference,
    tiled_flash_attention_forward,
)


class FlashAttentionTests(unittest.TestCase):
    def test_online_softmax_matches_standard(self):
        x = torch.tensor([12.0, -3.5, 14.2, 14.1, 0.0, 8.8, -10.0, 5.5, 11.1])
        probs_online, m_val, _ = online_softmax_1d(x, block_size=3)
        probs_ref = F.softmax(x, dim=0)

        self.assertAlmostEqual(m_val, 14.2, places=5)
        self.assertAlmostEqual(float(probs_online.sum().item()), 1.0, places=6)
        self.assertLess(float(torch.max(torch.abs(probs_online - probs_ref)).item()), 1e-6)

    def test_tiled_flash_attention_non_causal_and_causal(self):
        gen = torch.Generator().manual_seed(99)
        q = torch.randn(19, 12, generator=gen)
        k = torch.randn(19, 12, generator=gen)
        v = torch.randn(19, 12, generator=gen)

        for causal in (False, True):
            out_flash = tiled_flash_attention_forward(q, k, v, block_q=5, block_kv=4, causal=causal)
            out_ref = standard_attention_reference(q, k, v, causal=causal)
            max_err = float(torch.max(torch.abs(out_flash - out_ref)).item())
            self.assertLess(max_err, 1e-5)

    def test_hbm_bytes_reduction(self):
        std_b, flash_b, ratio = attention_hbm_bytes_comparison(seq_len=4096, d_head=128)
        self.assertGreater(std_b, flash_b)
        self.assertGreater(ratio, 15.0)


if __name__ == "__main__":
    unittest.main()
