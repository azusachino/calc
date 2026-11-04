"""Sliding-window next-token dataset used by the GPT-2 tokenizer exercise."""

import torch
from torch.utils.data import Dataset

from calc.core.tokenizer import GPT2BPETokenizer


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create input/target windows where targets are shifted one token forward."""

    def __init__(self, text: str, tokenizer: GPT2BPETokenizer, max_length: int, stride: int):
        if max_length < 1 or stride < 1:
            raise ValueError("max_length and stride must be positive")
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= max_length:
            raise ValueError("text must encode to more tokens than max_length")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride

    def __len__(self) -> int:
        return (len(self.token_ids) - self.max_length - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        inputs = self.token_ids[start : start + self.max_length]
        targets = self.token_ids[start + 1 : start + self.max_length + 1]
        return inputs, targets
