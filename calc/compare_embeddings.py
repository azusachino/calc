"""Compare embedding lookup and positional encoding mechanisms."""

import torch
from torch import nn

from calc import config
from calc.embeddings import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from calc.tokenizer import CharTokenizer


def main() -> None:
    print("=== 1. Token Embedding & Tensor Broadcasting ===")
    sample_text = ["To be, or not to", "that is the ques"]
    tokenizer = CharTokenizer("".join(sample_text) + "abcdefghijklmnopqrstuvwxyz .")
    vocab_size = tokenizer.vocab_size
    embedding_size = config.EMBEDDING_SIZE  # e.g., 128
    block_size = config.BLOCK_SIZE          # e.g., 64

    # Encode into batch: [B, T]
    batch = torch.tensor([tokenizer.encode(s) for s in sample_text], dtype=torch.long)
    b, t = batch.shape
    print(f"Input batch shape:       [B={b}, T={t}]")
    print(f"Sample 0 tokens:         {batch[0].tolist()}")

    # Token lookup
    token_emb = nn.Embedding(vocab_size, embedding_size)
    tok_vectors = token_emb(batch)
    print(f"Token embeddings shape:  {list(tok_vectors.shape)} (from nn.Embedding)")

    # Learned position lookup
    learned_pe = LearnedPositionalEmbedding(block_size, embedding_size)
    learned_pos = learned_pe(t)
    print(f"Position vector shape:   {list(learned_pos.shape)} (for T={t})")

    # Broadcasting addition: [B, T, D] + [T, D] -> [B, T, D]
    fused_input = tok_vectors + learned_pos
    print(f"Fused input tensor:      {list(fused_input.shape)} (via PyTorch broadcasting)\n")

    print("=== 2. Positional Encodings: Properties & Boundaries ===")
    sinusoidal = SinusoidalPositionalEncoding(embedding_size, max_length=block_size)
    sin_pos = sinusoidal(t)
    print(f"Sinusoidal shape:        {list(sin_pos.shape)}")
    print(f"  pos=0, dim=0 (sin):    {sin_pos[0, 0].item():.4f} (expected: 0.0)")
    print(f"  pos=0, dim=1 (cos):    {sin_pos[0, 1].item():.4f} (expected: 1.0)")

    rope = RotaryPositionalEmbedding(head_size=32, max_length=block_size)
    dummy_q = torch.randn(1, 4, t, 32)  # [B, num_heads, T, head_size]
    rotated_q = rope(dummy_q)
    print(f"RoPE transformed Q:      {list(rotated_q.shape)} (applied in attention heads)\n")

    print("=== 3. Relative Distance Inner-Product Decay ===")
    print("Testing dot-product correlation between position 0 and position t (same base vector):")
    base_vec = torch.ones(1, 32)
    base_vec = base_vec / base_vec.norm()

    # RoPE relative dot product
    q0 = rope(base_vec.unsqueeze(0))  # pos 0 -> [1, 1, 32]
    distances = [0, 1, 2, 4, 8, 16, 32]
    print(f"{'Distance (t)':<15} {'RoPE Dot-Product':<20} {'Sinusoidal Cos-Sim':<20}")
    for d in distances:
        if d >= block_size:
            continue
        # RoPE at pos d
        full_seq = base_vec.repeat(d + 1, 1).unsqueeze(0)  # [1, d+1, 32]
        rotated_seq = rope(full_seq)
        qd = rotated_seq[0, d]
        q0_vec = rotated_seq[0, 0]
        rope_dot = torch.dot(q0_vec, qd).item()

        # Sinusoidal similarity between PE(0) and PE(d)
        sin_pe = sinusoidal(d + 1)
        sin_sim = torch.cosine_similarity(sin_pe[0:1], sin_pe[d : d + 1]).item()

        print(f"{d:<15} {rope_dot:<20.4f} {sin_sim:<20.4f}")


if __name__ == "__main__":
    main()
