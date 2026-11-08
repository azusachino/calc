"""Mathematical foundations of Softmax, temperature scaling, and sampling.

Demonstrates:
1. Numerically stable Softmax with temperature scaling.
2. Shannon entropy as a measure of prediction confidence vs uncertainty.
3. Truncation sampling strategies: Greedy, Top-k, and Top-p (Nucleus).
"""

from __future__ import annotations

import math
import torch


def stable_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute numerically stable Softmax over logits with temperature scaling.

    Softmax(z / T)_i = exp((z_i - max(z)) / T) / sum(exp((z_j - max(z)) / T))
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be strictly positive.")

    # Scale by temperature
    scaled = logits / temperature

    # Subtract max for numerical stability (prevents float32 overflow)
    max_val = torch.max(scaled)
    exp_logits = torch.exp(scaled - max_val)
    probs = exp_logits / torch.sum(exp_logits)
    return probs


def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Shannon entropy in bits: H(P) = -sum(p * log2(p))."""
    # Clamp to avoid log2(0)
    clamped = torch.clamp(probs, min=eps)
    entropy = -torch.sum(clamped * torch.log2(clamped))
    return float(entropy.item())


def top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all tokens outside the top k probabilities and renormalize."""
    k = min(max(k, 1), probs.numel())
    topk_vals, topk_indices = torch.topk(probs, k)

    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, topk_indices, topk_vals)
    return filtered / torch.sum(filtered)


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability >= p and renormalize."""
    p = min(max(p, 0.0), 1.0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Find mask of tokens at which the cumulative probability reaches p
    # Keep at least the top-1 token
    cutoff_mask = cumulative_probs >= p
    # Shift right so that the first token that crosses threshold p is kept
    cutoff_mask[1:] = cutoff_mask[:-1].clone()
    cutoff_mask[0] = False

    filtered_sorted = sorted_probs.clone()
    filtered_sorted[cutoff_mask] = 0.0

    # Scatter back to original indices
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices, filtered_sorted)
    return filtered / torch.sum(filtered)


def sample_categorical(probs: torch.Tensor, seed: int = 42) -> int:
    """Sample an index from a discrete probability distribution using torch.multinomial."""
    generator = torch.Generator().manual_seed(seed)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def main() -> None:
    print("=" * 72)
    print("Lesson 06 Benchmark: Softmax, Temperature Scaling, and Nucleus Sampling")
    print("=" * 72)

    # Toy vocabulary of 6 candidate tokens
    vocab = ["the", "a", "cat", "quantum", "banana", "galaxy"]
    # Raw unnormalized model logits
    logits = torch.tensor([8.0, 6.5, 5.0, 1.2, 0.5, 0.1], dtype=torch.float32)

    print("\n1. Impact of Temperature on Probability Distribution and Entropy:")
    print("-" * 72)
    print(f"{'Token':<10} | {'Logit':<8} | {'T=0.2 (Cold)':<14} | {'T=1.0 (Normal)':<14} | {'T=3.0 (Hot)':<14}")
    print("-" * 72)

    p_cold = stable_softmax(logits, temperature=0.2)
    p_norm = stable_softmax(logits, temperature=1.0)
    p_hot = stable_softmax(logits, temperature=3.0)

    for i, token in enumerate(vocab):
        print(
            f"{token:<10} | {logits[i].item():<8.1f} | "
            f"{p_cold[i].item():<14.4f} | "
            f"{p_norm[i].item():<14.4f} | "
            f"{p_hot[i].item():<14.4f}"
        )

    print("-" * 72)
    h_cold = shannon_entropy(p_cold)
    h_norm = shannon_entropy(p_norm)
    h_hot = shannon_entropy(p_hot)
    print(f"{'Entropy (bits)':<21} | {h_cold:<14.2f} | {h_norm:<14.2f} | {h_hot:<14.2f}")
    print("Insight: T -> 0 collapses into deterministic argmax; T -> inf flattens into uniform chaos.")

    print("\n2. Truncation Strategies: Top-k vs. Top-p (Nucleus):")
    print("-" * 72)
    print(f"{'Token':<10} | {'Base Prob':<12} | {'Top-k (k=2)':<14} | {'Top-p (p=0.90)':<14}")
    print("-" * 72)

    p_topk = top_k_filter(p_norm, k=2)
    p_topp = top_p_filter(p_norm, p=0.90)

    for i, token in enumerate(vocab):
        print(
            f"{token:<10} | {p_norm[i].item():<12.4f} | "
            f"{p_topk[i].item():<14.4f} | "
            f"{p_topp[i].item():<14.4f}"
        )
    print("-" * 72)
    print("Insight: Top-k fixes token count (2); Top-p dynamically includes just enough")
    print("tokens to cover 90% cumulative probability mass, cleanly cutting long-tail garbage.")
    print("=" * 72)


if __name__ == "__main__":
    main()
