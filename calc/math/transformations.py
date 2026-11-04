"""Mathematical foundations of linear transformations and QKV projections.

Demonstrates:
1. Matrices as geometric space transformations (moving basis vectors).
2. Breaking attention symmetry: why X X^T is symmetric but (X W_Q)(X W_K)^T is directional.
3. Multi-head perspective decoupling in high-dimensional embeddings.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def transform_2d_grid(matrix: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply a 2x2 linear transformation matrix to a set of 2D coordinates [N, 2].

    Returns:
        Transformed coordinates of shape [N, 2].
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected 2x2 matrix, got {matrix.shape}")
    # vectors: [N, 2], matrix: [2, 2] -> vectors @ matrix.T
    return torch.matmul(vectors, matrix.T)


def check_attention_symmetry(
    dim: int = 64,
    num_tokens: int = 4,
    seed: int = 42,
) -> tuple[float, float]:
    """Measure the symmetry of attention scores with and without W_Q, W_K projections.

    Returns:
        (raw_asymmetry_error, projected_asymmetry_error)
        where 0.0 means perfectly symmetric (A = A^T).
    """
    generator = torch.Generator().manual_seed(seed)

    # Input token embeddings: [num_tokens, dim]
    x = torch.randn(num_tokens, dim, generator=generator)

    # 1. Unprojected attention: S = X @ X^T
    raw_scores = torch.matmul(x, x.T)
    # Symmetry error: ||S - S^T|| / ||S||
    raw_asym = float((torch.linalg.norm(raw_scores - raw_scores.T) / torch.linalg.norm(raw_scores)).item())

    # 2. Projected attention with learned W_Q, W_K
    # Standard Xavier/Kaiming initialization
    w_q = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)
    w_k = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)

    q = torch.matmul(x, w_q)
    k = torch.matmul(x, w_k)
    proj_scores = torch.matmul(q, k.T)
    proj_asym = float((torch.linalg.norm(proj_scores - proj_scores.T) / torch.linalg.norm(proj_scores)).item())

    return raw_asym, proj_asym


def measure_qkv_orthogonality(
    dim: int = 768,
    head_dim: int = 64,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Demonstrate how W_Q, W_K, W_V project a single token into three decoupled perspectives.

    Returns:
        Pairwise cosine similarities (cos_qk, cos_qv, cos_kv).
    """
    generator = torch.Generator().manual_seed(seed)

    # Single token embedding
    token = torch.randn(1, dim, generator=generator)

    # Independent projection weights
    w_q = torch.randn(dim, head_dim, generator=generator) / math.sqrt(dim)
    w_k = torch.randn(dim, head_dim, generator=generator) / math.sqrt(dim)
    w_v = torch.randn(dim, head_dim, generator=generator) / math.sqrt(dim)

    q = F.normalize(torch.matmul(token, w_q), dim=-1).squeeze(0)
    k = F.normalize(torch.matmul(token, w_k), dim=-1).squeeze(0)
    v = F.normalize(torch.matmul(token, w_v), dim=-1).squeeze(0)

    cos_qk = float(torch.dot(q, k).item())
    cos_qv = float(torch.dot(q, v).item())
    cos_kv = float(torch.dot(k, v).item())

    return cos_qk, cos_qv, cos_kv


def main() -> None:
    print("=" * 72)
    print("Lesson 03 Benchmark: Matrices as Transformations and QKV Projections")
    print("=" * 72)

    print("\n1. 2D Space Transformation (Basis Vector Relocation):")
    print("-" * 72)
    # Basis vectors [i_hat, j_hat]
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # A shear + scale matrix
    m = torch.tensor([[2.0, 1.0], [0.0, 1.5]])
    transformed_basis = transform_2d_grid(m, basis)
    print(f"Original i_hat (1, 0) => Transformed: ({transformed_basis[0, 0]:.1f}, {transformed_basis[0, 1]:.1f})")
    print(f"Original j_hat (0, 1) => Transformed: ({transformed_basis[1, 0]:.1f}, {transformed_basis[1, 1]:.1f})")
    print("Notice: The columns of M are exactly the new landing coordinates of the basis!")

    print("\n2. Breaking Attention Symmetry (Why We Need W_Q and W_K):")
    print("-" * 72)
    raw_asym, proj_asym = check_attention_symmetry()
    print(f"Unprojected Attention (X @ X^T) Asymmetry:          {raw_asym:.4f} (Strictly 0.00 => Symmetric)")
    print(f"Projected Attention (Q @ K^T) Asymmetry:            {proj_asym:.4f} (Strongly Directional)")
    print("-" * 72)
    print("Insight: In natural language, 'verb' attending to 'object' must NOT force")
    print("'object' to attend equally back to 'verb'. W_Q and W_K break symmetry!")

    print("\n3. Decoupling Perspectives (Q, K, V for a Single Token):")
    print("-" * 72)
    cos_qk, cos_qv, cos_kv = measure_qkv_orthogonality()
    print(f"Cosine Similarity (Query vs Key):   {cos_qk:+.4f} (Decoupled)")
    print(f"Cosine Similarity (Query vs Value): {cos_qv:+.4f} (Decoupled)")
    print(f"Cosine Similarity (Key vs Value):   {cos_kv:+.4f} (Decoupled)")
    print("-" * 72)
    print("Conclusion: W_Q, W_K, and W_V allow one word to pose a question (Q),")
    print("publish an index (K), and deliver payload (V) completely independently.")
    print("=" * 72)


if __name__ == "__main__":
    main()
