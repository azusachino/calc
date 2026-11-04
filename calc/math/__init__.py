"""Mathematical foundations and visual geometry labs for LLMs."""

from calc.math.attention import (
    dot_product_and_projection,
    measure_attention_variance,
    measure_softmax_saturation,
)
from calc.math.gradients import (
    compute_autograd,
    measure_residual_gradient_highway,
    quadratic_loss_and_analytical_grad,
    run_gradient_descent,
)
from calc.math.rotations import (
    apply_rotation_2d,
    check_rope_relative_identity,
    rotation_matrix_2d,
    simulate_multi_head_rope_decay,
)
from calc.math.sampling import (
    sample_categorical,
    shannon_entropy,
    stable_softmax,
    top_k_filter,
    top_p_filter,
)
from calc.math.transformations import (
    check_attention_symmetry,
    measure_qkv_orthogonality,
    transform_2d_grid,
)
from calc.math.vectors import (
    cosine_similarity,
    demonstrate_subspace_superposition,
    measure_dimension_orthogonality,
)

__all__ = [
    "apply_rotation_2d",
    "check_attention_symmetry",
    "check_rope_relative_identity",
    "compute_autograd",
    "cosine_similarity",
    "demonstrate_subspace_superposition",
    "dot_product_and_projection",
    "measure_attention_variance",
    "measure_dimension_orthogonality",
    "measure_qkv_orthogonality",
    "measure_residual_gradient_highway",
    "measure_softmax_saturation",
    "quadratic_loss_and_analytical_grad",
    "rotation_matrix_2d",
    "run_gradient_descent",
    "sample_categorical",
    "shannon_entropy",
    "simulate_multi_head_rope_decay",
    "stable_softmax",
    "top_k_filter",
    "top_p_filter",
    "transform_2d_grid",
]
