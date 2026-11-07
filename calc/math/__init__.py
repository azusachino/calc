"""Mathematical foundations and visual geometry labs for LLMs."""

from calc.math.attention import (
    dot_product_and_projection,
    measure_attention_variance,
    measure_softmax_saturation,
)
from calc.math.autograd import (
    StridedView,
    reverse_mode_linear_gelu_step,
    sum_to_shape,
)
from calc.math.convex import (
    soft_threshold,
    solve_lasso_admm,
    verify_ridge_strong_duality,
)
from calc.math.flash_attention import (
    attention_hbm_bytes_comparison,
    online_softmax_1d,
    standard_attention_reference,
    tiled_flash_attention_forward,
)
from calc.math.gradients import (
    compute_autograd,
    measure_residual_gradient_highway,
    quadratic_loss_and_analytical_grad,
    run_gradient_descent,
)
from calc.math.graphs import (
    laplacian_dirichlet_energy,
    measure_gcn_oversmoothing,
    personalized_pagerank,
)
from calc.math.rl import (
    bellman_optimality_step,
    compute_gae,
    grpo_group_loss,
    solve_mdp_value_iteration,
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
    "StridedView",
    "apply_rotation_2d",
    "attention_hbm_bytes_comparison",
    "bellman_optimality_step",
    "check_attention_symmetry",
    "check_rope_relative_identity",
    "compute_autograd",
    "compute_gae",
    "cosine_similarity",
    "demonstrate_subspace_superposition",
    "dot_product_and_projection",
    "grpo_group_loss",
    "laplacian_dirichlet_energy",
    "measure_attention_variance",
    "measure_dimension_orthogonality",
    "measure_gcn_oversmoothing",
    "measure_qkv_orthogonality",
    "measure_residual_gradient_highway",
    "measure_softmax_saturation",
    "online_softmax_1d",
    "personalized_pagerank",
    "quadratic_loss_and_analytical_grad",
    "reverse_mode_linear_gelu_step",
    "rotation_matrix_2d",
    "run_gradient_descent",
    "sample_categorical",
    "shannon_entropy",
    "simulate_multi_head_rope_decay",
    "soft_threshold",
    "solve_lasso_admm",
    "solve_mdp_value_iteration",
    "stable_softmax",
    "standard_attention_reference",
    "sum_to_shape",
    "tiled_flash_attention_forward",
    "top_k_filter",
    "top_p_filter",
    "transform_2d_grid",
    "verify_ridge_strong_duality",
]
