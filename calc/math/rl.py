"""Markov Decision Processes, Bellman contraction, GAE, and DeepSeek GRPO.

Companion lab for Stanford CS234 (Reinforcement Learning):
1. Tabular Bellman Optimality operator T* and empirical gamma-contraction verification.
2. Generalized Advantage Estimation (GAE) bias-variance recurrence.
3. Critic-free Group Relative Policy Optimization (GRPO) with unbiased Schulman k3 KL.
"""

from __future__ import annotations

import torch


def bellman_optimality_step(
    transitions: torch.Tensor,
    rewards: torch.Tensor,
    v_state: torch.Tensor,
    gamma: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one step of the Bellman Optimality operator (T* V)(s) = max_a [R(s,a) + gamma * P(s'|s,a) V(s')].

    Args:
        transitions: Tensor of shape (S, A, S) where row sums over s' equal 1.
        rewards: Tensor of shape (S, A).
        v_state: Value vector of shape (S,).
        gamma: Discount factor in [0, 1).

    Returns:
        (next_values, greedy_actions)
    """
    q_values = rewards + gamma * torch.einsum("sas,s->sa", transitions, v_state)
    next_v, greedy_a = torch.max(q_values, dim=1)
    return next_v, greedy_a


def solve_mdp_value_iteration(
    transitions: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_steps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run Value Iteration to fixed point V* and return (V*, greedy_policy, num_steps)."""
    num_states = transitions.shape[0]
    v = torch.zeros(num_states, dtype=rewards.dtype, device=rewards.device)
    policy = torch.zeros(num_states, dtype=torch.long, device=rewards.device)

    for step in range(1, max_steps + 1):
        v_next, policy = bellman_optimality_step(transitions, rewards, v, gamma=gamma)
        bellman_residual = float(torch.max(torch.abs(v_next - v)).item())
        v = v_next
        if bellman_residual < tol:
            return v, policy, step

    return v, policy, max_steps


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation A_t^{GAE(gamma, lam)} and target returns.

    Returns:
        (advantages, returns_to_go)
    """
    horizon = rewards.numel()
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in range(horizon - 1, -1, -1):
        v_next = next_value if t == horizon - 1 else float(values[t + 1].item())
        delta_t = float(rewards[t].item()) + gamma * v_next - float(values[t].item())
        gae = delta_t + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def grpo_group_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    group_rewards: torch.Tensor,
    eps_clip: float = 0.2,
    beta_kl: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DeepSeek GRPO group-normalized clipped objective and unbiased k3 KL penalty.

    Args:
        log_probs: Current policy log-probabilities for G sampled completions, shape (G,).
        old_log_probs: Rollout policy log-probabilities, shape (G,).
        ref_log_probs: Reference policy log-probabilities, shape (G,).
        group_rewards: Verifier rewards for the G completions, shape (G,).

    Returns:
        (total_loss, normalized_advantages, k3_kl_per_sample)
    """
    mean_r = torch.mean(group_rewards)
    std_r = torch.std(group_rewards, unbiased=False)
    advantages = (group_rewards - mean_r) / (std_r + 1e-8)

    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    policy_surrogate = torch.minimum(surr1, surr2)

    # Schulman's unbiased non-negative k3 KL estimator: exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1
    log_ratio_ref = ref_log_probs - log_probs
    k3_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    loss = -torch.mean(policy_surrogate - beta_kl * k3_kl)
    return loss, advantages, k3_kl


def main() -> None:
    print("=" * 72)
    print("Stanford CS234 Lab: Bellman Contraction, GAE(gamma, lambda), and GRPO")
    print("=" * 72)

    transitions = torch.tensor(
        [
            [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]],
            [[0.0, 0.7, 0.3], [0.0, 0.2, 0.8]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([[1.0, 0.5], [2.0, 4.0], [0.0, 0.0]], dtype=torch.float32)

    v_star, policy, steps = solve_mdp_value_iteration(transitions, rewards, gamma=0.9)
    print(f"Value Iteration converged in {steps} steps: V*={v_star.tolist()}, pi*={policy.tolist()}")

    r_group = torch.tensor([1.0, 0.0, 1.0, 0.0])
    lp = torch.tensor([-1.1, -1.5, -0.9, -1.3])
    old_lp = torch.tensor([-1.2, -1.4, -1.0, -1.2])
    ref_lp = torch.tensor([-1.3, -1.3, -1.1, -1.1])
    loss, adv, kl = grpo_group_loss(lp, old_lp, ref_lp, r_group)
    print(f"GRPO Group Advantages: {adv.tolist()} | Mean k3 KL: {float(kl.mean().item()):.4f} | Loss: {float(loss.item()):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
