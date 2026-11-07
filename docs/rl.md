# Bellman contraction, GAE, and DeepSeek GRPO

This document accompanies `calc/math/rl.py` (paired with **Stanford CS234: Reinforcement Learning**) and implements tabular Markov Decision Process value iteration, Generalized Advantage Estimation (`GAE`), and Group Relative Policy Optimization (`GRPO`).

Run the companion benchmark:

```bash
uv run calc/math/rl.py
```

---

## 1. Bellman optimality $\gamma$-contraction

The Bellman optimality operator $\mathcal{T}^*$ over state values $V \in \mathbb{R}^{|\mathcal{S}|}$ is defined by:

$$(\mathcal{T}^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V(s') \right]$$

Because $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$ and $\sum_{s'} P(s'\mid s,a) = 1$, $\mathcal{T}^*$ is a $\gamma$-contraction in the supremum norm $\|\cdot\|_\infty$:

$$\|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$$

By Banach's Fixed-Point Theorem, iterating $V_{k+1} = \mathcal{T}^* V_k$ converges geometrically to the unique optimal value function $V^*$.

---

## 2. Generalized Advantage Estimation ($\operatorname{GAE}(\gamma, \lambda)$)

Using 1-step temporal-difference residuals $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, GAE computes an exponentially weighted sum of $n$-step advantage estimates in a single backward pass:

$$\hat{A}_t^{\operatorname{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda)\hat{A}_{t+1}^{\operatorname{GAE}(\gamma, \lambda)}$$

---

## 3. Critic-free Group Relative Policy Optimization (`GRPO`)

For reasoning models with verifiable rewards (`RLVR`), `GRPO` eliminates the value critic network by sampling a group of $G$ outputs $\{o_1, \dots, o_G\}$ per prompt and normalizing rewards within the group:

$$A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \varepsilon}$$

Regularization uses Schulman's unbiased, strictly non-negative $k_3$ KL divergence estimator against $\pi_{\text{ref}}$:

$$\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1 \ge 0$$
