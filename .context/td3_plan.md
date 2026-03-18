# TD3 Implementation Plan

## Overview
Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018).
Deterministic off-policy actor-critic. Simpler than SAC — no entropy, no alpha.

Reference: Spinning Up, CleanRL, FastTD3 paper.
No Brax/Playground reference (they only have PPO + SAC).

## TD3 vs SAC — Implementation Diff

| Aspect | SAC | TD3 | Change needed |
|--------|-----|-----|---------------|
| Actor | Stochastic (GaussianHead) | Deterministic (DeterministicHead) | New head |
| Exploration | Entropy maximization (alpha) | Additive Gaussian noise at collection | Different select_action |
| Alpha/temperature | Auto-tuned log_alpha | None | Remove alpha loss entirely |
| Actor update freq | Every step | Every `policy_delay` steps (default 2) | Add counter |
| Target next action | Sample from current policy | Deterministic + clipped noise smoothing | Different target computation |
| Critic loss | Same | Same | Reuse |
| Polyak update | Same | Same (but only when actor updates) | Minor tweak |
| Q aggregation | min(Q1, Q2) | min(Q1, Q2) | Same |

## Losses

### Critic loss (same as SAC, minus entropy term)
```
a' = tanh(μ_target(s') + clip(N(0, σ), -c, c))    # smoothed target action
y  = r + γ * (1 - done) * min(Q1_target(s', a'), Q2_target(s', a'))
L_Q = 0.5 * mean((Q1(s,a) - y)^2 + (Q2(s,a) - y)^2)
```
Note: no `- alpha * log_prob` in the target. The entropy term is gone entirely.

### Actor loss (simpler than SAC)
```
a = μ(s)                        # deterministic, no sampling
L_actor = -mean(Q1(s, a))       # maximize Q1 only (not min — original TD3 uses Q1)
```
Note: original TD3 paper uses only Q1 for actor gradient, not min(Q1, Q2).
This avoids the issue of min creating discontinuous gradients for the actor.

### No alpha loss
TD3 has no temperature parameter.

## Key Design Decisions

### Policy delay
Update actor (and target networks) only every `policy_delay` critic updates.
Rationale: let the critic stabilize before updating the policy.
Implementation: track a step counter, `if counter % policy_delay == 0`.

### Target policy smoothing
Add clipped noise to next actions in critic target:
```python
noise = jnp.clip(jax.random.normal(key, next_action.shape) * target_noise_std, -noise_clip, noise_clip)
next_action = jnp.clip(target_actor(next_obs) + noise, -1.0, 1.0)
```
This regularizes the critic to be smooth w.r.t. actions — prevents exploiting Q peaks.

### Exploration noise (at collection time)
During training, add Gaussian noise to deterministic actions:
```python
action = actor(obs) + N(0, exploration_noise_std)
action = clip(action, -1, 1)
```
Not part of the loss — just for data collection.

## Reuse from SAC

- `QHead` — identical
- `ReplayBuffer` — identical
- `train_sac.py` structure → `train_td3.py` (copy + simplify)
- Polyak soft update — identical
- Truncation masking in critic loss — identical
- Checkpoint infrastructure — identical
- SAC presets pattern → TD3 presets

## New Files

1. `jax_rl/networks/heads/deterministic.py` — DeterministicHead: features → Dense → tanh
2. `jax_rl/configs/td3_config.py` — TD3Config dataclass
3. `jax_rl/algos/td3.py` — TD3 algorithm
4. `train_td3.py` — training script

## Config

```python
@dataclass
class TD3Config:
    tau: float = 0.005
    policy_delay: int = 2
    target_noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise_std: float = 0.1
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    batch_size: int = 256
    grad_updates_per_step: int = 1      # vanilla TD3 does 1:1 ratio
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    q_layer_norm: bool = False           # vanilla TD3 doesn't use it; FastTD3 does
```

## Benchmark Targets
- CheetahRun: >=700 at 5M steps (from framework plan, CleanRL reference)
- WalkerWalk: validate against SAC
- HumanoidRun: validate against SAC
