# Algorithms

Six RL algorithms, each self-contained with no shared base class.

??? note "Why closures instead of methods?"
    JAX's JIT compiler traces Python functions and captures the values they close over. If we used regular methods (`self.update`), JAX would try to trace `self`, which is a mutable Python object — this breaks JIT.

    Instead, all algorithms define JIT'd functions as closures inside `__init__` that capture only JAX-compatible values (networks, configs, constants), then assign them to `self._update`, `self.select_action`, etc. This pattern is standard for JAX RL implementations (Brax, PureJaxRL use the same approach).

---

| Algorithm | Type | Key difference |
|-----------|------|----------------|
| [PPO](#ppo) | On-policy | Clipped surrogate + GAE |
| [SAC](#sac) | Off-policy | Auto-tuned entropy, Gaussian policy |
| [TD3](#td3) | Off-policy | Deterministic policy, twin critics, delayed actor |
| [FastSAC](#fastsac) | Off-policy | C51 distributional critics, UTD 8, TD3-style delayed actor (`policy_delay=4`) |
| [FastTD3](#fasttd3) | Off-policy | C51 distributional critics, UTD 8 |
| [FlashSAC](#flashsac) | Off-policy | Inverted residual blocks + BatchNorm + adaptive reward scaling |

---

## PPO

```python
from jax_rl.algos.ppo import PPO
```

On-policy algorithm with clipped surrogate objective and GAE. The only on-policy algorithm in this framework — simpler to tune, lower sample efficiency than off-policy alternatives.

**Constructor**

```python
PPO(
    config: PPOConfig,
    obs_dim: int,
    action_dim: int,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    critic_obs_dim: int | None = None,  # asymmetric critic
)
```

**Methods**

`init(key) → TrainingState`
: Initialize network parameters and optimizer state. Call once before training.

`select_action(state, obs, key, deterministic=False, critic_obs=None) → (action, log_prob, value)`
: Sample action during rollout collection. Returns all three values needed for GAE computation.

`select_action_eval(actor_params, obs) → action`
: Deterministic action for evaluation and deployment. Takes only `actor_params` — no full `TrainingState` or `critic_obs` needed.

`update(state, batch, key, next_obs=None, critic_obs=None, ...) → (TrainingState, metrics)`
: Run one PPO update epoch over the collected rollout batch.

---

## SAC

```python
from jax_rl.algos.sac import SAC
```

Off-policy actor-critic with auto-tuned temperature and reparameterized Gaussian policy. Standard baseline for continuous control.

**Constructor**

```python
SAC(
    config: SACConfig,
    obs_dim: int,
    action_dim: int,
    optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    critic_obs_dim: int | None = None,
)
```

**Methods**

`init(key) → TrainingState`
: Initialize networks, replay buffer pointers, and temperature.

`select_action(actor_params, obs, key, deterministic=False) → action`
: Sample or take the deterministic tanh-squashed action.

`update(state, batch) → (TrainingState, metrics)`
: One SAC gradient step — updates actor, twin critics, and temperature.

`get_q_value(state, obs, action, critic_obs=None) → q`
: Query the first Q-network. Useful for debugging reward shaping.

---

## TD3

```python
from jax_rl.algos.td3 import TD3
```

Deterministic off-policy algorithm with twin critics and delayed actor updates. More stable than DDPG; lower variance than SAC on some tasks.

**Constructor**

```python
TD3(
    config: TD3Config,
    obs_dim: int,
    action_dim: int,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    critic_obs_dim: int | None = None,
)
```

**Methods**

`init(key) → TrainingState`
: Initialize actor, twin critics, and target networks.

`select_action(actor_params, obs, key, deterministic=False) → action`
: Deterministic action with optional exploration noise.

`update(state, batch) → (TrainingState, metrics)`
: One TD3 gradient step. Actor updated every `policy_delay` critic steps.

`get_q_value(state, obs, action, critic_obs=None) → q`
: Query the first Q-network.

---

## FastSAC

```python
from jax_rl.algos.fast_sac import FastSAC
```

SAC with C51 distributional critics, high UTD ratios (8–20), and TD3-style delayed actor+alpha updates (`policy_delay=4` by default). The actor, temperature, *and Polyak target-Q update* are gated by the same `jax.lax.cond` on `update_count`, so all three fire only every `policy_delay` critic steps. This means the **effective per-critic-step target decay is `tau / policy_delay`** — for the default `tau=0.125, policy_delay=4`, each critic step moves targets by ~0.031, not 0.125. Tune `tau` with this gating in mind. Better sample efficiency than standard SAC; preferred for off-policy locomotion training.

**Constructor**

```python
FastSAC(
    config: FastSACConfig,
    obs_dim: int,
    action_dim: int,
    optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    critic_obs_dim: int | None = None,
)
```

**Methods**

`init(key) → TrainingState`
: Initialize actor, C51 critics, and target networks.

`select_action(actor_params, obs, key, deterministic=False) → action`
: Same interface as SAC.

`update(state, batch) → (TrainingState, metrics)`
: One FastSAC update step — runs one critic gradient step per call. The UTD loop (calling `update()` multiple times per env step) is in `jax_rl/training/offpolicy_loop.py` via `grad_updates_per_step`. Actor and alpha are updated only every `policy_delay` calls (TD3-style delay).

`get_q_value(state, obs, action, critic_obs=None) → q`
: Expected Q-value from the C51 distributional critics.

---

## FastTD3

```python
from jax_rl.algos.fast_td3 import FastTD3
```

TD3 with C51 distributional critics and high UTD ratios. Deterministic policy counterpart to FastSAC. The actor and Polyak target-Q update are both gated by `policy_delay` (default `2`), so the **effective per-critic-step target decay is `tau / policy_delay`** — for the default `tau=0.125, policy_delay=2`, each critic step moves targets by ~0.063.

**Constructor**

```python
FastTD3(
    config: FastTD3Config,
    obs_dim: int,
    action_dim: int,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    critic_obs_dim: int | None = None,
)
```

**Methods**

`init(key) → TrainingState`
: Initialize actor, C51 twin critics, and target networks.

`select_action(actor_params, obs, key, deterministic=False) → action`
: Deterministic action with optional exploration noise.

`update(state, batch) → (TrainingState, metrics)`
: One FastTD3 update step with delayed actor updates.

`get_q_value(state, obs, action, critic_obs=None) → q`
: Expected Q-value from the C51 distributional critics.

---

## FlashSAC

```python
from jax_rl.algos.flash_sac import FlashSAC
```

Recent SAC variant combining inverted residual blocks, BatchNorm, weight normalization, and adaptive reward scaling. Eval **284.5 (single seed, post-truncation-fix, default DR — `train_flashsac.py` has no `--reset-mode` flag)** on Go2 joystick at 10M steps. FastSAC with `--reset-mode per_step` reaches **283.8 (single seed, post-fix)**; comparison is not strictly DR-mode-matched, and A/B across seeds is not yet established. Requires more tuning than FastSAC.

See the annotated end-to-end loop in [Reference → Training Loop](../reference/training-loop.md) for how these algorithms plug into the off-policy training script.

**Constructor**

```python
FlashSAC(
    config: FlashSACConfig,
    obs_dim: int,
    action_dim: int,
    optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    critic_obs_dim: int | None = None,
    num_envs: int = 1,  # needed for per-env reward scaling
)
```

**Methods**

`init(key) → TrainingState`
: Initialize flash networks, batch norm stats, and reward scaling state.

`select_action(actor_params, obs, key, deterministic=False, actor_batch_stats=None) → action`
: Action selection with BatchNorm statistics (FlashSAC always uses BatchNorm).

`update(state, batch) → (TrainingState, metrics)`
: One FlashSAC update — includes reward scaling normalization and flash critic updates.

`get_q_value(state, obs, action, critic_obs=None) → q`
: Expected Q-value from the flash distributional critics.
