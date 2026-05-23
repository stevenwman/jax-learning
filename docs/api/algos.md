# Algorithms

Eight RL algorithms, each self-contained with no shared base class.

??? note "Why closures instead of methods?"
    JAX's JIT compiler traces Python functions and captures the values they close over. If we used regular methods (`self.update`), JAX would try to trace `self`, which is a mutable Python object — this breaks JIT.

    Instead, all algorithms define JIT'd functions as closures inside `__init__` that capture only JAX-compatible values (networks, configs, constants), then assign them to `self._update`, `self.select_action`, etc. This pattern is standard for JAX RL implementations (Brax, PureJaxRL use the same approach).

---

| Algorithm | Type | Key difference |
|-----------|------|----------------|
| [PPO](#ppo) | On-policy | Clipped surrogate + GAE |
| [PPOContraction](#ppocontraction) | On-policy | PPO + Lipschitz contraction-metric regularizer (research) |
| [SAC](#sac) | Off-policy | Auto-tuned entropy, Gaussian policy |
| [TD3](#td3) | Off-policy | Deterministic policy, twin critics, delayed actor |
| [FastSAC](#fastsac) | Off-policy | C51 distributional critics, UTD 8, TD3-style delayed actor (`policy_delay=4`) |
| [FastTD3](#fasttd3) | Off-policy | C51 distributional critics, UTD 8 |
| [FlashSAC](#flashsac) | Off-policy | Inverted residual blocks + BatchNorm + adaptive reward scaling |
| [TDMPC2](#tdmpc2) | Model-based | Learned world model + MPPI planner, two-hot value/reward |

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

## PPOContraction

```python
from jax_rl.algos.ppo_contraction import PPOContraction
```

PPO + Lipschitz contraction-metric regularizer (Zinage et al.). Adds a learned SPD metric `M(x)` over a chosen state coordinate `c(x)` and a reward augmentation that penalizes `c̈ + αċ` violating contraction. The metric is learned alongside the policy via constraint loss; reward augmentation feeds back into PPO's advantage signal.

**Status: research.** Closed-neutral A/B on `Go2BongoHandstand` (100M, 5-seed: baseline 38.5 vs contraction 37.8). Different failure-mode seeds suggest a distinct strategy, not a superior one. Code is faithful to the reference; nominal-return parity is the floor — the paper's wind-perturbation robustness claim has not been re-tested in this codebase. See [`lessons-learned.md`](../reference/lessons-learned.md) and the closed `.superpowers/plans/archive/2026-04-21-contraction-ppo.md`.

**Constructor**

```python
PPOContraction(
    config: PPOConfig,
    obs_dim: int,
    action_dim: int,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    critic_obs_dim: int | None = None,
)
```

The metric optimizer is built internally from `config.contraction.metric_lr` rather than passed in. `PPOConfig.contraction` (`ContractionConfig`) carries the algorithm-specific knobs: `alpha` (contraction rate), `epsilon` (strict-inequality slack), `penalty_coef`, `constraint_coef`, `metric_hidden`, `metric_lr`. When `config.contraction is None`, falls back to baseline PPO behavior — bit-identical to `PPO` for the same inputs.

**Training entry point:** `scripts/train_ppo_contraction.py`. Bonus contraction-only flags: `--alpha`, `--epsilon`, `--penalty-coef`, `--metric-lr`, `--constraint-coef`, `--metric-hidden`.

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

---

## TDMPC2

```python
from jax_rl.algos.tdmpc2 import TDMPC2State, make_update_step, make_plan_batched
```

Model-based RL with a learned latent world model and an MPPI planner — structurally different from the actor-critic algorithms above. It learns an encoder, latent dynamics, a reward head, and a Q-ensemble, then selects actions at runtime by Model Predictive Path Integral planning: rolling sampled action sequences through the learned dynamics, scoring each by predicted discounted reward plus a terminal Q bootstrap, and iteratively refining a Gaussian toward the elite trajectories. A learned policy prior seeds the planner (and provides a fast `prior` collect mode). Reward and value are predicted as two-hot categorical distributions over `num_bins=101` bins; the latent uses SimNorm. Targets DM Control tasks and PushT — **not** Go2 (no preset; see [Environment Presets](../reference/env-presets.md)).

!!! note "Functional API — no class"
    Unlike the other algorithms, TDMPC2 has no `TDMPC2` class or constructor. Its public surface is an immutable `TDMPC2State` pytree plus factory functions that return JIT-compiled closures: build the networks and optimizers, get an `update_step` and a `plan_fn`, then thread `TDMPC2State` through your loop. See `scripts/train_tdmpc2.py` for the full wiring.

**Factory functions**

`make_update_step(cfg, wm_optimizer, policy_optimizer, *, encoder, dynamics, reward_net, q_ensemble_net, policy_net) → update_step`
: Returns a JIT'd `update_step(state, batch) → (TDMPC2State, metrics)`. One call runs: world-model forward/backward → policy forward/backward → Q-scale update → target EMA (on the Q-ensemble only — there is no target encoder/dynamics/reward/policy) → repacked state.

`make_plan_batched(*, dynamics, reward_net, q_ensemble_net, policy_net) → plan_fn`
: Returns a JIT + vmap'd MPPI planner `plan_fn(plan_params, z0, prev_mean, t0, cfg, keys, eval_mode) → (actions, new_prev_means)`. JIT is load-bearing here (~800 ms cold → <10 ms once warm).

`build_world_model_optimizer(cfg)` / `build_policy_optimizer(cfg)`
: Optimizer builders. The world-model optimizer applies a separate LR scale (`enc_lr_scale`) to the encoder param group via `optax.multi_transform`.

**State**

`TDMPC2State`
: `flax.struct.dataclass` holding every param group (encoder, dynamics, reward, Q-ensemble + its target, policy), the optimizer states, the Q-scale EMA, the planner warm-start `prev_mean`, the RNG key, and the step count.

**Config:** [`TDMPC2Config`](configs.md#tdmpc2config) — built per-env via `make_tdmpc2_config(action_dim, episode_length, task_name)`, which also derives `discount` from the episode length.

**Training entry point:** `scripts/train_tdmpc2.py` (e.g. `uv run python scripts/train_tdmpc2.py --env CheetahRun`). Helpers: `eval_tdmpc2.py` (MPPI vs prior eval), `record_video_tdmpc2.py`, and `check_tdmpc2_determinism.py`.
