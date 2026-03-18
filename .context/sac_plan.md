# SAC Implementation Plan

## Overview

Soft Actor-Critic (SAC) — off-policy maximum entropy RL. Target: match MuJoCo Playground's
SAC figure (~200 return on HumanoidRun at 5M steps, WalkerWalk as initial validation target).

---

## Design Decisions

### Replay Buffer: Numpy (CPU)
- Buffer insertion happens from Python (env is CPU-side, stepped from Python loop)
- No way to keep inside a JIT boundary without `lax.scan` over env steps
- Consistent with PPO: Python outer loop → buffer → JAX gradient step
- GPU only sees sampled minibatches per gradient step

### Training Loop Structure
```
for each env step:
    action = actor(obs)               # JAX, jitted
    next_obs, reward, done = env.step(action)  # Python
    buffer.add(obs, action, reward, next_obs, done, truncation)  # numpy
    norm_state.update(obs)            # running stats

    if buffer has enough samples:
        for _ in range(grad_updates_per_step):
            batch = buffer.sample()   # numpy → JAX
            training_state = sac.update(training_state, batch)  # JAX, jitted
```

### Obs Normalization: Store Raw, Normalize at Grad Step
- Running stats updated with raw obs each env step
- Normalization applied inside `sac.update()` at gradient time
- Consistent with PPO convention established in this codebase

---

## Network Architecture (matching Brax)

### Actor
- Input: obs (obs_dim,)
- Encoder: MLP (256, 256), ReLU, LeCun uniform init  ← reuse MlpEncoder
- Head: outputs `2 * action_dim` → split into (mean, log_scale)
  - std = softplus(log_scale) + min_std (min_std=0.001)  ← NOT exp(log_std)
  - Actions: tanh-squashed (same GaussianHead we have, but std via softplus)
- Distribution: sample raw z ~ N(μ, σ), compute log_prob with Jacobian, postprocess via tanh

### Twin Critics (Q-networks)
- Input: concat(obs, action)  →  scalar Q-value
- Two independent MLPs, (256, 256), ReLU, LeCun uniform init
- Both evaluated together, outputs stacked → shape (batch, 2)
- min() taken before use in losses
- New `QHead` needed (obs+action → scalar)

### Temperature (Alpha)
- Stored as `log_alpha` scalar (initialized to 0.0 → alpha=1.0)
- Separate optimizer: adam(3e-4)  (3x higher LR than policy/Q)
- `alpha = exp(log_alpha)` computed on-the-fly

### Target Networks
- Target Q only (no target policy)
- Polyak soft update after every gradient step: `target = (1-tau)*target + tau*online`
- tau = 0.005

---

## SAC Loss Functions — Textbook vs Brax Design Decisions

### 1. Critic (Q-function) Loss

**Textbook:**
```
V(s') = min_i Q_target_i(s', a') - alpha * log_pi(a'|s')
target = r + gamma * (1 - done) * V(s')
L_Q = 0.5 * E[(Q(s,a) - target)^2]
```

**Brax:**
Same math, but with two non-obvious additions:

**a) Truncation masking on Q-error:**
```python
q_error = Q(s,a) - target
q_error *= (1 - truncation)   # zero out at episode boundaries
L_Q = 0.5 * mean(q_error^2)
```
Why: auto-reset envs give `next_obs = reset_obs` at episode timeout, not the true terminal obs.
The Q-target `r + gamma * V(reset_obs)` is wrong (value of fresh episode ≠ value of terminal state).
Zeroing the error at truncation boundaries is the same fix we applied to PPO's GAE.
Without this: critic learns systematically wrong values at every episode boundary.

**b) `discount` field in transition (not `1 - done`):**
Brax stores `discount = 1 - done` directly in the transition rather than computing it in the loss.
This is cleaner — the buffer stores what actually matters for the Bellman backup.

---

### 2. Actor Loss

**Textbook:**
```
L_pi = E[alpha * log_pi(a|s) - Q(s,a)]
```
Gradient flows through: the sampled action `a` (reparameterization trick) and through log_pi.

**Brax:** Identical math. One subtle implementation point:

**Action sampling order matters for numerical stability:**
```python
# CORRECT order:
z = sample_raw(dist_params)            # raw Gaussian sample, no tanh
log_prob = log_prob_with_jacobian(z)   # log N(z|μ,σ) - log|1 - tanh(z)^2|
a = tanh(z)                            # postprocess AFTER log_prob
q = Q(obs, a)
loss = alpha * log_prob - q

# WRONG order:
a = tanh(sample_raw(dist_params))      # postprocess first
log_prob = atanh(a)                    # atanh(±1) = ±inf → NaN
```
Why: tanh maps ℝ → (-1,1). To get log_prob you need atanh, which blows up at ±1.
Sample in pre-tanh space, apply Jacobian correction to log_prob, then tanh for the Q-eval.
This is already how our `GaussianHead` + `distributions.py` works for PPO.

---

### 3. Alpha (Temperature) Loss

**Textbook:**
```
L_alpha = -alpha * (log_pi(a|s) + H_target)
```
Gradient of this w.r.t. alpha: `-(log_pi + H_target)`
But gradient also flows through `log_pi`, back into the policy — which is wrong.
When optimizing alpha, we want log_pi treated as a fixed sample.

**Brax:**
```python
L_alpha = alpha * stop_grad(-log_prob - H_target)
```
The `stop_grad` prevents the alpha gradient from flowing through `log_prob` into the policy.
Without stop_grad: alpha update bleeds gradient into the policy network, corrupting the actor loss.
This is a subtle bug that's easy to miss — the math looks equivalent but the computation graph isn't.

**Target entropy:**
```
H_target = -0.5 * action_dim   (Brax)
H_target = -action_dim         (common textbook/CleanRL)
```
Brax's target is softer (allows the policy to be more deterministic at convergence).
The 0.5 factor is an empirical choice — lower target entropy → more exploitation.
Either works; we'll use Brax's -0.5 * action_dim to match the reference.

---

### 4. std Parameterization: softplus vs exp

**Textbook:** `std = exp(log_std)`, `log_std` clamped to e.g. [-5, 2]

**Brax:** `std = softplus(log_scale) + min_std`

Why softplus:
- `softplus(x) = log(1 + exp(x))` is always positive, smooth, no discontinuity
- `min_std = 0.001` ensures policy is never fully deterministic — `log_prob` never → -inf
- No clamping needed (softplus + min_std is inherently bounded below)

Practical consequence: the policy can never fully collapse to a delta function.
This is important for SAC because entropy is part of the objective — zero entropy would give -inf alpha loss.

---

## File Plan

### New files
```
jax_rl/
  algos/
    sac.py              — SAC algorithm (TrainingState, losses, update, select_action)
  buffers/
    replay_buffer.py    — Numpy circular replay buffer
  configs/
    sac_config.py       — SACConfig dataclass
  networks/
    heads/
      q_head.py         — Q-value head: concat(obs, action) → scalar
train_sac.py            — SAC training script
```

### Modified files
```
jax_rl/configs/__init__.py     — export SACConfig
jax_rl/configs/env_presets.py  — SAC presets for WalkerWalk, HumanoidRun
```

### Reused as-is
```
jax_rl/networks/encoders/mlp.py    — MlpEncoder (same arch)
jax_rl/networks/heads/gaussian.py  — GaussianHead (tanh squash already works)
jax_rl/utils/normalization.py      — running stats (identical usage)
```

---

## Truncation Handling — When to Use It

Truncation masking (zeroing Q-error / GAE delta at episode boundaries) is only needed when
the env **auto-resets** — meaning `next_obs` after a timeout is the reset observation, not
the true terminal observation.

| Env type | Auto-resets? | `handle_truncation` |
|----------|-------------|---------------------|
| MuJoCo Playground / Brax wrappers | YES — signals via `info["truncation"]` | `True` |
| IsaacGym / IsaacLab | YES — parallel envs always reset immediately | `True` |
| `gym.wrappers.AutoResetWrapper` | YES | `True` |
| Raw Gymnasium `TimeLimit` | NO — `next_obs` is true last obs, `truncated=True` is a flag | `False` |
| Raw dm_control (no brax wrapper) | NO | `False` |
| Real robots | NO | `False` |

**Rule:** if you can trust `next_obs` to be the real continuation state, use `handle_truncation=False`
and standard `(1 - done)` masking. If `next_obs` might be a reset observation injected by the
env wrapper, use `handle_truncation=True`.

**Implementation:** `handle_truncation: bool = True` lives in `TrainConfig` (shared between PPO
and SAC). When `False`, truncation array is zeroed out before passing to GAE / Q-loss, so the
math reduces to standard done masking with no code path changes needed.

---

## SACConfig Fields

```python
@dataclass
class SACConfig:
    # Core SAC
    tau: float = 0.005              # Polyak soft update
    gamma: float = 0.99             # discount
    alpha_lr: float = 3e-4          # temperature optimizer LR
    target_entropy_scale: float = 0.5   # target_entropy = -scale * action_dim
    # Brax uses 0.5 (softer, more exploitation). Common alternative: 1.0 (=-action_dim, more exploration).
    # Override entirely by computing target_entropy manually and passing to update().

    # Replay buffer
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000   # steps before first gradient update
    batch_size: int = 256
    grad_updates_per_step: int = 1

    # Network
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"        # SAC uses ReLU (not swish like PPO)

    # Reward
    reward_scaling: float = 1.0
```

---

## TrainingState

```python
@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    actor_opt_state: optax.OptState
    q_params: Any           # both critics' params (twin Q in one pytree)
    q_opt_state: optax.OptState
    target_q_params: Any    # Polyak target
    log_alpha: jnp.ndarray  # scalar
    alpha_opt_state: optax.OptState
```

---

## Reference Hyperparameters (MuJoCo Playground dm_control)

From `dm_control_suite_params.brax_sac_config()` — same for WalkerWalk and HumanoidRun:

| Param | Value |
|-------|-------|
| num_timesteps | 5,000,000 |
| num_envs | 128 |
| batch_size | 512 |
| grad_updates_per_step | 8 |
| learning_rate | 1e-3 (policy, Q, alpha) |
| discounting | 0.99 |
| reward_scaling | 1.0 |
| tau | 0.005 |
| normalize_observations | True |
| episode_length | 1000 |
| min_replay_size | 8192 |
| max_replay_size | 4,194,304 (4M) |
| q_network_layer_norm | True |

Note: `q_network_layer_norm=True` — layer norm applied inside Q-network. This is important for
stability with off-policy learning and large replay buffers. Needs to be implemented in QHead.

---

## Validation Plan

1. **WalkerWalk first** — PPO baseline is 833 avg. SAC should reach similar or better in fewer env steps.
   - Paper shows SAC at ~200 on HumanoidRun at 5M steps — WalkerWalk should be much faster.
2. **Watch for:** Q-value divergence (values growing unboundedly), alpha collapsing to 0 or exploding,
   entropy going to 0 (policy collapsed).
3. **Healthy training signals:**
   - Q-values growing slowly, not exploding
   - Alpha decreasing from 1.0 toward 0.1-0.3 range
   - Entropy staying positive but decreasing gradually
   - Actor loss decreasing (more negative = policy improving)
