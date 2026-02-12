# JAX RL Framework — Session Context

## Overview
Building a JAX-native RL framework with Flax NNX. Goal: accessible, efficient robot learning with sim-to-real pipeline.

Full design doc: `rl_framework_plan.md`

---

## Completed Files

| File | Contents | Status |
|------|----------|--------|
| `configs.py` | `EncoderConfig` dataclass | ✅ Done |
| `encoder.py` | `MLPEncoder` (NNX module, configurable layers) | ✅ Done |
| `heads.py` | `GaussianHead` (mean + log_std), `ValueHead` | ✅ Done |
| `policy.py` | `Policy` (encoder + head, TanhNormal support via `squash` flag) | ✅ Done |
| `buffers.py` | `RolloutBuffer` with `compute_gae` (backward scan) | ✅ Done |
| `ppo.py` | Reference implementation (backup, not reviewed) | ⚠️ Backup only |

---

## Key Technical Decisions

### Architecture
- **Flax NNX** (not Linen) — newer API, more Pythonic
- **Encoder + Head pattern** — swap components without rewriting algos
- **Policy outputs [-1, 1]** — env wrapper scales to actual bounds

### PPO-Specific
- **Separate actor/critic backprops** — no `value_coef` needed, cleaner
- **Shared encoder still OK** — but two optimizers, two backward passes
- **GAE computed with backward scan** — `jax.lax.scan` over reversed timesteps

### Simulation
- **MuJoCo Playground** with **MJWarp backend** (not MJX directly)
- MJWarp: up to 152x faster for locomotion, 313x for manipulation
- Use via `impl='warp'` in MJX API
- NVIDIA GPU required (CUDA 12.4+)
- **Not using mjlab** — that's PyTorch-native, we're JAX

---

## Current Task: PPOConfig

Writing the config dataclass with **separate actor/critic** (no value_coef):

```python
@dataclass
class PPOConfig:
    # Actor
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    
    # Critic
    critic_lr: float = 3e-4
    
    # Shared
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    batch_size: int = 2048
    max_grad_norm: float = 0.5
    
    # Environment
    num_envs: int = 4096
    num_steps: int = 32  # Rollout length before update
    
    # Network (nested config)
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(obs_dim=17))
```

---

## Next Steps

1. **Finish PPOConfig** — user writing this
2. **PPO loss functions** — policy loss (clipped surrogate), value loss (MSE), entropy bonus
3. **PPO class** — holds actor, critic, two optimizers, buffer
4. **Trainer loop** — collect rollouts, call update
5. **MuJoCo Playground adapter** — connect to MJX envs
6. **Test on CartPole/HalfCheetah**

---

## NNX Gotchas Discovered

- `nnx.List` required for lists of modules (not Python list)
- `nnx.Rngs` for init, raw `jax.random.PRNGKey` for runtime
- NNX 0.11+ API changes:
  - `nnx.Optimizer(model, tx, wrt=nnx.Param)`
  - `optimizer.update(model, grads)` — requires both args

---

## GAE Recap

Recursive formula:
```
A_t = δ_t + (γλ)(1 - done_t) * A_{t+1}
```
Where `δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)`

Scan backward from T-1 to 0. Bootstrap final step with `next_value` from critic.

Returns = advantages + values (target for critic).

---

## Useful JAX Syntax

```python
x[None]       # Add axis at front: (3,) → (1, 3)
x[:, None]    # Add axis at end: (3,) → (3, 1)
x[::-1]       # Reverse
x[1:]         # From index 1 to end
x[:-1]        # All but last
x.at[i].set(v)  # Immutable update, returns new array
```

---

## Dependencies

```bash
pip install jax flax optax distrax
```

For MuJoCo Playground: see their install docs (includes mujoco, mujoco-mjx, etc.)
