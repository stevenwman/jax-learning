# Buffers

Data storage for on-policy and off-policy algorithms.

| Buffer | Used by | Storage |
|--------|---------|---------|
| [JaxReplayBuffer](#jaxreplaybuffer) | SAC, TD3, FastSAC, FastTD3, FlashSAC | GPU-resident circular FIFO |
| [RolloutBuffer](#rolloutbuffer) | PPO | CPU/GPU trajectory buffer |

---

## JaxReplayBuffer

```python
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
```

GPU-resident circular FIFO replay buffer with uniform random sampling. Supports frame stacking and asymmetric critic observations.

**Constructor**

```python
JaxReplayBuffer(
    obs_dim: int,
    action_dim: int,
    max_size: int = 1_000_000,
    frame_stack_config: FrameStackConfig | None = None,
    extra_obs_dims: dict[str, int] | None = None,  # asymmetric critic
)
```

**Methods**

`add_batch(obs, action, reward, next_obs, done, truncation=None, **extra)`
: Add a batch of transitions. Accepts JAX or NumPy arrays (auto-converts).

`sample(batch_size, key) → dict`
: Random minibatch with keys: `obs`, `action`, `reward`, `next_obs`, `done`, `truncation` (plus any extra fields).

`__len__() → int`
: Current number of stored transitions.

---

### FrameStackConfig

```python
from jax_rl.buffers.jax_replay_buffer import FrameStackConfig
```

Config for sample-time frame stack reconstruction (avoids storing redundant frames).

| Field | Type | Description |
|-------|------|-------------|
| `n_frames` | `int` | Number of frames to stack |
| `raw_dim` | `int` | Single-frame observation dimension |
| `num_envs` | `int` | Stride for same-environment lookback |

---

## RolloutBuffer

```python
from jax_rl.buffers.rollout import RolloutBuffer, RolloutBatch
```

Stores on-policy trajectories and computes GAE advantages.

**Constructor**

```python
RolloutBuffer(num_steps: int, num_envs: int, obs_dim: int, action_dim: int)
```

**Methods**

`add(obs, action, reward, done, truncation, log_prob, value)`
: Add one step of experience across all environments.

`get(next_value, gamma, gae_lambda) → RolloutBatch`
: Return batch with computed advantages and returns via GAE.

`reset()`
: Reset buffer pointer to 0.

---

### RolloutBatch

NamedTuple returned by `RolloutBuffer.get()`. All arrays are shape `(num_steps, num_envs, ...)`.

| Field | Description |
|-------|-------------|
| `obs` | Observations |
| `actions` | Actions taken |
| `rewards` | Rewards received |
| `dones` | Terminal flags |
| `truncations` | Truncation flags |
| `log_probs` | Action log-probabilities |
| `values` | Value estimates |
| `advantages` | GAE advantages (computed by `get()`) |
| `returns` | `advantages + values` |

---

## compute_gae

```python
from jax_rl.buffers.rollout import compute_gae
```

```python
compute_gae(rewards, values, dones, truncations, next_value, gamma, gae_lambda)
    → (advantages, returns)
```

Generalized Advantage Estimation matching Brax's approach. Handles truncation by zeroing deltas at timeout steps — prevents gradient flow across resets.
