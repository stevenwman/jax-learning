# Networks

Modular Flax `nn.Module` architecture with separate encoders, heads, and builders.

```
obs → Encoder → features → Head → output
              ↗
    Builder (composes encoder + head)
```

| Layer | Role | Used by |
|-------|------|---------|
| [Actor](#actor) | Stochastic: encoder → (mean, log_std) | PPO, SAC, FastSAC |
| [DeterministicActor](#deterministicactor) | Deterministic: encoder → tanh(action) | TD3, FastTD3 |
| [VCritic](#vcritic) | Value: encoder → scalar V(s) | PPO |
| [MlpEncoder](#mlpencoder) | Raw obs → feature vector | All |
| [DeterministicHead](#deterministichead) | Features → tanh(action) | TD3, FastTD3 |
| [GaussianHead](#gaussianhead) | Features → (mean, log_std) | Stochastic actors |
| [ValueHead](#valuehead) | Features → scalar | PPO critic |
| [QHead](#qhead) | (obs, action) → scalar Q | SAC, TD3 |
| [DistributionalQHead](#distributionalqhead) | (obs, action) → atom logits | FastSAC, FastTD3 |

---

## Builders

Compose an encoder and head into a complete network.

### Actor

```python
from jax_rl.networks.builders import Actor
```

Stochastic actor: `encoder → GaussianHead → (mean, log_std)`.

| Field | Type |
|-------|------|
| `encoder_config` | `EncoderConfig` |
| `policy_config` | `PolicyHeadConfig` |

`__call__(obs) → (mean, log_std)`

---

### DeterministicActor

```python
from jax_rl.networks.builders import DeterministicActor
```

Deterministic actor: `encoder → DeterministicHead → tanh(action)`.

| Field | Type |
|-------|------|
| `encoder_config` | `EncoderConfig` |
| `action_dim` | `int` |

`__call__(obs) → action`

---

### VCritic

```python
from jax_rl.networks.builders import VCritic
```

Value critic: `encoder → ValueHead → scalar V(s)`.

| Field | Type |
|-------|------|
| `encoder_config` | `EncoderConfig` |

`__call__(obs) → value`

---

## Encoders

### MlpEncoder

```python
from jax_rl.networks.encoders.mlp import MlpEncoder
```

Multi-layer perceptron: `obs → [Dense → Norm? → activation] × N → features`.

| Field | Type |
|-------|------|
| `config` | `EncoderConfig` |

`__call__(obs, context=None) → features`

`feature_dim → int` (property)
: Output dimensionality = `config.hidden_dim[-1]`.

---

## Heads

### DeterministicHead

```python
from jax_rl.networks.heads.deterministic import DeterministicHead
```

Deterministic policy: `features → Dense → tanh(action)`. Used by TD3 and FastTD3.

| Field | Type |
|-------|------|
| `action_dim` | `int` |

`__call__(features) → action`

---

### GaussianHead

```python
from jax_rl.networks.heads.gaussian import GaussianHead
```

Diagonal Gaussian output. Optional state-dependent or state-independent std.

| Field | Type |
|-------|------|
| `config` | `PolicyHeadConfig` |

`__call__(features) → (mean, log_std)`

---

### ValueHead

```python
from jax_rl.networks.heads.value import ValueHead
```

Single Dense layer → scalar value. No configurable fields.

`__call__(features) → value`

---

### QHead

```python
from jax_rl.networks.heads.q_head import QHead
```

MLP Q-network: `concat(obs, action) → scalar Q-value`. Optional layer norm.

| Field | Type | Default |
|-------|------|---------|
| `hidden_dim` | `tuple` | — |
| `activation` | `str` | `"relu"` |
| `layer_norm` | `bool` | `True` |

`__call__(obs, action) → q_value`

---

### DistributionalQHead

```python
from jax_rl.networks.heads.q_distributional import DistributionalQHead
```

C51 categorical Q-network: `concat(obs, action) → (batch, num_atoms) logits`.

| Field | Type | Default |
|-------|------|---------|
| `hidden_dim` | `tuple` | — |
| `num_atoms` | `int` | `51` |
| `activation` | `str` | `"relu"` |
| `layer_norm` | `bool` | `True` |

`__call__(obs, action) → logits`

---

## Flash Blocks

```python
from jax_rl.networks.flash_blocks import (
    FlashSACActor, FlashSACCritic, FlashSACEmbedder,
    FlashSACBlock, UnitRMSNorm, normalize_weights,
)
```

Inverted residual architecture for FlashSAC. All modules accept a `train: bool` flag for BatchNorm.

### FlashSACEmbedder

Projects raw input to `hidden_dim` after BatchNorm. No activation — first block provides nonlinearity.

`__call__(x, train) → features`

### FlashSACBlock

Inverted residual: expand to `hidden_dim * expansion`, project back with skip connection.

| Field | Type | Default |
|-------|------|---------|
| `hidden_dim` | `int` | — |
| `expansion` | `int` | `4` |

`__call__(x, train) → features`

### UnitRMSNorm

RMS normalization with learnable per-feature scale. No bias.

`__call__(x) → normalized`

### FlashSACActor

`Embedder → N × Block → UnitRMSNorm → (mean, log_std)`. Log-std clamped to `[-10, 2]` via tanh.

| Field | Type |
|-------|------|
| `hidden_dim` | `int` |
| `num_blocks` | `int` |
| `expansion` | `int` |
| `action_dim` | `int` |

`__call__(obs, train) → (mean, log_std)`

### FlashSACCritic

`concat(obs, action) → Embedder → N × Block → UnitRMSNorm → num_atoms logits`.

| Field | Type |
|-------|------|
| `hidden_dim` | `int` |
| `num_blocks` | `int` |
| `expansion` | `int` |
| `num_atoms` | `int` |

`__call__(obs, action, train) → logits`

### normalize_weights

```python
normalize_weights(params) → params
```

Project parameters to FlashSAC constraint set. Normalizes Dense kernels (per-column L2), BatchNorm (scale+bias jointly), UnitRMSNorm scale.
