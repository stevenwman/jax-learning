"""Network builders — compose encoder + head into Linen modules.

All algos use these builders for actor/critic construction. When we add a new
encoder type (CNN, ViT), we change MlpEncoder → CnnEncoder here — zero algo changes.

Builder types:
  Actor              — encoder + GaussianHead (SAC, FastSAC, PPO)
  DeterministicActor — encoder + DeterministicHead (TD3, FastTD3)
  VCritic            — encoder + ValueHead (PPO)

Q critics (QHead, DistributionalQHead) are NOT wrapped — they process
concat(obs, action) jointly, not through an obs encoder. They stay as standalone
modules used directly by algos.
"""

from flax import linen as nn
import jax
from jax_rl.configs import EncoderConfig, PolicyHeadConfig
from jax_rl.networks.encoders import MlpEncoder
from jax_rl.networks.heads import GaussianHead, ValueHead, DeterministicHead


class Actor(nn.Module):
    """Stochastic actor: encoder → GaussianHead → (mean, log_std).

    Used by: PPO, SAC, FastSAC.
    """
    encoder_config: EncoderConfig
    policy_config: PolicyHeadConfig

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        encoder = MlpEncoder(self.encoder_config)
        policy_head = GaussianHead(self.policy_config)
        return policy_head(encoder(obs))


class DeterministicActor(nn.Module):
    """Deterministic actor: encoder → DeterministicHead → tanh(action).

    Used by: TD3, FastTD3.
    """
    encoder_config: EncoderConfig
    action_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        encoder = MlpEncoder(self.encoder_config)
        head = DeterministicHead(self.action_dim)
        return head(encoder(obs))


class VCritic(nn.Module):
    """Value critic: encoder → ValueHead → scalar V(s).

    Used by: PPO.
    """
    encoder_config: EncoderConfig

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        encoder = MlpEncoder(self.encoder_config)
        value_head = ValueHead()
        return value_head(encoder(obs))
