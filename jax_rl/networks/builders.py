"""Network builders to compose encoder + head from configs."""

from flax import linen as nn
import jax
from jax_rl.configs import EncoderConfig, PolicyHeadConfig, ValueHeadConfig
from jax_rl.networks.encoders import MlpEncoder
from jax_rl.networks.heads import GaussianHead, ValueHead
from jax_rl.networks.distributions import sample_gaussian, gaussian_log_prob


class Actor(nn.Module):
    encoder_config: EncoderConfig
    policy_config: PolicyHeadConfig

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        encoder = MlpEncoder(self.encoder_config)
        policy_head = GaussianHead(self.policy_config)
        return policy_head(encoder(obs))


class Critic(nn.Module):
    encoder_config: EncoderConfig
    value_config: ValueHeadConfig

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        encoder = MlpEncoder(self.encoder_config)
        value_head = ValueHead(self.value_config)
        return value_head(encoder(obs))