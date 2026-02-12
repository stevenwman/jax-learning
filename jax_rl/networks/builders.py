"""Network builders to compose encoder + head from configs."""

from flax import nnx
import jax
from jax_rl.configs import EncoderConfig, PolicyHeadConfig, ValueHeadConfig
from jax_rl.networks.encoders import MLPEncoder
from jax_rl.networks.heads import GaussianHead, ValueHead
from jax_rl.networks.distributions import sample_gaussian, gaussian_log_prob


class Actor(nnx.Module):
    """Actor network: encoder + policy head.

    Composes an encoder and policy head into a full actor network.
    Provides methods for forward pass and sampling actions.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        policy_config: PolicyHeadConfig,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize actor network.

        Args:
            encoder_config: Configuration for encoder
            policy_config: Configuration for policy head
            rngs: Random number generators for initialization
        """
        # Build encoder
        self.encoder = MLPEncoder(encoder_config, rngs)

        # Build policy head
        feature_dim = self.encoder.feature_dim
        self.head = GaussianHead(feature_dim, policy_config, rngs)

        # Store config
        self.policy_config = policy_config

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass: obs -> (mean, log_std).

        Args:
            obs: Observation, shape (batch, obs_dim) or (obs_dim,)

        Returns:
            (mean, log_std) tuple for Gaussian policy
        """
        features = self.encoder(obs)
        return self.head(features)

    def sample(
        self, obs: jax.Array, key: jax.random.PRNGKey
    ) -> tuple[jax.Array, jax.Array]:
        """Sample action from policy.

        Args:
            obs: Observation, shape (batch, obs_dim) or (obs_dim,)
            key: PRNGKey for sampling

        Returns:
            (action, log_prob) tuple
        """
        mean, log_std = self(obs)
        return sample_gaussian(mean, log_std, key, squash=self.policy_config.squash)

    def log_prob(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Compute log probability of action under current policy.

        Args:
            obs: Observation, shape (batch, obs_dim) or (obs_dim,)
            action: Action, shape (batch, action_dim) or (action_dim,)

        Returns:
            Log probability, shape (batch,) or scalar
        """
        mean, log_std = self(obs)
        return gaussian_log_prob(mean, log_std, action, squash=self.policy_config.squash)


class Critic(nnx.Module):
    """Critic network: encoder + value head.

    Composes an encoder and value head into a full critic network.
    Estimates state value V(s) for PPO or other value-based methods.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        value_config: ValueHeadConfig,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize critic network.

        Args:
            encoder_config: Configuration for encoder
            value_config: Configuration for value head
            rngs: Random number generators for initialization
        """
        # Build encoder
        self.encoder = MLPEncoder(encoder_config, rngs)

        # Build value head
        feature_dim = self.encoder.feature_dim
        self.head = ValueHead(feature_dim, value_config, rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass: obs -> value.

        Args:
            obs: Observation, shape (batch, obs_dim) or (obs_dim,)

        Returns:
            State value(s), shape (batch,) or scalar
        """
        features = self.encoder(obs)
        return self.head(features)


def build_actor_critic(
    encoder_config: EncoderConfig,
    policy_config: PolicyHeadConfig,
    value_config: ValueHeadConfig,
    rngs: nnx.Rngs,
) -> tuple[Actor, Critic]:
    """Build actor and critic networks from configs.

    Convenience function to build both networks at once.

    Args:
        encoder_config: Configuration for encoders (same for actor and critic)
        policy_config: Configuration for policy head
        value_config: Configuration for value head
        rngs: Random number generators for initialization

    Returns:
        (actor, critic) tuple
    """
    actor = Actor(encoder_config, policy_config, rngs)
    critic = Critic(encoder_config, value_config, rngs)
    return actor, critic
