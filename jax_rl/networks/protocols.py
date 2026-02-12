"""Protocol definitions for network components.

These protocols define the interfaces that encoder and head implementations must follow.
This enables swapping components (e.g., MLP encoder -> CNN encoder) without changing algorithm code.
"""

from typing import Protocol
import jax
import jax.numpy as jnp


class Encoder(Protocol):
    """Encodes observations (and optionally actions/context) into feature vectors.

    Used by both policy and value networks.
    - For policy: encoder(obs) -> features
    - For Q-function: encoder(obs, action) -> features
    - For goal/skill-conditioned: encoder(obs, context=goal) -> features
    """

    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array | None = None,
        context: jax.Array | None = None,
    ) -> jax.Array:
        """Encode inputs to feature vector.

        Args:
            obs: Observation array, shape (batch, obs_dim) or (obs_dim,)
            action: Optional action array for Q-networks, shape (batch, action_dim) or (action_dim,)
            context: Optional context (goal, skill, task embedding)

        Returns:
            Feature vector, shape (batch, feature_dim) or (feature_dim,)
        """
        ...


class PolicyHead(Protocol):
    """Policy head that outputs action distribution parameters.

    For Gaussian policies: outputs (mean, log_std)
    For deterministic policies (TD3): outputs action directly
    """

    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array] | jax.Array:
        """Compute policy outputs from features.

        Args:
            features: Feature vector from encoder, shape (batch, feature_dim) or (feature_dim,)

        Returns:
            For stochastic: (mean, log_std) tuple, each shape (batch, action_dim) or (action_dim,)
            For deterministic: action array, shape (batch, action_dim) or (action_dim,)
        """
        ...


class ValueHead(Protocol):
    """Value head that outputs state value V(s)."""

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute state value from features.

        Args:
            features: Feature vector from encoder, shape (batch, feature_dim) or (feature_dim,)

        Returns:
            State value(s), shape (batch,) or scalar
        """
        ...


class QHead(Protocol):
    """Q-value head that outputs Q(s,a)."""

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute Q-value from features.

        Args:
            features: Feature vector from encoder (already includes action),
                     shape (batch, feature_dim) or (feature_dim,)

        Returns:
            Q-value(s), shape (batch,) or scalar
        """
        ...


class DistributionalQHead(Protocol):
    """Distributional Q-head (C51) for FastTD3/FastSAC."""

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute distribution over Q-values (C51 logits).

        Args:
            features: Feature vector from encoder, shape (batch, feature_dim) or (feature_dim,)

        Returns:
            Logits over atoms, shape (batch, num_atoms) or (num_atoms,)
        """
        ...

    def q_value(self, features: jax.Array) -> jax.Array:
        """Get expected Q-value from distribution.

        Args:
            features: Feature vector from encoder

        Returns:
            Expected Q-value, shape (batch,) or scalar
        """
        ...
