import flax
import jax.numpy as jnp

@flax.struct.dataclass
class NormalizationState:
    """Normalization state for a single feature."""
    # Var = E[X^2] - E[X]^2
    mean: jnp.ndarray
    mean_of_squares: jnp.ndarray
    count: int

def update(state: NormalizationState, x: jnp.ndarray) -> NormalizationState:
    """Update the normalization state with a new sample."""
    batch_mean = x.mean(axis=0)
    batch_mean_of_squares = (x**2).mean(axis=0)
    batch_count = x.shape[0]
    mean = (state.count * state.mean + batch_mean * batch_count) / (state.count + batch_count)
    mean_of_squares = (state.count * state.mean_of_squares + batch_mean_of_squares * batch_count) / (state.count + batch_count)
    count = state.count + batch_count
    return state.replace(mean=mean, mean_of_squares=mean_of_squares, count=count)

def normalize(state: NormalizationState, x: jnp.ndarray) -> jnp.ndarray:
    """Normalize the input using the normalization state."""
    return (x - state.mean) / (jnp.sqrt(jnp.maximum(state.mean_of_squares - state.mean**2, 0.0)) + 1e-8)

def unnormalize(state: NormalizationState, x: jnp.ndarray) -> jnp.ndarray:
    """Unnormalize the input using the normalization state."""
    return x * (jnp.sqrt(jnp.maximum(state.mean_of_squares - state.mean**2, 0.0)) + 1e-8) + state.mean

def init(obs_dim: int) -> NormalizationState:
    """Initialize the normalization state."""
    return NormalizationState(mean=jnp.zeros(obs_dim), mean_of_squares=jnp.zeros(obs_dim), count=0)