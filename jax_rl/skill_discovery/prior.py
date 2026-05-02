"""Skill priors for skill discovery."""
import jax
import jax.numpy as jnp


def sample_skill(key, prior, num_envs, skill_dim):
    if prior == "one_hot":
        idx = jax.random.randint(key, (num_envs,), 0, skill_dim)
        return jax.nn.one_hot(idx, skill_dim)
    if prior == "dirichlet":
        raise NotImplementedError("Dirichlet prior deferred to SD-E")
    if prior == "hypersphere":
        raise NotImplementedError("Hypersphere prior deferred to SD-E")
    raise ValueError(f"unknown prior: {prior}")


def validate_skill(z, prior, skill_dim):
    if z.shape[-1] != skill_dim:
        raise ValueError(f"skill_dim mismatch: got {z.shape[-1]}, expected {skill_dim}")
    if prior == "one_hot":
        sums = z.sum(axis=-1)
        if not jnp.allclose(sums, 1.0):
            raise ValueError("one-hot prior requires rows to sum to 1")
        if not jnp.all((z == 0) | (z == 1)):
            raise ValueError("one-hot prior requires 0/1 entries")
