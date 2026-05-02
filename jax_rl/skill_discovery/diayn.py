"""DIAYN auxiliary module — discriminator, sample-time reward, and gradient update.

DIAYN (Eysenbach et al. 2018) trains a discriminator q(z | s) to predict the
skill from the current state, and uses log q(z | s) - log p(z) as the intrinsic
reward. The reward is computed at sample time from the current state s (not the
next state s' — that's DADS).

SD-A ships a plain ReLU MLP discriminator. ELU + SimBa residual upgrades land
in SD-C/E.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from jax_rl.networks.activations import ACTIVATIONS

# Matches DIAYN reference (ben-eysenbach/sac:diayn.py:21, 185)
EPS = 1e-6


class Discriminator(nn.Module):
    """Plain MLP discriminator q(z | s)."""

    hidden_dim: tuple[int, ...]
    num_skills: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for h in self.hidden_dim:
            x = act(nn.Dense(h)(x))
        return nn.Dense(self.num_skills)(x)


def diayn_reward(disc_params, disc, obs_factor, z_onehot, num_skills):
    """Sample-time DIAYN reward: log q(z|s) - log p(z) + EPS.

    Uses current state `obs_factor` (Eysenbach 2018), uniform skill prior
    p(z) = 1 / num_skills.
    """
    log_q = jax.nn.log_softmax(disc.apply(disc_params, obs_factor), axis=-1)
    log_q_z = jnp.sum(log_q * z_onehot, axis=-1)
    log_p_z = -jnp.log(num_skills)
    return log_q_z - log_p_z + EPS


def diayn_update(disc_params, opt_state, disc, optimizer, obs_factor, z_indices):
    """One Adam step of softmax cross-entropy on disc(obs_factor) vs z_indices."""

    def loss_fn(p):
        logits = disc.apply(p, obs_factor)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, z_indices).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == z_indices)
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, disc_params)
    new_params = optax.apply_updates(disc_params, updates)
    return new_params, new_opt_state, {"disc_loss": loss, "disc_accuracy": acc}
