"""Tests for make_collect factory (onpolicy_collect.py).

Uses a pure-JAX stub env so CPU-only is sufficient.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from jax_rl.training.onpolicy_collect import make_collect, StepData
from jax_rl.utils.normalization import init as norm_init


OBS_DIM = 4
ACTION_DIM = 2
NUM_ENVS = 3
NUM_STEPS = 5


# ── Stub env ──────────────────────────────────────────────────────────────

class StubEnvState(NamedTuple):
    obs: dict
    reward: jax.Array
    done: jax.Array
    info: dict


def _reset_stub(num_envs):
    return StubEnvState(
        obs={"state": jnp.zeros((num_envs, OBS_DIM)),
             "privileged_state": jnp.zeros((num_envs, OBS_DIM))},
        reward=jnp.zeros(num_envs),
        done=jnp.zeros(num_envs),
        info={"truncation": jnp.zeros(num_envs)},
    )


def _step_stub(state, action):
    # Reward = sum of actions; obs = previous obs + action[..., :OBS_DIM] padded
    new_reward = action.sum(axis=-1)
    pad = jnp.zeros((action.shape[0], OBS_DIM - ACTION_DIM))
    delta = jnp.concatenate([action, pad], axis=-1)
    new_obs = state.obs["state"] + delta
    return StubEnvState(
        obs={"state": new_obs, "privileged_state": new_obs},
        reward=new_reward,
        done=jnp.zeros(action.shape[0]),
        info={"truncation": jnp.zeros(action.shape[0])},
    )


class DummyTS(NamedTuple):
    actor_params: jax.Array
    critic_params: jax.Array


def _select_stochastic(ap, cp, obs, critic_obs, key):
    # Deterministic constant action so rollouts are byte-identical across runs
    action = jnp.full((obs.shape[0], ACTION_DIM), 0.1)
    log_prob = jnp.zeros(obs.shape[0])
    value = jnp.zeros(obs.shape[0])
    return action, log_prob, value


def _select_deterministic(ap, cp, obs, critic_obs):
    action = jnp.zeros((obs.shape[0], ACTION_DIM))
    log_prob = jnp.zeros(obs.shape[0])
    value = jnp.zeros(obs.shape[0])
    return action, log_prob, value


def _make_collect_base(extra_rollout_fn=None, reward_augment_fn=None):
    return make_collect(
        env_step=_step_stub,
        select_stochastic=_select_stochastic,
        select_deterministic=_select_deterministic,
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        reward_scaling=1.0,
        handle_truncation=False,
        get_policy_obs=lambda o: o["state"],
        get_critic_obs=lambda o: o["privileged_state"],
        n_frame_stack=1,
        policy_raw_dim=OBS_DIM,
        critic_raw_dim=OBS_DIM,
        extra_rollout_fn=extra_rollout_fn,
        reward_augment_fn=reward_augment_fn,
    )


# ── Tests ──────────────────────────────────────────────────────────────────

def _run_once(collect_fn):
    env_state = _reset_stub(NUM_ENVS)
    ts = DummyTS(jnp.zeros(1), jnp.zeros(1))
    ns = norm_init(OBS_DIM)
    cns = norm_init(OBS_DIM)
    key = jax.random.PRNGKey(0)
    running_ep_return = jnp.zeros(NUM_ENVS)
    return collect_fn(ts, env_state, ns, cns, key, running_ep_return)


def test_baseline_collect_no_hooks():
    """No hooks → extras is empty tuple in each step; shapes correct."""
    collect = _make_collect_base()
    out = _run_once(collect)
    rollout = out[4]
    assert rollout.obs.shape == (NUM_STEPS, NUM_ENVS, OBS_DIM)
    assert rollout.action.shape == (NUM_STEPS, NUM_ENVS, ACTION_DIM)
    assert rollout.reward.shape == (NUM_STEPS, NUM_ENVS)
    # extras is empty tuple, stacked by scan to... an empty tuple (no leaves)
    assert rollout.extras == ()


def test_extra_rollout_fn_stacked():
    """Hook returning NamedTuple gets stacked over steps."""

    class Extras(NamedTuple):
        c: jax.Array       # (E, 3)
        c_dot: jax.Array   # (E, 3)

    def extra_fn(ts, env_state, key):
        return Extras(
            c=jnp.ones((NUM_ENVS, 3)),
            c_dot=jnp.ones((NUM_ENVS, 3)) * 2.0,
        )

    collect = _make_collect_base(extra_rollout_fn=extra_fn)
    out = _run_once(collect)
    rollout = out[4]
    assert rollout.extras.c.shape == (NUM_STEPS, NUM_ENVS, 3)
    assert rollout.extras.c_dot.shape == (NUM_STEPS, NUM_ENVS, 3)
    assert jnp.allclose(rollout.extras.c, 1.0)
    assert jnp.allclose(rollout.extras.c_dot, 2.0)


def test_reward_augment_fn_adds_to_reward():
    """reward_augment_fn bonus added AFTER reward_scaling."""

    def extra_fn(ts, env_state, key):
        return jnp.zeros((NUM_ENVS,))  # minimal placeholder

    def augment_fn(ts, env_state, extras):
        return jnp.ones(NUM_ENVS) * 0.5  # constant bonus

    collect_with = _make_collect_base(extra_rollout_fn=extra_fn, reward_augment_fn=augment_fn)
    collect_base = _make_collect_base()

    rollout_with = _run_once(collect_with)[4]
    rollout_base = _run_once(collect_base)[4]

    diff = rollout_with.reward - rollout_base.reward
    assert jnp.allclose(diff, 0.5), f"diff={diff}"


def test_reward_augment_only_no_extras():
    """reward_augment_fn works with no extras hook — receives `()` as extras."""
    def augment_fn(ts, env_state, extras):
        # extras is () (empty tuple) when no extras hook set
        return jnp.ones(NUM_ENVS) * 0.25

    collect_with = _make_collect_base(reward_augment_fn=augment_fn)
    collect_base = _make_collect_base()

    rollout_with = _run_once(collect_with)[4]
    rollout_base = _run_once(collect_base)[4]

    diff = rollout_with.reward - rollout_base.reward
    assert jnp.allclose(diff, 0.25), f"diff={diff}"
