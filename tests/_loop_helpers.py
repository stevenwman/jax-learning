"""Shared helpers for hermetic training-loop smoke tests."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training import EnvBundle


def _stub_env_bundle(num_envs=2, obs_dim=4, action_dim=2):
    """Hermetic stub env bundle for loop tests."""

    @dataclass
    class StubEnvState:
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray
        info: dict

    def stub_step(state, action):
        new_obs = state.obs + 0.01 * jnp.ones_like(state.obs)
        return StubEnvState(
            obs=new_obs,
            reward=jnp.ones((num_envs,)) * 0.5,
            done=jnp.zeros((num_envs,)),
            info={"truncation": jnp.zeros((num_envs,))},
        )

    initial_state = StubEnvState(
        obs=jnp.zeros((num_envs, obs_dim)),
        reward=jnp.zeros((num_envs,)),
        done=jnp.zeros((num_envs,)),
        info={"truncation": jnp.zeros((num_envs,))},
    )

    class StubEnv:
        action_size = action_dim

        def reset(self, keys):
            return initial_state

        def step(self, state, action):
            return stub_step(state, action)

    env = StubEnv()
    return EnvBundle(
        env=env,
        env_step=stub_step,
        env_state=initial_state,
        eval_env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=None,
        has_privileged=False,
        dict_obs=False,
        key=jax.random.PRNGKey(0),
        num_envs=num_envs,
    )


def _common_cfg(num_envs):
    return TrainConfig(
        env_name="StubEnv",
        num_envs=num_envs,
        total_timesteps=40,
        episode_length=100,
        eval_every_n_episodes=10**9,
        gamma=0.99,
        lr=3e-4,
        reward_scaling=1.0,
        n_frame_stack=1,
        handle_truncation=True,
    )


def _patch_eval(monkeypatch):
    import jax_rl.training.offpolicy_loop as ol_module

    def _noop_maybe_eval(*args, **kwargs):
        last_eval_eps = kwargs.get("last_eval_eps", args[7] if len(args) > 7 else 0)
        key = kwargs.get("key", args[8] if len(args) > 8 else None)
        return last_eval_eps, key

    monkeypatch.setattr(ol_module, "maybe_eval_and_checkpoint", _noop_maybe_eval)
    monkeypatch.setattr(ol_module, "final_eval_and_checkpoint", lambda *a, **kw: None)
