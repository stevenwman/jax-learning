"""Environment creation — replaces the 6-line block duplicated across all train scripts."""

import functools

import jax
import jax.numpy as jnp

from mujoco_playground import registry as pg_registry
from mujoco_playground._src import locomotion as pg_locomotion
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.normalization import NormalizationState


# ── Register custom envs with Playground's registry ─────────────────────
def _register_custom_envs():
    from jax_rl.envs.locomotion.go2_joystick import Joystick, default_config
    if "Go2JoystickFlat" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2JoystickFlat",
            functools.partial(Joystick, task="flat_terrain"),
            default_config,
        )

_register_custom_envs()


def _safe_obs(obs, has_bad):
    """Zero out obs for envs with NaN/Inf. Works with flat arrays or dicts."""
    if isinstance(obs, dict):
        return {k: jnp.where(has_bad[:, None], 0.0, v) for k, v in obs.items()}
    return jnp.where(has_bad[:, None], 0.0, obs)


def _obs_has_bad(obs):
    """Check for NaN/Inf in obs. Works with flat arrays or dicts."""
    if isinstance(obs, dict):
        # Check all obs arrays, any bad in any key → bad
        bads = [jnp.any(jnp.isnan(v) | jnp.isinf(v), axis=-1) for v in obs.values()]
        return functools.reduce(lambda a, b: a | b, bads)
    return jnp.any(jnp.isnan(obs) | jnp.isinf(obs), axis=-1)


def _make_nan_safe_step(raw_step):
    """Wrap env.step to guard against MJX physics NaN/Inf.

    MuJoCo's MJX backend can produce NaN or Inf obs/rewards when the simulation
    encounters extreme states (contact solver failure, singular mass matrix,
    velocity overflow). This happens stochastically with humanoid envs at
    1024 parallel worlds.

    IMPORTANT: Must check BOTH isnan() AND isinf(). MJX produces Inf from
    velocity overflow (different from NaN which comes from solver failure).
    Inf * 0 = NaN, so unguarded Inf corrupts network params silently.
    See LESSONS.md "Inf guard" for the full debugging trail.

    When NaN/Inf is detected:
    - obs replaced with zeros (safe for network forward pass)
    - reward set to 0
    - done set to 1 (triggers auto-reset on next step)

    Supports both flat array obs and dict obs (e.g. {"state", "privileged_state"}).
    """
    @jax.jit
    def safe_step(state, action):
        # Guard NaN/Inf actions (from bad obs → actor forward → bad action)
        action = jnp.where(jnp.isnan(action) | jnp.isinf(action), 0.0, action)
        state = raw_step(state, action)
        # Guard NaN/Inf obs/rewards from MJX physics failures
        has_bad = _obs_has_bad(state.obs)
        has_bad = has_bad | jnp.isnan(state.reward) | jnp.isinf(state.reward)
        safe_obs = _safe_obs(state.obs, has_bad)
        safe_reward = jnp.where(has_bad, 0.0, state.reward)
        safe_done = jnp.where(has_bad, 1.0, state.done)
        state = state.replace(obs=safe_obs, reward=safe_reward, done=safe_done)
        return state
    return safe_step


def make_envs(cfg: TrainConfig, seed: int):
    """Create training env + eval env, JIT env.step, reset training env.

    Returns:
        env: wrapped training environment
        env_step: JIT'd, NaN-safe env.step function
        env_state: initial env state (reset with num_envs)
        eval_env: separate wrapped env for evaluation
        obs_dim: observation dimensionality
        action_dim: action dimensionality
    """
    env = pg_registry.load(cfg.env_name)
    env = wrap_for_brax_training(env, episode_length=cfg.episode_length)
    env_step = _make_nan_safe_step(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    eval_env = pg_registry.load(cfg.env_name)
    eval_env = wrap_for_brax_training(eval_env, episode_length=cfg.episode_length)

    # Dict obs → obs_dim is the policy obs ("state" key).
    if isinstance(env_state.obs, dict):
        obs_dim = env_state.obs["state"].shape[-1]
    else:
        obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    return env, env_step, env_state, eval_env, obs_dim, action_dim, key


def make_identity_norm_state(obs_dim: int) -> NormalizationState:
    """Identity norm state for off-policy algos (no obs normalization).

    Q-network LayerNorm handles input scaling instead.
    Kept as identity for checkpoint/inference compatibility with record_video.py.
    """
    return NormalizationState(
        mean=jnp.zeros(obs_dim),
        mean_of_squares=jnp.ones(obs_dim),
        count=1,
    )
