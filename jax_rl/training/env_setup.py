"""Environment creation — replaces the 6-line block duplicated across all train scripts."""

import functools
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from mujoco_playground import registry as pg_registry
from mujoco_playground._src import locomotion as pg_locomotion
from jax_rl.envs.wrappers import wrap_for_training

from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.normalization import NormalizationState


# ── Register custom envs with Playground's registry ─────────────────────
def _register_custom_envs():
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
    from jax_rl.envs.locomotion.go2_warp_joystick import default_config as warp_default_config
    if "Go2WarpJoystickFlat" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickFlat",
            functools.partial(WarpJoystick, task="flat_terrain"),
            warp_default_config,
        )
    # Variant: linear torque-speed actuator limit (approximates motor saturation).
    # Playground's registry.load passes config_overrides=None by default, which
    # would clobber a partial(..., config_overrides=...). Bake the flag into a
    # dedicated default_config factory instead.
    def _warp_default_config_torque_speed():
        cfg = warp_default_config()
        cfg.torque_speed_model = True
        return cfg
    if "Go2WarpJoystickFlatTorqueSpeed" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickFlatTorqueSpeed",
            functools.partial(WarpJoystick, task="flat_terrain"),
            _warp_default_config_torque_speed,
        )
    # Ablation: actor obs without accelerometer (state 45d, priv 119d).
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystickNoAccel
    if "Go2WarpJoystickFlatNoAccel" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickFlatNoAccel",
            functools.partial(WarpJoystickNoAccel, task="flat_terrain"),
            warp_default_config,
        )
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.locomotion.go2_warp_curriculum import default_config as curriculum_default_config
    if "Go2WarpJoystickCurriculum" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickCurriculum",
            functools.partial(WarpJoystickCurriculum, task="flat_terrain"),
            curriculum_default_config,
        )
    def _curriculum_ts_default_config():
        cfg = curriculum_default_config()
        cfg.torque_speed_model = True
        return cfg
    if "Go2WarpJoystickCurriculumTorqueSpeed" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickCurriculumTorqueSpeed",
            functools.partial(WarpJoystickCurriculum, task="flat_terrain"),
            _curriculum_ts_default_config,
        )
    from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand
    from jax_rl.envs.locomotion.go2_bongo_handstand import default_config as bongo_default_config
    if "Go2BongoHandstand" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2BongoHandstand",
            functools.partial(BongoHandstand, task="bongo_handstand"),
            bongo_default_config,
        )
    def _bongo_default_config_contraction():
        c = bongo_default_config()
        c.observe_contraction = True
        return c
    if "Go2BongoHandstandContraction" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2BongoHandstandContraction",
            functools.partial(BongoHandstand, task="bongo_handstand"),
            _bongo_default_config_contraction,
        )

    # (MuJoCo Warp PushEnv removed 2026-04-20 — replaced by vendored pymunk
    # gym-pusht (`jax_rl/envs/manipulation/pusht/`) for cross-shape work.)


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

    # Apply wrapper pipeline (action delay, frame stacking, etc.)
    from jax_rl.envs.wrappers import apply_wrapper_pipeline
    env = apply_wrapper_pipeline(env, cfg)

    reset_mode = getattr(cfg, 'reset_mode', 'legacy')
    if reset_mode == "per_step":
        from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
        if isinstance(env.unwrapped, WarpJoystickCurriculum):
            from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper
            env = TerrainCurriculumDRWrapper(
                env, episode_length=cfg.episode_length, mode=reset_mode,
                num_envs=cfg.num_envs,
            )
        else:
            from jax_rl.envs.wrappers.domain_rand import DomainRandWrapper
            env = DomainRandWrapper(env, episode_length=cfg.episode_length, mode=reset_mode)
    else:
        env = wrap_for_training(env, episode_length=cfg.episode_length)
    env_step = _make_nan_safe_step(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    eval_env = pg_registry.load(cfg.env_name)
    # Eval env: same pipeline, but action delay uses fixed max (not randomized).
    import dataclasses
    eval_cfg = cfg
    if cfg.action_delay_range_ms is not None:
        eval_cfg = dataclasses.replace(cfg, action_delay_ms=cfg.action_delay_range_ms[1], action_delay_range_ms=None)
    eval_env = apply_wrapper_pipeline(eval_env, eval_cfg)
    eval_env = wrap_for_training(eval_env, episode_length=cfg.episode_length)

    # Dict obs → obs_dim is the policy obs ("state" key).
    if isinstance(env_state.obs, dict):
        obs_dim = env_state.obs["state"].shape[-1]
    else:
        obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    return env, env_step, env_state, eval_env, obs_dim, action_dim, key


@dataclass
class EnvBundle:
    """Env setup bundle for off-policy training scripts.

    Wraps make_envs output with dict-obs detection so training scripts don't
    need to re-detect asymmetric critic structure.
    """
    env: Any
    env_step: Callable
    env_state: Any
    eval_env: Any
    obs_dim: int
    action_dim: int
    critic_obs_dim: int | None  # None if symmetric
    has_privileged: bool
    dict_obs: bool
    key: Any  # jax.Array


def make_env_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Wrap make_envs + dict obs detection. For off-policy training scripts.

    Returns EnvBundle with dict_obs / has_privileged / critic_obs_dim populated.
    When dict obs with privileged_state is present, obs_dim (already set
    correctly by make_envs) refers to the actor ("state") dim and
    critic_obs_dim refers to the privileged dim.

    Prints a one-line summary when dict obs is detected (matches existing
    per-script print behavior).
    """
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = False
    critic_obs_dim = None

    if dict_obs:
        # NOTE: make_envs already set obs_dim = env_state.obs["state"].shape[-1]
        # for dict obs (see make_envs above). We don't re-extract.
        has_privileged = "privileged_state" in env_state.obs
        if has_privileged:
            critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
            print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs detected: using 'state' key ({obs_dim}d) for off-policy")

    return EnvBundle(
        env=env, env_step=env_step, env_state=env_state, eval_env=eval_env,
        obs_dim=obs_dim, action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        has_privileged=has_privileged,
        dict_obs=dict_obs,
        key=key,
    )


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
