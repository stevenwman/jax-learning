"""MJX (MuJoCo Playground / MJWarp) backend for the env-bundle pipeline.

Owns:
- Custom env registration with Playground's registry.
- NaN/Inf safe wrapper around env.step (MJX physics can blow up at scale).
- `make_envs(cfg, seed)`: builds train + eval envs, returns the legacy 7-tuple.
- `make_mjx_env_bundle(cfg, seed)`: returns an EnvBundle.

Registers itself as the "mjx" backend on import.
"""

import functools

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from mujoco_playground import registry as pg_registry
from mujoco_playground._src import locomotion as pg_locomotion

from jax_rl.configs.train_config import TrainConfig
from jax_rl.envs.wrappers import wrap_for_training
from jax_rl.training.env_bundle import EnvBundle


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
    # Prototype: flat-ground PosTrack — delta_xy_yaw obs + Lorentzian reward.
    # Designed to share parameterization with Go2WarpSplitbeltPosTrack so
    # cross-deploy is direct. See jax_rl/envs/locomotion/go2_warp_flat_postrack.py.
    from jax_rl.envs.locomotion.go2_warp_flat_postrack import (
        WarpFlatPosTrack as WarpFlatPosTrack,
        default_config as warp_flat_postrack_default_config,
    )
    if "Go2WarpFlatPosTrackProto" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpFlatPosTrackProto",
            functools.partial(WarpFlatPosTrack, task="flat_terrain"),
            warp_flat_postrack_default_config,
        )
    # Hardware-conservative variant: 45d state (no accel) + action_scale=0.25.
    # Named "Unitree" for the parts that are partially aligned with
    # unitree_rl_lab's Go2 deploy contract:
    #   - matched: action_scale 0.25, no accelerometer in actor obs,
    #     Kp=20/Kd=0.5 (already shared)
    #   - NOT matched: explicit per-term obs scales (gyro×0.2, jvel×0.05);
    #     we rely on running obs_norm for whitening instead. Reward scaling,
    #     command sampling, and event randomization also differ.
    # Closer-than-default to the working Unitree stack; not bitwise parity.
    def _warp_default_config_unitree():
        cfg = warp_default_config()
        cfg.action_scale = 0.25
        return cfg
    if "Go2WarpJoystickUnitree" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpJoystickUnitree",
            functools.partial(WarpJoystickNoAccel, task="flat_terrain"),
            _warp_default_config_unitree,
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

    from jax_rl.envs.locomotion.g1_warp_joystick import G1WarpJoystick
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config as g1_joystick_default_config
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma as g1_joystick_holosoma_config
    if "G1WarpJoystickFlat" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickFlat",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_default_config,
        )
    if "G1WarpJoystickHolo" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHolo",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_config,
        )
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma_soft as g1_joystick_holosoma_soft_config
    if "G1WarpJoystickHoloSoft" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHoloSoft",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_soft_config,
        )
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma_wide as g1_joystick_holosoma_wide_config
    if "G1WarpJoystickHoloWide" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHoloWide",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_wide_config,
        )
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma_lift as g1_joystick_holosoma_lift_config
    if "G1WarpJoystickHoloLift" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHoloLift",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_lift_config,
        )
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma_clearance as g1_joystick_holosoma_clearance_config
    if "G1WarpJoystickHoloClearance" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHoloClearance",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_clearance_config,
        )
    from jax_rl.envs.locomotion.g1_warp_joystick import default_config_holosoma_clearance_wide as g1_joystick_holosoma_clearance_wide_config
    if "G1WarpJoystickHoloClearanceWide" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpJoystickHoloClearanceWide",
            functools.partial(G1WarpJoystick, task="flat_terrain"),
            g1_joystick_holosoma_clearance_wide_config,
        )

    from jax_rl.envs.locomotion.g1_warp_splitbelt import G1WarpSplitbeltEnv
    from jax_rl.envs.locomotion.g1_warp_splitbelt import default_config as g1_splitbelt_default_config
    from jax_rl.envs.locomotion.g1_warp_splitbelt import default_config_tied as g1_splitbelt_tied_config
    if "G1WarpSplitbelt" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpSplitbelt",
            functools.partial(G1WarpSplitbeltEnv, task="splitbelt"),
            g1_splitbelt_default_config,
        )
    if "G1WarpSplitbeltTied" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpSplitbeltTied",
            functools.partial(G1WarpSplitbeltEnv, task="splitbelt"),
            g1_splitbelt_tied_config,
        )
    from jax_rl.envs.locomotion.g1_warp_splitbelt import default_config_informed as g1_splitbelt_informed_config
    if "G1WarpSplitbeltInformed" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpSplitbeltInformed",
            functools.partial(G1WarpSplitbeltEnv, task="splitbelt"),
            g1_splitbelt_informed_config,
        )
    from jax_rl.envs.locomotion.g1_warp_splitbelt import default_config_informed_tied as g1_splitbelt_informed_tied_config
    if "G1WarpSplitbeltInformedTied" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpSplitbeltInformedTied",
            functools.partial(G1WarpSplitbeltEnv, task="splitbelt"),
            g1_splitbelt_informed_tied_config,
        )
    from jax_rl.envs.locomotion.g1_warp_splitbelt import default_config_clearance_tied as g1_splitbelt_clearance_tied_config
    if "G1WarpSplitbeltClearanceTied" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "G1WarpSplitbeltClearanceTied",
            functools.partial(G1WarpSplitbeltEnv, task="splitbelt"),
            g1_splitbelt_clearance_tied_config,
        )

    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from jax_rl.envs.locomotion.go2_warp_splitbelt import default_config as splitbelt_default_config
    if "Go2WarpSplitbelt" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbelt",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt"),
            splitbelt_default_config,
        )
    # DR variant: every episode samples (vL, vR) from (v_range, ratio_range).
    # Trains a policy across the full belt-speed manifold (A2 protocol prep).
    def _splitbelt_dr_default_config():
        cfg = splitbelt_default_config()
        cfg.schedule_kind = "random_per_episode"
        cfg.schedule_params = config_dict.create(
            v_range=(0.3, 1.5),
            ratio_range=(0.5, 2.0),
        )
        return cfg
    if "Go2WarpSplitbeltDR" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbeltDR",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt_dr"),
            _splitbelt_dr_default_config,
        )
    # Pose-track + DR variant: actor sees world-frame body pose (idealized,
    # NOT real-robot deployable). Belt speed DR via random_per_episode.
    # Reward = pose-position + pose-orientation tracking (treadmill_drift dropped).
    def _splitbelt_pose_dr_default_config():
        cfg = splitbelt_default_config()
        cfg.obs_mode = "pose_track"
        cfg.schedule_kind = "random_per_episode"
        cfg.schedule_params = config_dict.create(
            v_range=(0.3, 1.5),
            ratio_range=(0.5, 2.0),
        )
        # Swap reward: drop treadmill_drift, enable pose-track terms.
        cfg.reward_config.scales.treadmill_drift = 0.0
        cfg.reward_config.scales.pose_pos_track = 5.0
        cfg.reward_config.scales.pose_orient_track = 2.0
        return cfg
    if "Go2WarpSplitbeltPoseDR" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbeltPoseDR",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt_pose_dr"),
            _splitbelt_pose_dr_default_config,
        )
    # Position-tracking unified variant (2026-05-11). Drops vel tracking and
    # Gaussian pose_pos_track; uses Lorentzian-kernel position tracking with
    # a marching target (fixed at origin for splitbelt cmd=0). Heavy-tail
    # kernel avoids the OOD collapse seen in PoseDR's exp reward. Same belt
    # DR as PoseDR (random_per_episode v∈[0.3,1.5], ratio∈[0.5,2.0]).
    def _splitbelt_pos_track_default_config():
        cfg = splitbelt_default_config()
        cfg.obs_mode = "pos_track"   # 45d state — matches Go2WarpFlatPosTrackProto
        cfg.schedule_kind = "random_per_episode"
        cfg.schedule_params = config_dict.create(
            v_range=(0.3, 1.5),
            ratio_range=(0.5, 2.0),
        )
        # Unified PosTrack reward — body-frame Lorentzian on delta_xy + cos(d_yaw).
        # Matches Go2WarpFlatPosTrackProto verbatim so a flat-trained policy is
        # in-distribution at cross-deploy on this env.
        cfg.reward_config.scales.tracking_lin_vel = 0.0
        cfg.reward_config.scales.tracking_ang_vel = 0.0   # subsumed by orient_track_yaw
        cfg.reward_config.scales.treadmill_drift = 0.0    # subsumed by pos_track_xy
        cfg.reward_config.scales.pose_pos_track = 0.0     # legacy Gaussian — off
        cfg.reward_config.scales.pose_orient_track = 0.0  # legacy upvector — off
        cfg.reward_config.scales.pos_track_unified = 0.0  # legacy 2d term — off
        cfg.reward_config.scales.pos_track_xy = 10.0      # new principal term
        cfg.reward_config.scales.orient_track_yaw = 5.0
        cfg.reward_config.pos_track_lx = 0.5
        cfg.reward_config.pos_track_ly = 0.3
        return cfg
    if "Go2WarpSplitbeltPosTrack" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbeltPosTrack",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt_pos_track"),
            _splitbelt_pos_track_default_config,
        )
    # TiedDR ablation (2026-05-12). Same as PosTrack but ratio collapsed to
    # 1.0 — both belts share the same per-episode-sampled speed (no split
    # asymmetry during training). Isolates "did belt-speed-magnitude DR help?"
    # vs "did belt-asymmetry DR help?" axes.
    def _splitbelt_pos_track_tied_dr_default_config():
        cfg = _splitbelt_pos_track_default_config()
        cfg.unlock()
        cfg.schedule_params = config_dict.create(
            v_range=(0.3, 1.5),
            ratio_range=(1.0, 1.0),  # tied — speed-only DR
        )
        return cfg
    if "Go2WarpSplitbeltPosTrackTiedDR" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbeltPosTrackTiedDR",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt_pos_track_tied_dr"),
            _splitbelt_pos_track_tied_dr_default_config,
        )

    # (MuJoCo Warp PushEnv removed 2026-04-20 — replaced by vendored pymunk
    # gym-pusht (`jax_rl/envs/manipulation/pusht/`) for cross-shape work.)

    # Factory PegInsert (Warp-backed manipulation env).
    # Panda 7-DoF arm + welded capsule peg + bore-tile-ring hole at 114µm clearance.
    # SDF substrate explored + retired (see .context/journals/2026-05-27-factory-phase0.md);
    # bore tiles + capsule_convex narrowphase is the load-bearing physics path.
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config as factory_peg_insert_default_config,
    )
    if "FactoryPegInsert" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "FactoryPegInsert",
            FactoryPegInsert,
            factory_peg_insert_default_config,
        )

    # Factory GearMesh (Warp-backed manipulation env).
    # Panda 7-DoF arm + welded medium gear (CoACD) + flanking gears
    # (hinge-constrained, CoACD) + plate primitive (box slab + 3 cylinder pegs).
    from jax_rl.envs.manipulation.factory.factory_gear_mesh import (
        FactoryGearMesh,
        default_config as factory_gear_mesh_default_config,
    )
    if "FactoryGearMesh" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "FactoryGearMesh",
            FactoryGearMesh,
            factory_gear_mesh_default_config,
        )


_register_custom_envs()


def maybe_load_custom_env(env_name: str):
    """Return a locally-defined MJX env instance for env_name, or None.

    For envs we hand-roll outside of `mujoco_playground.registry` (so they
    are NOT registered in pg_registry), this returns the constructed env.
    Callers should fall through to ``pg_registry.load`` when this returns
    None. The check is intentionally explicit (`if env is None`) at call
    sites — env structs can evaluate falsy under JAX dataclasses.
    """
    if env_name == "AntMJX":
        from jax_rl.envs.locomotion.ant import Ant
        return Ant()
    if env_name == "AntMJXClassic":
        from jax_rl.envs.locomotion.ant import Ant, default_config
        cfg = default_config()
        cfg.unlock()
        cfg.include_cfrc_ext_in_observation = False  # 27d obs (Gym v4-style, closer to DIAYN paper)
        return Ant(config=cfg)
    return None


def _safe_obs(obs, has_bad):
    """Zero out obs for envs with NaN/Inf. Works with flat arrays or dicts."""
    if isinstance(obs, dict):
        return {k: jnp.where(has_bad[:, None], 0.0, v) for k, v in obs.items()}
    return jnp.where(has_bad[:, None], 0.0, obs)


def _obs_has_bad(obs):
    """Check for NaN/Inf in obs. Works with flat arrays or dicts."""
    if isinstance(obs, dict):
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
        action = jnp.where(jnp.isnan(action) | jnp.isinf(action), 0.0, action)
        state = raw_step(state, action)
        has_bad = _obs_has_bad(state.obs)
        has_bad = has_bad | jnp.isnan(state.reward) | jnp.isinf(state.reward)
        safe_obs = _safe_obs(state.obs, has_bad)
        safe_reward = jnp.where(has_bad, 0.0, state.reward)
        safe_done = jnp.where(has_bad, 1.0, state.done)
        state = state.replace(obs=safe_obs, reward=safe_reward, done=safe_done)
        return state
    return safe_step


def make_envs(cfg: TrainConfig, seed: int):
    """Create MJX training env + eval env, JIT env.step, reset training env.

    Returns:
        env: wrapped training environment
        env_step: JIT'd, NaN-safe env.step function
        env_state: initial env state (reset with num_envs)
        eval_env: separate wrapped env for evaluation
        obs_dim: observation dimensionality
        action_dim: action dimensionality
        key: PRNG key (after reset_key consumption)
    """
    env = maybe_load_custom_env(cfg.env_name)
    if env is None:
        env = pg_registry.load(cfg.env_name)

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
        env = wrap_for_training(env, episode_length=cfg.episode_length, action_repeat=cfg.action_repeat)
    env_step = _make_nan_safe_step(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    eval_env = maybe_load_custom_env(cfg.env_name)
    if eval_env is None:
        eval_env = pg_registry.load(cfg.env_name)
    import dataclasses
    eval_cfg = cfg
    if cfg.action_delay_range_ms is not None:
        eval_cfg = dataclasses.replace(cfg, action_delay_ms=cfg.action_delay_range_ms[1], action_delay_range_ms=None)
    eval_env = apply_wrapper_pipeline(eval_env, eval_cfg)
    eval_env = wrap_for_training(eval_env, episode_length=cfg.episode_length, action_repeat=cfg.action_repeat)

    if isinstance(env_state.obs, dict):
        obs_dim = env_state.obs["state"].shape[-1]
    else:
        obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    return env, env_step, env_state, eval_env, obs_dim, action_dim, key


def make_mjx_env_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Build an MJX EnvBundle (Playground / MJWarp) for the given config.

    This is the original `make_env_bundle` body, just relocated to its
    dedicated backend module. Registered as the "mjx" backend at import time.
    """
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = False
    critic_obs_dim = None

    if dict_obs:
        has_privileged = "privileged_state" in env_state.obs
        if has_privileged:
            critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
            print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs detected: using 'state' key ({obs_dim}d) for off-policy")

    # Pick up env-supplied training-loop hooks if the env class provides them
    # (e.g. WarpJoystickCurriculum exposes terrain log helpers). Generic loops
    # call these via the bundle so they don't have to import locomotion modules.
    extra_metrics_fn = getattr(env, "log_extra_metrics", None)
    extra_image_fn = getattr(env, "log_extra_image", None)
    debug_dump_fn = getattr(env, "print_debug_dump", None)

    return EnvBundle(
        env=env, env_step=env_step, env_state=env_state, eval_env=eval_env,
        obs_dim=obs_dim, action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        has_privileged=has_privileged,
        dict_obs=dict_obs,
        key=key,
        backend_kind="mjx",
        num_envs=cfg.num_envs,
        render_fn=None,   # MJX render lives in record_video.py for now (Phase 5).
        extra_metrics_fn=extra_metrics_fn,
        extra_image_fn=extra_image_fn,
        debug_dump_fn=debug_dump_fn,
    )


# Register on import.
from jax_rl.training.env_backends import register_backend
register_backend("mjx", make_mjx_env_bundle)
