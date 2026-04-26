"""Record a video of any trained policy.

Usage:
  uv run python record_video.py                    # random policy
  uv run python record_video.py --checkpoint ckpt  # trained policy
  MUJOCO_GL=osmesa uv run python record_video.py   # force osmesa if EGL unavailable

Works with ALL algos (PPO, PPOContraction, SAC, TD3, FastSAC, FastTD3, FlashSAC, TDMPC2) — reads meta.json
to determine algo type and reconstruct the actor network automatically.

Two-phase approach:
  1. JIT-scan the rollout on GPU (fast — collects all states)
  2. Render frames on CPU from saved states (slow but unavoidable)
"""

import argparse
from datetime import datetime
import os
import time

# Default to EGL for headless rendering (no DISPLAY required).
# Override with MUJOCO_GL=osmesa if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")

# On-demand GPU allocation: preallocator locks out Warp's CUDA graph capture
# (~1-2 GiB transient) on contested GPUs. One-shot inference gains nothing
# from preallocation anyway.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax

from mujoco_playground import registry as pg_registry

from jax_rl.training.checkpointing import load_actor_for_inference
# Register custom envs (Go2 etc.) with Playground's registry.
import jax_rl.training.env_setup  # noqa: F401 — side effect: registers custom envs
from jax_rl.training.env_backends import detect_backend
from jax_rl.utils.normalization import normalize as norm_normalize
from jax_rl.utils.rollout import build_ppo_rollout_step, build_offpolicy_rollout_step
from jax_rl.envs.locomotion.go2_rendering import apply_kicks, render_command_overlays


ENV_DEFAULTS = {
    "CartpoleBalance":  ((64, 64), "fixed"),
    "CheetahRun":       ((256, 256), "side"),
    "WalkerWalk":       ((256, 256), "side"),
    "WalkerRun":        ((256, 256), "side"),
    "HumanoidRun":      ((256, 256), "side"),
    "HumanoidWalk":     ((256, 256), "side"),
    "HumanoidStand":    ((256, 256), "side"),
    "Go2WarpJoystickFlat": ((480, 480), None),  # no named camera — use free cam
}


def _build_select_action(meta, obs_dim, action_dim):
    """Build a select_action function from checkpoint metadata. Algo-agnostic.

    Returns:
        select_action_fn(actor_params, obs, key, deterministic) -> action
    """
    algo = meta.get("algo", "ppo")
    tc = meta.get("train_config", {})
    # PPO config may be nested under "ppo" key.
    ppo_tc = tc.get("ppo", {}) if tc.get("ppo") else {}
    dummy_opt = optax.adam(1e-3)

    if algo in ("ppo", "ppocontr"):
        from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
        policy_dim = tuple(ppo_tc.get("policy_hidden_dim", None) or tc.get("policy_hidden_dim", None) or (32, 32, 32, 32))
        value_dim = tuple(ppo_tc.get("value_hidden_dim", None) or tc.get("value_hidden_dim", None) or (256, 256, 256, 256, 256))
        activation = ppo_tc.get("activation", None) or tc.get("activation", "swish")
        squash = ppo_tc.get("squash", tc.get("squash", True))
        state_dep_std = ppo_tc.get("state_dependent_std", tc.get("state_dependent_std", False))
        config = PPOConfig(
            encoder=EncoderConfig(obs_dim=obs_dim,
                                  hidden_dim=policy_dim,
                                  activation=activation),
            critic_encoder=EncoderConfig(obs_dim=obs_dim,
                                         hidden_dim=value_dim,
                                         activation=activation),
            policy_head=PolicyHeadConfig(action_dim=action_dim,
                                         squash=squash,
                                         state_dependent_std=state_dep_std),
            num_envs=1,
        )
        # PPOContraction uses the same Actor/VCritic as PPO for the action-selection
        # path; metric network is not needed for rollout recording.
        if algo == "ppocontr":
            from jax_rl.algos import PPOContraction
            ppo = PPOContraction(config, obs_dim, action_dim, dummy_opt, dummy_opt)
        else:
            from jax_rl.algos.ppo import PPO
            ppo = PPO(config, obs_dim, action_dim, dummy_opt, dummy_opt)
        return ppo, "ppo"

    elif algo in ("sac", "fast_sac"):
        algo_cfg_key = "sac_config" if "sac_config" in meta else "fast_sac_config"
        sc = meta.get(algo_cfg_key, {})
        if algo == "fast_sac":
            from jax_rl.algos.fast_sac import FastSAC
            from jax_rl.configs.fast_sac_config import FastSACConfig
            fast_sac_cfg = FastSACConfig(
                hidden_dim=tuple(sc.get("hidden_dim", (512, 256, 128))),
                activation=sc.get("activation", "swish"),
                q_layer_norm=sc.get("q_layer_norm", True),
                target_entropy_scale=sc.get("target_entropy_scale", 0.0),
                num_atoms=sc.get("num_atoms", 101),
                v_min=sc.get("v_min", -20.0),
                v_max=sc.get("v_max", 20.0),
                q_aggregation=sc.get("q_aggregation", "avg"),
                critic_hidden_dim=tuple(sc["critic_hidden_dim"]) if sc.get("critic_hidden_dim") else None,
            )
            sac = FastSAC(fast_sac_cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
        else:
            from jax_rl.algos.sac import SAC
            from jax_rl.configs.sac_config import SACConfig
            sac_cfg = SACConfig(
                hidden_dim=tuple(sc.get("hidden_dim", (256, 256))),
                activation=sc.get("activation", "relu"),
                q_layer_norm=sc.get("q_layer_norm", True),
                target_entropy_scale=sc.get("target_entropy_scale", 0.5),
            )
            sac = SAC(sac_cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
        return sac, "offpolicy"

    elif algo == "flash_sac":
        from jax_rl.algos.flash_sac import FlashSAC
        from jax_rl.configs.flash_sac_config import FlashSACConfig
        sc = meta.get("flash_sac_config", {})
        flash_cfg = FlashSACConfig(
            num_blocks=sc.get("num_blocks", 2),
            actor_hidden_dim=sc.get("actor_hidden_dim", 128),
            critic_hidden_dim=sc.get("critic_hidden_dim", 256),
            expansion=sc.get("expansion", 4),
            num_atoms=sc.get("num_atoms", 101),
            v_min=sc.get("v_min", -5.0),
            v_max=sc.get("v_max", 5.0),
            sigma_target=sc.get("sigma_target", 0.15),
        )
        flash = FlashSAC(flash_cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
        return flash, "offpolicy"

    elif algo in ("td3", "fast_td3"):
        from jax_rl.configs.td3_config import TD3Config
        algo_cfg_key = "td3_config" if "td3_config" in meta else "fast_td3_config"
        tc_algo = meta.get(algo_cfg_key, {})
        if algo == "fast_td3":
            from jax_rl.algos.fast_td3 import FastTD3
            from jax_rl.configs.fast_td3_config import FastTD3Config
            td3_cfg = FastTD3Config(
                hidden_dim=tuple(tc_algo.get("hidden_dim", (512, 512))),
                activation=tc_algo.get("activation", "relu"),
                q_layer_norm=tc_algo.get("q_layer_norm", True),
            )
            td3 = FastTD3(td3_cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
        else:
            td3_cfg = TD3Config(
                hidden_dim=tuple(tc_algo.get("hidden_dim", (256, 256))),
                activation=tc_algo.get("activation", "relu"),
                q_layer_norm=tc_algo.get("q_layer_norm", False),
            )
            from jax_rl.algos.td3 import TD3
            td3 = TD3(td3_cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
        return td3, "offpolicy"

    else:
        raise ValueError(f"Unknown algo: {algo}")


def record(env_name: str | None = None, checkpoint: str | None = None,
           out: str = "rollout.mp4", max_steps: int = 1000,
           camera: str | None = None, video_seed: int = 0,
           kicks: bool = False,
           terrain_level: int | None = None,
           terrain_type: str | None = None,
           force_zero_linvel: bool = False,
           force_zero_yaw: bool = False):

    # ── Load checkpoint ───────────────────────────────────────────────────
    algo_type = "ppo"  # default
    norm_state = None
    actor_params = None

    if checkpoint is not None:
        meta, actor_params, norm_state, actor_batch_stats = load_actor_for_inference(checkpoint)
        algo_name = meta.get("algo", "ppo")
        env_name = env_name or meta.get("train_config", {}).get("env_name")
        # Detect if training used obs normalization (stored in algo config)
        algo_cfg_keys = [f"{algo_name}_config", "sac_config", "fast_sac_config",
                         "td3_config", "fast_td3_config", "ppo"]
        use_obs_norm = False
        for k in algo_cfg_keys:
            if k in meta and meta[k].get("obs_normalization", False):
                use_obs_norm = True
                break
        print(f"Loaded checkpoint: algo={algo_name}, env={env_name}, obs_norm={use_obs_norm}")
    else:
        meta = {}
        algo_name = "ppo"
        actor_batch_stats = None
        use_obs_norm = False
        print("Using random (untrained) policy")

    env_name = env_name or "CartpoleBalance"

    # ── Dispatch by backend ──────────────────────────────────────────────
    backend = detect_backend(env_name)
    if backend == "gym":
        return _record_gym(
            env_name=env_name, meta=meta,
            actor_params=actor_params, norm_state=norm_state,
            actor_batch_stats=actor_batch_stats,
            checkpoint=checkpoint, out=out,
            max_steps=max_steps, video_seed=video_seed,
        )

    defaults = ENV_DEFAULTS.get(env_name, ((256, 256), None))
    camera = camera or defaults[1]

    # ── Create env (unwrapped — single env, no auto-reset) ────────────────
    env = pg_registry.load(env_name)

    # Apply wrapper pipeline from checkpoint config (action delay, frame stacking, etc.)
    train_cfg = meta.get("train_config", {})
    # For recording, use fixed delay (max of range if randomized)
    if train_cfg.get("action_delay_range_ms"):
        train_cfg = {**train_cfg, "action_delay_ms": train_cfg["action_delay_range_ms"][1], "action_delay_range_ms": None}
    from jax_rl.envs.wrappers import apply_wrapper_pipeline, build_wrapper_pipeline
    pipeline = build_wrapper_pipeline(train_cfg)
    if pipeline:
        env = apply_wrapper_pipeline(env, train_cfg)
        print(f"  Wrappers: {[name for name, _, _ in pipeline]}")

    env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(video_seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(reset_key)

    # ── Curriculum-env debug override (optional) ──────────────────────────
    # Force spawn at a specific (terrain_level, terrain_type) tile. Useful for
    # isolating failure modes: e.g., "does policy handle pyramid_down L7?".
    if terrain_level is not None or terrain_type is not None:
        from jax_rl.envs.locomotion.curriculum_logging import TERRAIN_TYPE_NAMES as _TERRAIN_TYPE_NAMES
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        if not hasattr(base_env, "_terrain_origins"):
            print(f"[warn] --terrain-level/--terrain-type ignored: env {env_name} is not a curriculum env")
        else:
            tl = int(terrain_level) if terrain_level is not None else 0
            if terrain_type is not None:
                if terrain_type not in _TERRAIN_TYPE_NAMES:
                    raise ValueError(f"--terrain-type must be one of {_TERRAIN_TYPE_NAMES}, got '{terrain_type}'")
                tt = _TERRAIN_TYPE_NAMES.index(terrain_type)
            else:
                tt = int(env_state.info.get("terrain_type", 0))
            tl = max(0, min(tl, base_env._num_rows - 1))
            tt = max(0, min(tt, base_env._num_cols - 1))
            print(f"  [curriculum override] terrain_level={tl}, terrain_type={_TERRAIN_TYPE_NAMES[tt]} (col {tt})")

            # Resample spawn + goal at target tile.
            key, spawn_rng, yaw_rng = jax.random.split(key, 3)
            spawn_local, goal_local, spawn_yaw = base_env._sample_spawn_goal(
                jnp.int32(tt), base_env._tile_size, spawn_rng, yaw_rng
            )
            tile_origin = base_env._terrain_origins[tl, tt]
            spawn_world_xy = spawn_local[:2] + tile_origin[:2]
            spawn_world_z = tile_origin[2] + spawn_local[2]
            goal_world_xy = goal_local[:2] + tile_origin[:2]

            new_qpos = env_state.data.qpos.at[0].set(spawn_world_xy[0])
            new_qpos = new_qpos.at[1].set(spawn_world_xy[1])
            new_qpos = new_qpos.at[2].set(spawn_world_z)
            new_qpos = new_qpos.at[3].set(jnp.cos(spawn_yaw / 2.0))
            new_qpos = new_qpos.at[4].set(0.0)
            new_qpos = new_qpos.at[5].set(0.0)
            new_qpos = new_qpos.at[6].set(jnp.sin(spawn_yaw / 2.0))
            env_state = env_state.replace(data=env_state.data.replace(qpos=new_qpos))
            env_state.info["terrain_level"] = jnp.int32(tl)
            env_state.info["terrain_type"] = jnp.int32(tt)
            env_state.info["goal_xy"] = goal_world_xy
            env_state.info["initial_distance"] = jnp.linalg.norm(spawn_world_xy - goal_world_xy)
            env_state.info["episode_reached_goal"] = jnp.bool_(False)
            env_state.info["episode_min_distance"] = env_state.info["initial_distance"]
            env_state.info["episode_fallen"] = jnp.bool_(False)
            env_state.info["target_speed"] = jnp.float32(
                0.5 + tl / max(1, base_env._num_rows - 1) * 0.5
            )
            if force_zero_linvel:
                env_state.info["force_zero_linvel"] = jnp.bool_(True)
                print("  [curriculum override] force_zero_linvel=True (Class A cmd_vx=cmd_vy=0)")
            if force_zero_yaw:
                env_state.info["force_zero_yaw"] = jnp.bool_(True)
                print("  [curriculum override] force_zero_yaw=True (cmd_yaw_rate=0)")

    raw_obs = env_state.obs
    policy_obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
    obs_dim = policy_obs.shape[-1]
    action_dim = env.action_size

    # ── Build algo for select_action ──────────────────────────────────────
    algo, algo_type = _build_select_action(meta, obs_dim, action_dim)

    # Init dummy state to get correct param structure, then replace with loaded params
    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    if actor_params is not None:
        training_state = training_state.replace(actor_params=actor_params)
    if actor_batch_stats is not None and hasattr(training_state, "actor_batch_stats"):
        training_state = training_state.replace(actor_batch_stats=actor_batch_stats)
        algo._default_actor_bs = actor_batch_stats
    if norm_state is None:
        from jax_rl.utils.normalization import init as norm_init
        norm_state = norm_init(obs_dim)

    # ── Build rollout step function ───────────────────────────────────────
    kicks_fn = apply_kicks if kicks else None

    # Frame-stack depth from saved training config (defaults to 1 for non-stacked envs).
    tc = meta.get("train_config", {}) if isinstance(meta, dict) else {}
    n_fs = int(tc.get("n_frame_stack", 1))

    if algo_type == "ppo":
        rollout_step, _ = build_ppo_rollout_step(
            algo, training_state, norm_state, env_step,
            kicks_fn=kicks_fn, n_frame_stack=n_fs)
    else:
        rollout_step, _ = build_offpolicy_rollout_step(
            algo, training_state.actor_params, norm_state, env_step,
            use_obs_norm, kicks_fn=kicks_fn, n_frame_stack=n_fs)
    init_carry = (env_state, key)

    # ── Phase 1: Python-loop rollout (low peak HBM) ───────────────────────
    # A jit'd scan would preallocate full-State × max_steps on device (~1 GB
    # for Go2). Calling the jit'd step in a Python loop reuses one State
    # buffer and only retains numpy copies of qpos/qvel/action/reward/info
    # on host. Compile cost: ~one rollout_step trace (5-10s). Dispatch
    # overhead: ~100ms total across 1000 steps (negligible vs CPU render).
    jit_rollout_step = jax.jit(rollout_step)

    init_qpos = np.asarray(env_state.data.qpos)
    init_qvel = np.asarray(env_state.data.qvel)

    qpos_hist, qvel_hist, act_hist, rew_hist = [], [], [], []
    cmd_hist = []
    goal_xy_hist = []
    reward_components_hist: dict[str, list] = {}

    print("JIT-compiling rollout step + running Python loop...")
    t0 = time.time()
    carry = init_carry
    num_frames = max_steps
    for i in range(max_steps):
        carry, (state_i, action_i) = jit_rollout_step(carry, i)
        qpos_hist.append(np.asarray(state_i.data.qpos))
        qvel_hist.append(np.asarray(state_i.data.qvel))
        act_hist.append(np.asarray(action_i))
        rew_hist.append(float(state_i.reward))
        info = state_i.info
        if 'command' in info:
            cmd_hist.append(np.asarray(info['command']))
        # Only record goal_xy for goal-directed (Class B) tiles — Class A uses
        # a placeholder goal=spawn that shouldn't be visualized.
        if 'goal_xy' in info and bool(info.get('is_goal_directed', True)):
            goal_xy_hist.append(np.asarray(info['goal_xy']))
        else:
            goal_xy_hist.append(None)
        if 'reward_components' in info:
            for k, v in info['reward_components'].items():
                reward_components_hist.setdefault(k, []).append(np.asarray(v))
        if float(state_i.done) > 0.5:
            num_frames = i + 1
            print(f"Episode ended at step {num_frames}")
            break
    t_rollout = time.time() - t0
    print(f"Rollout done: {num_frames} steps in {t_rollout:.2f}s (includes JIT compilation)")

    total_reward = float(np.sum(rew_hist[:num_frames]))
    print(f"Total reward: {total_reward:.1f}")
    if num_frames == max_steps:
        print(f"Episode ran full {max_steps} steps (no termination)")

    # ── Phase 2: Render frames on CPU ─────────────────────────────────────
    print(f"Rendering {num_frames + 1} frames (CPU)...")
    t0 = time.time()

    # Initial state + post-step states
    all_qpos = [init_qpos] + qpos_hist[:num_frames]
    all_qvel = [init_qvel] + qvel_hist[:num_frames]

    # Render with mujoco.Renderer for better quality + resolution control
    import mujoco
    renderer = mujoco.Renderer(env.mj_model, width=640, height=480)
    frames = []
    mj_data = mujoco.MjData(env.mj_model)

    has_commands = len(cmd_hist) > 0

    for idx in tqdm(range(len(all_qpos))):
        mj_data.qpos[:] = all_qpos[idx]
        mj_data.qvel[:] = all_qvel[idx]
        mujoco.mj_forward(env.mj_model, mj_data)
        if camera is not None:
            renderer.update_scene(mj_data, camera=camera)
        else:
            # Free camera with body tracking
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 1  # base/base_link (body ID 1 in both models)
            cam.distance = 6.0  # close tracking (was 12.0)
            cam.azimuth = 135
            cam.elevation = -30
            renderer.update_scene(mj_data, camera=cam)

        # Add command arrow overlays (Go2-specific)
        if has_commands and idx > 0 and idx <= len(cmd_hist):
            cmd = cmd_hist[idx - 1]
            goal_xy = goal_xy_hist[idx - 1] if goal_xy_hist and idx <= len(goal_xy_hist) else None
            render_command_overlays(renderer, mj_data, cmd, idx, goal_xy=goal_xy)

        frames.append(renderer.render())
    renderer.close()
    t_render = time.time() - t0
    print(f"Render done: {len(frames)} frames in {t_render:.2f}s "
          f"({t_render / len(frames) * 1000:.1f}ms/frame)")

    # ── Save video ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out != "rollout.mp4":
        # User explicitly set --out; honor it verbatim.
        video_path = out
    elif checkpoint is not None:
        video_path = os.path.join(checkpoint, f"{timestamp}_rollout.mp4")
    else:
        video_path = f"{timestamp}_rollout.mp4"

    print(f"Saving to {video_path}...")
    imageio.mimsave(video_path, frames, fps=50)  # 50Hz policy = 50fps for real-time
    print(f"Done: {video_path}")

    # ── Save trajectory .npz for offline analysis ────────────────────────
    npz_path = video_path.replace(".mp4", "_traj.npz")
    traj_data = {
        "qpos": np.stack(qpos_hist[:num_frames]),
        "qvel": np.stack(qvel_hist[:num_frames]),
        "actions": np.stack(act_hist[:num_frames]),
        "rewards": np.asarray(rew_hist[:num_frames], dtype=np.float32),
    }
    for k, vs in reward_components_hist.items():
        traj_data[f"reward_{k}"] = np.stack(vs[:num_frames])
    if cmd_hist:
        traj_data["commands"] = np.stack(cmd_hist[:num_frames])

    np.savez_compressed(npz_path, **traj_data)
    print(f"Trajectory saved: {npz_path} ({len(traj_data)} arrays)")


def _record_gym(env_name, meta, actor_params, norm_state, actor_batch_stats,
                checkpoint, out, max_steps, video_seed):
    """Record a rollout for a gym-backend env (PushT etc.).

    Single-env Python loop; uses env.render(mode='rgb_array') per step
    to capture frames. Saves mp4 + traj npz like the MJX path, but with
    obs/actions/rewards instead of qpos/qvel.
    """
    # Build the gym env via the backend factory using stored env_kwargs.
    from jax_rl.configs.train_config import TrainConfig
    from jax_rl.training.env_backends.gym_backend import GYM_ENV_FACTORIES

    if env_name not in GYM_ENV_FACTORIES:
        raise ValueError(
            f"Gym env {env_name!r} has no registered factory. Known: "
            f"{sorted(GYM_ENV_FACTORIES)}"
        )

    train_cfg = meta.get("train_config", {})
    env_kwargs = dict(train_cfg.get("env_kwargs") or {})
    cfg = TrainConfig(env_name=env_name, num_envs=1, env_kwargs=env_kwargs)
    make_env_thunk = GYM_ENV_FACTORIES[env_name](cfg)
    env = make_env_thunk()

    obs, _ = env.reset(seed=int(video_seed))
    obs_dim = obs.shape[-1] if not isinstance(obs, dict) else obs["state"].shape[-1]
    action_dim = int(np.asarray(env.action_space.sample()).shape[-1])

    # Build the actor (algo-agnostic — same factory as MJX path).
    if actor_params is not None:
        algo, kind = _build_select_action(meta, obs_dim, action_dim)
        if kind == "ppo":
            select_action = algo.select_action
        else:
            select_action = algo.select_action
    else:
        algo, kind = None, "random"
        select_action = None

    use_obs_norm = False
    for k in ("sac_config", "fast_sac_config", "td3_config", "fast_td3_config",
              "ppo", "flash_sac_config", "tdmpc2_config"):
        if k in meta and meta[k].get("obs_normalization", False):
            use_obs_norm = True
            break

    # ── Rollout (Python loop, one env) ───────────────────────────────────
    print(f"Rollout: {env_name} (gym backend), max {max_steps} steps...")
    t0 = time.time()
    obs_hist, act_hist, rew_hist = [obs], [], []
    frames = [env.render()]

    key = jax.random.PRNGKey(video_seed)
    for step in range(max_steps):
        key, ak = jax.random.split(key)
        if select_action is not None:
            obs_arr = obs["state"] if isinstance(obs, dict) else obs
            obs_jax = jnp.asarray(obs_arr)[None]
            if use_obs_norm and norm_state is not None:
                obs_jax = norm_normalize(norm_state, obs_jax)
            action = np.asarray(
                select_action(actor_params, obs_jax, ak, deterministic=True)
            )[0]
        else:
            action = np.asarray(env.action_space.sample(), dtype=np.float32)
        obs, r, term, trunc, info = env.step(action.astype(np.float32))
        obs_hist.append(obs); act_hist.append(action); rew_hist.append(float(r))
        frames.append(env.render())
        if term or trunc:
            print(f"Episode ended at step {step + 1} (term={term}, trunc={trunc})")
            break

    t_rollout = time.time() - t0
    print(f"Rollout + render: {len(frames)} frames in {t_rollout:.2f}s "
          f"({t_rollout / max(len(frames), 1) * 1000:.1f}ms/frame)")
    print(f"Total reward: {sum(rew_hist):.1f}")
    if "coverage" in info:
        print(f"Final coverage: {info['coverage']:.3f}")

    # ── Save video + traj ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out != "rollout.mp4":
        video_path = out
    elif checkpoint is not None:
        video_path = os.path.join(checkpoint, f"{timestamp}_rollout.mp4")
    else:
        video_path = f"{timestamp}_rollout.mp4"
    print(f"Saving to {video_path}...")
    imageio.mimsave(video_path, frames, fps=int(env.metadata.get("render_fps", 30)))
    print(f"Done: {video_path}")

    npz_path = video_path.replace(".mp4", "_traj.npz")
    obs_arr = np.stack([o["state"] if isinstance(o, dict) else o for o in obs_hist])
    np.savez_compressed(
        npz_path,
        obs=obs_arr,
        actions=np.stack(act_hist) if act_hist else np.zeros((0, action_dim), dtype=np.float32),
        rewards=np.asarray(rew_hist, dtype=np.float32),
    )
    print(f"Trajectory saved: {npz_path}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Importable for docs/tooling without parse_args()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--kicks", action="store_true",
                        help="Zero velocity command + random velocity kicks every 1.5s")
    parser.add_argument("--force-zero-linvel", action="store_true",
                        help="Curriculum Class A only: force cmd_vx=cmd_vy=0 for the whole episode (DR sanity check)")
    parser.add_argument("--force-zero-yaw", action="store_true",
                        help="Curriculum: force cmd_yaw_rate=0 for the whole episode (DR sanity check)")
    parser.add_argument("--terrain-level", type=int, default=None,
                        help="Curriculum env only: force spawn at this level (0-9)")
    parser.add_argument("--terrain-type", type=str, default=None,
                        choices=[None, "rough", "pyramid_up", "pyramid_down", "tilted", "flat"],
                        help="Curriculum env only: force spawn at this terrain type")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    record(
        env_name=args.env, checkpoint=args.checkpoint, out=args.out,
        max_steps=args.max_steps,
        camera=args.camera, video_seed=args.seed,
        kicks=args.kicks,
        terrain_level=args.terrain_level,
        terrain_type=args.terrain_type,
        force_zero_linvel=args.force_zero_linvel,
        force_zero_yaw=args.force_zero_yaw,
    )
