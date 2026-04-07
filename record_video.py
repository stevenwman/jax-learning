"""Record a video of any trained policy.

Usage:
  MUJOCO_GL=egl uv run python record_video.py                    # random policy
  MUJOCO_GL=egl uv run python record_video.py --checkpoint ckpt  # trained policy

Works with ALL algos (PPO, SAC, TD3, FastTD3, FastSAC, FastDSAC) — reads meta.json
to determine algo type and reconstruct the actor network automatically.

Two-phase approach:
  1. JIT-scan the rollout on GPU (fast — collects all states)
  2. Render frames on CPU from saved states (slow but unavoidable)
"""

import argparse
from datetime import datetime
import os
import time

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

    if algo == "ppo":
        from jax_rl.algos.ppo import PPO
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
           kicks: bool = False):

    # ── Load checkpoint ───────────────────────────────────────────────────
    algo_type = "ppo"  # default
    norm_state = None
    actor_params = None

    if checkpoint is not None:
        meta, actor_params, norm_state = load_actor_for_inference(checkpoint)
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
        use_obs_norm = False
        print("Using random (untrained) policy")

    env_name = env_name or "CartpoleBalance"
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
    if norm_state is None:
        from jax_rl.utils.normalization import init as norm_init
        norm_state = norm_init(obs_dim)

    # ── Build rollout step function ───────────────────────────────────────
    kicks_fn = apply_kicks if kicks else None

    if algo_type == "ppo":
        rollout_step, _ = build_ppo_rollout_step(
            algo, training_state, norm_state, env_step, kicks_fn=kicks_fn)
        init_carry = (env_state, norm_state, key)
    else:
        rollout_step, _ = build_offpolicy_rollout_step(
            algo, training_state.actor_params, norm_state, env_step,
            use_obs_norm, kicks_fn=kicks_fn)
        init_carry = (env_state, key)

    # ── Phase 1: JIT-scan rollout on GPU ──────────────────────────────────
    print("JIT-compiling rollout scan...")
    t0 = time.time()
    _, (trajectory, actions) = jax.lax.scan(rollout_step, init_carry, jnp.arange(max_steps), length=max_steps)
    jax.block_until_ready(trajectory.obs)
    t_rollout = time.time() - t0
    print(f"Rollout done: {max_steps} steps in {t_rollout:.2f}s (includes JIT compilation)")

    total_reward = float(trajectory.reward.sum())
    print(f"Total reward: {total_reward:.1f}")

    # Find first done step
    dones = np.asarray(trajectory.done)
    done_indices = np.where(dones > 0.5)[0]
    if len(done_indices) > 0:
        num_frames = int(done_indices[0]) + 1
        print(f"Episode ended at step {num_frames}")
    else:
        num_frames = max_steps
        print(f"Episode ran full {max_steps} steps (no termination)")

    # ── Phase 2: Render frames on CPU ─────────────────────────────────────
    print(f"Rendering {num_frames + 1} frames (CPU)...")
    t0 = time.time()

    states = [env_state]
    for i in range(num_frames):
        state_i = jax.tree.map(lambda x: x[i], trajectory)
        states.append(state_i)

    # Render with mujoco.Renderer for better quality + resolution control
    import mujoco
    renderer = mujoco.Renderer(env.mj_model, width=640, height=480)
    frames = []
    mj_data = mujoco.MjData(env.mj_model)

    has_commands = hasattr(trajectory, 'info') and 'command' in trajectory.info

    for idx, state_i in enumerate(tqdm(states)):
        mj_data.qpos[:] = np.array(state_i.data.qpos)
        mj_data.qvel[:] = np.array(state_i.data.qvel)
        mujoco.mj_forward(env.mj_model, mj_data)
        if camera is not None:
            renderer.update_scene(mj_data, camera=camera)
        else:
            # Free camera with body tracking
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 1  # base/base_link (body ID 1 in both models)
            cam.distance = 2.0
            cam.azimuth = 135
            cam.elevation = -20
            renderer.update_scene(mj_data, camera=cam)

        # Add command arrow overlays (Go2-specific)
        if has_commands and idx > 0 and idx <= len(trajectory.info['command']):
            cmd = np.array(trajectory.info['command'][idx - 1])
            render_command_overlays(renderer, mj_data, cmd, idx)

        frames.append(renderer.render())
    renderer.close()
    t_render = time.time() - t0
    print(f"Render done: {len(frames)} frames in {t_render:.2f}s "
          f"({t_render / len(frames) * 1000:.1f}ms/frame)")

    # ── Save video ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint is not None:
        video_path = os.path.join(checkpoint, f"{timestamp}_rollout.mp4")
    else:
        video_path = out if out != "rollout.mp4" else f"{timestamp}_rollout.mp4"

    print(f"Saving to {video_path}...")
    imageio.mimsave(video_path, frames, fps=50)  # 50Hz policy = 50fps for real-time
    print(f"Done: {video_path}")

    # ── Save trajectory .npz for offline analysis ────────────────────────
    npz_path = video_path.replace(".mp4", "_traj.npz")
    traj_data = {
        "qpos": np.array(trajectory.data.qpos[:num_frames]),
        "qvel": np.array(trajectory.data.qvel[:num_frames]),
        "actions": np.array(actions[:num_frames]),
        "rewards": np.array(trajectory.reward[:num_frames]),
    }
    # Save reward components if available
    if hasattr(trajectory, 'info') and 'reward_components' in trajectory.info:
        for k, v in trajectory.info['reward_components'].items():
            traj_data[f"reward_{k}"] = np.array(v[:num_frames])
    # Save commands if available
    if hasattr(trajectory, 'info') and 'command' in trajectory.info:
        traj_data["commands"] = np.array(trajectory.info['command'][:num_frames])

    np.savez_compressed(npz_path, **traj_data)
    print(f"Trajectory saved: {npz_path} ({len(traj_data)} arrays)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--kicks", action="store_true",
                        help="Zero velocity command + random velocity kicks every 1.5s")
    args = parser.parse_args()
    record(
        env_name=args.env, checkpoint=args.checkpoint, out=args.out,
        max_steps=args.max_steps,
        camera=args.camera, video_seed=args.seed,
        kicks=args.kicks,
    )
