"""Brax PPO baseline for Go2JoystickFlat.

Uses Playground's built-in Brax PPO — the exact same training pipeline that
works for Go1. If this produces walking, our custom PPO has a bug. If this
also fails, the env/reward is the problem.

This is a diagnostic script, not a permanent training script.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.55")

import functools
import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from mujoco_playground._src.wrapper import wrap_for_brax_training

# Register our Go2 env
from jax_rl.envs.locomotion.go2_joystick import Joystick, default_config


def progress_fn(step, metrics):
    if 'eval/episode_reward' in metrics:
        print(f"Step {step:>10,} | "
              f"Eval {metrics['eval/episode_reward']:.1f} | "
              f"Reward {metrics.get('training/reward', 0):.4f}")


def main():
    env = Joystick(task="flat_terrain")
    wrapped_env = wrap_for_brax_training(env, episode_length=1000)

    # Asymmetric actor-critic: policy sees "state", critic sees "privileged_state"
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    # Exact Go1JoystickFlatTerrain config from Playground
    make_policy, params, metrics = ppo_train.train(
        environment=wrapped_env,
        wrap_env=False,
        num_timesteps=50_000_000,
        num_envs=1024,
        episode_length=1000,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        unroll_length=20,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.3,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=network_factory,
        seed=42,
        num_evals=20,
        num_eval_envs=128,
        progress_fn=progress_fn,
    )

    print("=" * 60)
    print(f"Brax PPO baseline complete.")
    print(f"Final eval: {metrics.get('eval/episode_reward', 'N/A')}")

    # ── Save Brax params for offline analysis ────────────────────────
    import pickle
    import numpy as np
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"checkpoints/brax_go2_{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save params as pickle (JAX arrays serialize fine)
    with open(os.path.join(ckpt_dir, "brax_params.pkl"), "wb") as f:
        pickle.dump(params, f)

    # Save normalizer params separately
    normalizer_params = params[0]  # Brax stores (normalizer_params, policy_params)
    with open(os.path.join(ckpt_dir, "normalizer_params.pkl"), "wb") as f:
        pickle.dump(normalizer_params, f)

    print(f"Brax params saved to {ckpt_dir}/")

    # ── Record video from trained policy ────────────────────────────

    inference_fn = make_policy(params, deterministic=True)
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    state = jit_reset(jax.random.PRNGKey(99))
    rollout_states = []
    rollout_actions = []
    rollout_rewards = []
    rollout_reward_components = []
    rollout_commands = []
    total_reward = 0.0

    for i in range(1000):
        act_rng = jax.random.PRNGKey(i)
        action, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout_states.append(state.data)
        rollout_actions.append(np.array(action))
        rollout_rewards.append(float(state.reward))
        if 'reward_components' in state.info:
            rollout_reward_components.append({k: float(v) for k, v in state.info['reward_components'].items()})
        if 'command' in state.info:
            rollout_commands.append(np.array(state.info['command']))
        total_reward += float(state.reward)

    print(f"Rollout reward: {total_reward:.1f}")

    # Save trajectory npz
    traj_data = {
        "qpos": np.array([np.array(s.qpos) for s in rollout_states]),
        "qvel": np.array([np.array(s.qvel) for s in rollout_states]),
        "actions": np.array(rollout_actions),
        "rewards": np.array(rollout_rewards),
    }
    if rollout_commands:
        traj_data["commands"] = np.array(rollout_commands)
    if rollout_reward_components:
        for k in rollout_reward_components[0]:
            traj_data[f"reward_{k}"] = np.array([rc[k] for rc in rollout_reward_components])
    np.savez_compressed(os.path.join(ckpt_dir, "traj.npz"), **traj_data)
    print(f"Trajectory saved: {ckpt_dir}/traj.npz ({len(traj_data)} arrays)")

    # Render frames
    import mujoco
    import mediapy

    renderer = mujoco.Renderer(env.mj_model, width=640, height=480)
    frames = []
    for data_mjx in rollout_states:
        mj_data = mujoco.MjData(env.mj_model)
        mj_data.qpos[:] = np.array(data_mjx.qpos)
        mj_data.qvel[:] = np.array(data_mjx.qvel)
        mujoco.mj_forward(env.mj_model, mj_data)
        renderer.update_scene(mj_data, camera="track")
        frames.append(renderer.render())

    out_path = os.path.join(ckpt_dir, "rollout.mp4")
    mediapy.write_video(out_path, frames, fps=50)
    print(f"Video saved: {out_path}")


if __name__ == "__main__":
    main()
