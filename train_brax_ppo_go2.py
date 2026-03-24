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
              f"Reward {metrics.get('training/reward', 0):.2f}")


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
        num_envs=512,
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
    print("Brax PPO baseline complete.")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
