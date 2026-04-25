"""Brax PPO on Go1 Joystick — sanity check.

Does Go1 walk with the same training budget (512 envs, 50M steps)?
If yes → Go2-specific issue. If no → training budget issue.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.55")

import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from mujoco_playground import registry as pg_registry
from mujoco_playground._src.wrapper import wrap_for_brax_training


def progress_fn(step, metrics):
    if 'eval/episode_reward' in metrics:
        print(f"Step {step:>10,} | Eval {metrics['eval/episode_reward']:.1f}")


def main():
    env = pg_registry.load("Go1JoystickFlatTerrain")
    wrapped_env = wrap_for_brax_training(env, episode_length=1000)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    make_policy, params, metrics = ppo_train.train(
        environment=wrapped_env,
        wrap_env=False,
        num_timesteps=100_000_000,
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
    print(f"Go1 Brax PPO baseline complete.")
    print(f"Final eval: {metrics.get('eval/episode_reward', 'N/A')}")


if __name__ == "__main__":
    main()
