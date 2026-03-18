"""Run Brax's own PPO on HumanoidRun as a baseline.

Usage:
  uv run python bench_brax_ppo.py              # full config (2048 envs)
  uv run python bench_brax_ppo.py --half-gpu   # half GPU budget (1024 envs)
"""

import argparse
import time
from brax.training.agents.ppo.train import train as ppo_train
from mujoco_playground import dm_control_suite
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground._src.wrapper import wrap_for_brax_training


def progress(step, metrics):
    if "eval/episode_reward" in metrics:
        print(f"  Step {step:>10,} | Return {metrics['eval/episode_reward']:.1f}")


parser = argparse.ArgumentParser()
parser.add_argument("--half-gpu", action="store_true", help="Halve num_envs and batch_size")
args = parser.parse_args()

env = dm_control_suite.load("HumanoidRun")
cfg = dm_control_suite_params.brax_ppo_config("HumanoidRun")

num_envs = cfg.num_envs
batch_size = cfg.batch_size
if args.half_gpu:
    num_envs = num_envs // 2
    batch_size = batch_size // 2

print("Brax PPO baseline — HumanoidRun")
print(f"  num_timesteps={cfg.num_timesteps:,}")
print(f"  num_envs={num_envs}, batch_size={batch_size}")
print(f"  unroll_length={cfg.unroll_length}, num_minibatches={cfg.num_minibatches}")
print(f"  num_updates_per_batch={cfg.num_updates_per_batch}")
print(f"  learning_rate={cfg.learning_rate}, entropy_cost={cfg.entropy_cost}")
print(f"  discounting={cfg.discounting}, reward_scaling={cfg.reward_scaling}")
print(f"  episode_length={cfg.episode_length}")

t0 = time.time()
make_policy, params, metrics = ppo_train(
    environment=env,
    num_timesteps=120_000_000,
    num_evals=cfg.num_evals,
    num_envs=num_envs,
    batch_size=batch_size,
    unroll_length=cfg.unroll_length,
    num_minibatches=cfg.num_minibatches,
    num_updates_per_batch=cfg.num_updates_per_batch,
    episode_length=cfg.episode_length,
    action_repeat=cfg.action_repeat,
    learning_rate=cfg.learning_rate,
    entropy_cost=cfg.entropy_cost,
    discounting=cfg.discounting,
    reward_scaling=cfg.reward_scaling,
    normalize_observations=cfg.normalize_observations,
    wrap_env_fn=wrap_for_brax_training,
    seed=0,
    progress_fn=progress,
)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")
