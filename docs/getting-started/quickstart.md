# Quickstart

Train a CartpoleBalance policy and record a video.

## Prerequisites

Make sure you've completed the [Installation](installation.md) steps and verified GPU access.

## 1. Train

Run PPO on CartpoleBalance with 64 parallel environments:

```bash
uv run python scripts/train_ppo_fast.py \
    --env CartpoleBalance \
    --num-envs 64 \
    --total-timesteps 500000
```

Eval scores are printed periodically. Expect **~950+** by end of training.

!!! tip "What does the eval score mean?"
    Eval score = average return over eval episodes. Each env defines its own reward scale, so scores aren't comparable across envs. CartpoleBalance maxes around 1000; CheetahRun around 900; Go2 joystick depends on episode length + reward weights.

```
eval/episode_reward: 342.1  (step 50000)
eval/episode_reward: 687.4  (step 150000)
eval/episode_reward: 951.2  (step 400000)
eval/episode_reward: 968.7  (step 500000)
```

!!! tip "Training speed"
    With 64 parallel envs on a modern GPU, 500k steps should finish in about 1-2 minutes.

A checkpoint is saved automatically to the `checkpoints/` directory.

## 2. Record a video

Render a video of your trained policy:

```bash
MUJOCO_GL=egl uv run python scripts/record_video.py \
    --checkpoint checkpoints/<latest>
```

Replace `<latest>` with the actual checkpoint folder name (it includes a timestamp).

!!! note "The `MUJOCO_GL=egl` prefix"
    This tells MuJoCo to render offscreen using EGL (GPU-based rendering without a display). You need this on headless servers or when running over SSH.

## What just happened?

1. **Environment** — `CartpoleBalance` is an MJX environment: a cart with a pole that the agent must keep upright. The physics runs entirely on GPU via JAX.
2. **Algorithm** — PPO (Proximal Policy Optimization) collected experience from 64 parallel environments, then updated the policy using that experience. This [on-policy](../glossary.md#on-policy) loop repeated until 500k total steps.
3. **Checkpoint** — The trained policy network weights were saved to disk so you can load them later for evaluation or deployment.
4. **Video** — `record_video.py` loaded the checkpoint, ran the policy in the environment, and rendered the result to an MP4 file.

## Next steps

- Read [Concepts](concepts.md) to understand the framework architecture
- Try a harder environment: [Train Locomotion](../tutorials/train-locomotion.md) walks through Go2 quadruped training
- Explore the [API Reference](../api/algos.md) for algorithm details
