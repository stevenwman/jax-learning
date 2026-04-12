# Train Go2 Locomotion

*Intermediate — assumes you've completed the [Quickstart](../getting-started/quickstart.md).*

This tutorial covers training a Unitree Go2 to follow velocity commands. Expect ~30 minutes on a single GPU.

## Overview

A policy that takes joystick-style velocity commands (forward, lateral, yaw) and controls 12 joint motors to make the Go2 walk, turn, and stop on flat ground.

<video class="tutorial-video" autoplay loop muted playsinline preload="metadata">
  <source src="../../assets/videos/go2_joystick_walk.mp4" type="video/mp4">
</video>

## Prerequisites

- A machine with a CUDA-capable GPU (tested on RTX 4090)
- The `jax-learning` repo cloned and dependencies installed (`uv sync`)
- Basic familiarity with reinforcement learning concepts (reward, policy, episode)

## Step 1: Choose the Environment

We'll use **`Go2WarpJoystickFlat`** — a velocity-tracking task for the Unitree Go2 on flat terrain.

```
Go2WarpJoystickFlat
├── Backend: MuJoCo Warp (GPU-accelerated parallel simulation)
├── MJCF: unitree_mujoco's go2.xml (exact robot model from Unitree)
├── Collision: Full cylinder + box geometry (no simplifications)
├── Obs: dict with "state" (51d) and "privileged_state" (125d)
└── Action: 12d joint position targets (PD controller computes torques)
```

!!! note "Why Warp over MJX?"
    MuJoCo offers two JAX backends: **MJX** and **Warp**. We use Warp because it loads Unitree's exact MJCF file (go2.xml) with full collision geometry — cylinders for legs, boxes for the body. MJX requires a simplified Menagerie MJCF with convex hulls, which creates a sim-to-sim gap when deploying to the real robot. Training on Warp means the policy sees the same physics model it'll run on during deployment.

## Step 2: Pick an Algorithm and Preset

We'll use **FastSAC** — a high-UTD (update-to-data) ratio variant of SAC from [Seo et al. 2025](https://arxiv.org/abs/2512.01996). It trains faster than PPO on this task and produces smoother gaits.

The preset configures everything:

- **PD gains:** Kp=20, Kd=0.5 (matches Unitree's official gains)
- **Parallel envs:** 1024 environments running simultaneously on GPU
- **Critic architecture:** Tapered MLP (768-384-192) with C51 distributional heads
- **Actor architecture:** Tapered MLP (512-256-128)
- **UTD ratio:** 8 gradient updates per environment step

## Step 3: Train

```bash
uv run python train_fast_sac.py \
    --env Go2WarpJoystickFlat \
    --num-envs 1024 \
    --total-timesteps 20000000
```

This runs 20 million timesteps across 1024 parallel environments. On an RTX 4090, expect ~18,000 steps/second.

!!! tip "Useful flags"
    - `--wandb` — log metrics to Weights & Biases for experiment tracking
    - `--reset-mode per_step` — use `DomainRandWrapper` for per-episode domain randomization (randomized friction, mass, center-of-mass, motor strength, and more) declared by the env
    - `--seed 42` — set the random seed for reproducibility

## Step 4: Monitor Training

The training script prints evaluation results to stdout periodically. Look for lines starting with `EVAL`:

You should see the eval score climbing over time:

```
EVAL | steps=2000000  | mean=85.3  | std=12.1
EVAL | steps=5000000  | mean=168.7 | std=8.4
EVAL | steps=10000000 | mean=232.5 | std=6.2
EVAL | steps=18000000 | mean=272.1 | std=4.8
```

!!! note "What does the eval score mean?"
    The eval score is the undiscounted sum of weighted rewards over a full episode (1000 steps = 20 seconds of simulated time). It combines velocity tracking accuracy, energy efficiency, gait quality, and stability penalties. A score of **270+** indicates reliable locomotion with smooth gaits and accurate command tracking. The theoretical maximum depends on the reward weights but scores above 280 are rare.

## Step 5: Record a Video

Once training finishes (or you want to check a checkpoint mid-training):

```bash
MUJOCO_GL=egl uv run python record_video.py \
    --checkpoint checkpoints/<run-dir>
```

!!! tip
    `MUJOCO_GL=egl` tells MuJoCo to render using EGL (headless GPU rendering), which works over SSH without a display server.

The video is saved to the checkpoint directory. It shows the Go2 following randomized velocity commands.

## Step 6: What's Next

- **Try FlashSAC:** `uv run python train_flashsac.py --env Go2WarpJoystickFlat --seed 100` — uses inverted residual blocks, BatchNorm, and adaptive reward scaling. Eval 282.4 on Go2 at 10M steps (single seed — variance across seeds not yet characterized).
- **Deploy to real hardware:** See the [Sim-to-Real](sim2real.md) tutorial
- **Add domain randomization:** Append `--reset-mode per_step` to the training command — `DomainRandWrapper` applies the env's declared DR specs per episode, which transfers better to real robots
- **Try a custom task:** See [Custom Environment](custom-env.md) to build your own Go2 task
- **Understand the reward function:** See [Custom Rewards](custom-rewards.md) for how the 17 reward terms work together

## Dict Observations

The environment returns observations as a Python dict, not a flat array:

```python
obs = {
    "state": jax.Array,           # 51d — what the actor (policy) sees
    "privileged_state": jax.Array  # 125d — what the critic sees during training
}
```

The **"state"** group contains: local linear velocity (3d), gyroscope (3d), gravity vector (3d), linear velocity (3d), accelerometer (3d), joint position offsets from default pose (12d), joint velocities (12d), last action (12d), and velocity command (3d).

The **"privileged_state"** group includes everything in "state" plus: clean (noise-free) sensor readings, actuator forces, contact states, foot velocities, foot air times, and external forces. The critic uses this extra information during training, but only the 51d "state" is needed at deployment. See [Asymmetric Critic](asymmetric-critic.md) for details.
