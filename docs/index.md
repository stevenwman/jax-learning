# jax-learning

**A JAX-based reinforcement learning framework for training robot policies in simulation and deploying them on real hardware.**

jax-learning is built around the [Unitree Go2](https://www.unitree.com/go2/) quadruped robot. It provides GPU-accelerated environments, multiple RL algorithms, and a sim-to-real deployment pipeline — all in pure JAX for maximum speed.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline style="width: 100%; border-radius: 8px;">
  <source src="assets/videos/go2_joystick_walk.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-size: 0.85em; color: gray;">Go2 locomotion — FastSAC, eval 276.5</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline style="width: 100%; border-radius: 8px;">
  <source src="assets/videos/go2_bongo_handstand.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-size: 0.85em; color: gray;">Bongo board handstand — PPO</p>
</div>
</div>

## Key Features

- **5 RL algorithms** — PPO, SAC, TD3, FastSAC, and FastTD3 (distributional critics with large-batch training)
- **GPU-accelerated environments** — Two backends: MJX (JAX-native MuJoCo) and MuJoCo Warp (preferred for Go2 — supports cylinder collisions and the exact Unitree MJCF)
- **Sim-to-real pipeline** — Train in simulation, deploy on the real Go2 over UDP
- **Composable rewards and observations** — Swap reward terms and observation groups without touching environment internals
- **Fast iteration** — PPO at ~110k steps/sec, off-policy algorithms scale to 100M+ steps

## Benchmarks

### CheetahRun (6-dim actions)

| Algorithm | Eval Score | Training Steps |
|-----------|-----------|----------------|
| PPO       | 826       | 20M            |
| SAC       | 771       | 5M             |
| FastTD3   | 880       | 86M            |

### HumanoidRun (21-dim actions)

| Algorithm | Eval Score | Training Steps |
|-----------|-----------|----------------|
| PPO       | ~10       | 60M            |
| SAC       | 426       | 20M            |
| FastSAC   | 892       | 100M           |

### Go2 Joystick — Warp (12-dim actions, Unitree MJCF)

| Algorithm                | Eval Score | Training Steps |
|--------------------------|-----------|----------------|
| FastSAC (asym. critic)   | 279.2     | 20M            |
| FastSAC (symmetric)      | 276.5     | 18M            |

!!! note "What do these scores mean?"
    Eval scores are average undiscounted episode returns. Higher is better. The Go2 Joystick task rewards tracking velocity commands while maintaining stable locomotion.

## Quick Links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up jax-learning with `uv sync` in under 2 minutes.

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Train CartpoleBalance in 5 minutes, from zero to video.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Concepts**

    ---

    Understand the architecture: environments, algorithms, configs.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Train locomotion, build custom environments, go sim-to-real.

    [:octicons-arrow-right-24: Tutorials](tutorials/train-locomotion.md)

</div>
