# jax-learning

**A JAX-based reinforcement learning framework for training robot policies in simulation and deploying them on real hardware.**

jax-learning is built around the [Unitree Go2](https://www.unitree.com/go2/) quadruped robot. It provides GPU-accelerated environments, multiple RL algorithms, and a sim-to-real deployment pipeline — all in pure JAX.

<div class="video-grid" markdown>
<div>
<video autoplay loop muted playsinline>
  <source src="assets/videos/go2_joystick_walk.mp4" type="video/mp4">
</video>
<p class="caption">Go2 locomotion — FastSAC, eval 276.5</p>
</div>
<div>
<video autoplay loop muted playsinline>
  <source src="assets/videos/go2_bongo_handstand.mp4" type="video/mp4">
</video>
<p class="caption">Bongo board handstand — PPO</p>
</div>
</div>

## Key Features

- **6 RL algorithms** — PPO, SAC, TD3, FastSAC, FastTD3, and FlashSAC (inverted residual blocks + BatchNorm + adaptive reward scaling)
- **GPU-accelerated environments** — MuJoCo Warp backend (primary — supports cylinder collisions and the exact Unitree MJCF). MJX (JAX-native) available for DM Control benchmarks.
- **Sim-to-real pipeline** — Train in simulation, deploy on the real Go2 over UDP
- **Composable rewards and observations** — Swap reward terms and observation groups without touching environment internals
- **Fast iteration** — PPO at ~110k steps/sec, off-policy algorithms scale to 100M+ steps

## Quick Links

- :material-download: [**Installation**](getting-started/installation.md) — Set up jax-learning with `uv sync`
- :material-rocket-launch: [**Quickstart**](getting-started/quickstart.md) — Train CartpoleBalance and record a video
- :material-book-open-variant: [**Concepts**](getting-started/concepts.md) — Understand the architecture: environments, algorithms, configs
- :material-school: [**Tutorials**](tutorials/train-locomotion.md) — Train locomotion, build custom environments, go sim-to-real
- :material-book-alphabet: [**Glossary**](glossary.md) — New to RL or JAX? Start here
