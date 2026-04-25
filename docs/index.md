# jax-learning

**A modular, JAX-native reinforcement learning framework for robot learning research.**

jax-learning is a learning vehicle and lab tool — every algorithm is implemented from fundamentals with clear mappings to the papers. It provides GPU-accelerated simulation environments via MuJoCo Playground, 6 RL algorithms, composable reward/observation specs, and a sim-to-real deployment pipeline (training in simulation, then running the policy on real hardware). Current focus: [Unitree Go2](https://www.unitree.com/go2/) quadruped locomotion and planar-pushing manipulation (see [PushT](api/pusht.md)). If a term below is unfamiliar, the [Glossary](glossary.md) has one-liner definitions.

<div class="video-grid" markdown>
<div markdown>
<video autoplay loop muted playsinline preload="metadata">
  <source src="assets/videos/go2_joystick_walk.mp4" type="video/mp4">
</video>
<p class="caption" markdown="1">Go2 locomotion — FastSAC</p>
</div>
<div markdown>
<video controls loop muted playsinline preload="metadata">
  <source src="assets/videos/go2_bongo_handstand.mp4" type="video/mp4">
</video>
<p class="caption" markdown="1">Bongo board handstand — PPO</p>
</div>
</div>

<small>*All eval scores throughout the docs are single-seed results. Treat as approximate — see [Lessons Learned](reference/lessons-learned.md) for seed variance notes.*</small>

## Key Features

- **7 RL algorithms** — PPO, PPOContraction (Lipschitz contraction-metric regularizer), SAC, TD3, FastSAC, FastTD3, and FlashSAC (SAC variant with learned feature extractors; see [algorithms reference](api/algos.md) for architectural details)
- **GPU-accelerated environments** — MuJoCo Warp backend (primary — supports cylinder collisions and the exact Unitree MJCF). MJX (JAX-native) available for DM Control benchmarks.
- **Manipulation benchmark** — Vendored [PushT](api/pusht.md) planar-pushing env (RL-from-scratch recipe + expert demos, shape-agnostic obs for cross-shape transfer)
- **Sim-to-real pipeline** — Train in simulation, deploy on the real Go2 over UDP
- **Composable rewards and observations** — Swap reward terms and observation groups without touching environment internals
- **Fast iteration** — PPO at ~110k steps/sec, off-policy algorithms scale to 100M+ steps

## Quick Links

- :material-download: [**Installation**](getting-started/installation.md) — Set up jax-learning with `uv sync`
- :material-rocket-launch: [**Quickstart**](getting-started/quickstart.md) — Train CartpoleBalance and record a video
- :material-book-open-variant: [**Concepts**](getting-started/concepts.md) — Understand the architecture: environments, algorithms, configs
- :material-school: [**Tutorials**](tutorials/train-locomotion.md) — Train locomotion, build custom environments, go sim-to-real
- :material-book-alphabet: [**Glossary**](glossary.md) — New to RL or JAX? Start here
