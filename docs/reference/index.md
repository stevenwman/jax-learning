# Reference

In-depth reference material for the training infrastructure, CLI, and hard-won engineering lessons. Useful once you have a working setup and want to understand or tune the internals.

- [**CLI Flags**](cli-flags.md) — All command-line arguments across the training scripts, auto-generated from argparse
- [**Environment Presets**](env-presets.md) — Per-environment hyperparameter presets, auto-generated from `env_presets.py`
- [**Architecture**](architecture.md) — Three-layer system design: environments, algorithms, training loops
- [**Training Loop**](training-loop.md) — Annotated walkthrough of the off-policy and PPO training loops
- [**Lessons Learned**](lessons-learned.md) — Non-obvious findings on JAX performance, PPO scaling, and sim-to-real gaps
