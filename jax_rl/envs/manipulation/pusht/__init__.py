"""Vendored gym-pusht (Chi et al. 2023 DP paper env).

Static benchmark — old and not maintained upstream. Vendored so we can:
- Pin pymunk version (upstream broken on pymunk 7+)
- Add modular reward_mode without forking
- Bundle expert demos for BC / offline RL comparisons

Upstream: https://github.com/huggingface/gym-pusht (Apache 2.0)
LICENSE file retained.

Usage:
    import gymnasium as gym
    from jax_rl.envs.manipulation.pusht import PushTEnv
    env = PushTEnv(reward_mode="shaped")
    # or register:
    gym.register("jax_rl/PushT-v0", PushTEnv)
"""
from jax_rl.envs.manipulation.pusht.pusht import PushTEnv

__all__ = ["PushTEnv"]
