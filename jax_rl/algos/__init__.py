"""Algorithm implementations."""

from jax_rl.algos.ppo import PPO
from jax_rl.algos.sac import SAC
from jax_rl.algos.td3 import TD3
from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.algos.fast_sac import FastSAC
from jax_rl.algos.fast_dsac import FastDSAC

__all__ = ["PPO", "SAC", "TD3", "FastTD3", "FastSAC", "FastDSAC"]
