"""Configuration dataclasses for algorithms and networks."""

from jax_rl.configs.networks_config import (
    EncoderConfig,
    PolicyHeadConfig,
)
from jax_rl.configs.contraction_config import ContractionConfig
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import (
    get_preset, get_sac_preset, get_td3_preset,
    get_fast_td3_preset, get_fast_sac_preset,
)

__all__ = [
    "EncoderConfig",
    "PolicyHeadConfig",
    "ContractionConfig",
    "PPOConfig",
    "SACConfig",
    "TD3Config",
    "FastTD3Config",
    "FastSACConfig",
    "TrainConfig",
    "get_preset",
    "get_sac_preset",
    "get_td3_preset",
    "get_fast_td3_preset",
    "get_fast_sac_preset",
]
