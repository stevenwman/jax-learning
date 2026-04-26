"""Per-environment training presets (MuJoCo Playground reference configs)."""

import dataclasses

from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.configs.train_config import TrainConfig

PRESETS: dict[str, TrainConfig] = {
    "CartpoleBalance": TrainConfig(
        env_name="CartpoleBalance",
        total_timesteps=1_000_000,
        num_envs=64,
        gamma=0.99,
        lr=3e-4,
        reward_scaling=1.0,
        ppo=PPOConfig(num_steps=64, max_grad_norm=0.5, entropy_coef=0.01, num_epochs=4),
    ),
    # NOTE: CartpoleSwingup / -Sparse presets below are *smoke-test* configs —
    # HPs picked for fast cheap runs during ContractionPPO A/B (2026-04-24), NOT
    # tuned for peak return. ~30s per 3M-step run at 256 envs. Use for algo
    # smoke tests or quick sanity checks, not as published benchmarks.
    "CartpoleSwingup": TrainConfig(
        env_name="CartpoleSwingup",
        total_timesteps=3_000_000,
        num_envs=256,
        gamma=0.99,
        lr=3e-4,
        reward_scaling=1.0,
        episode_length=1000,
        eval_every_n_episodes=500,
        ppo=PPOConfig(
            num_steps=64, num_minibatches=32, num_updates_per_batch=1,
            num_epochs=4, entropy_coef=0.01, clip_eps=0.2,
            max_grad_norm=0.5, anneal_lr=True,
            policy_hidden_dim=(64, 64), value_hidden_dim=(64, 64),
            activation="tanh", squash=False,
        ),
    ),
    # Sparse variant: reward=1 only when pole near-upright. Much harder, 2/3
    # seeds fail at 5M. Kept for future sparse-reward algo probes.
    "CartpoleSwingupSparse": TrainConfig(
        env_name="CartpoleSwingupSparse",
        total_timesteps=5_000_000,
        num_envs=256,
        gamma=0.99,
        lr=3e-4,
        reward_scaling=1.0,
        episode_length=1000,
        ppo=PPOConfig(
            num_steps=64, num_minibatches=32, num_updates_per_batch=1,
            num_epochs=4, entropy_coef=0.01, clip_eps=0.2,
            max_grad_norm=0.5, anneal_lr=True,
            policy_hidden_dim=(64, 64), value_hidden_dim=(64, 64),
            activation="tanh", squash=False,
        ),
    ),
    "CheetahRun": TrainConfig(
        env_name="CheetahRun",
        total_timesteps=20_000_000,
        num_envs=2048,
        gamma=0.995,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(num_steps=30, entropy_coef=1e-2, num_epochs=16),
    ),
    "WalkerWalk": TrainConfig(
        env_name="WalkerWalk",
        total_timesteps=60_000_000,
        num_envs=2048,
        gamma=0.995,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(num_steps=30, entropy_coef=1e-2, num_epochs=16),
    ),
    "HumanoidRun": TrainConfig(
        env_name="HumanoidRun",
        total_timesteps=60_000_000,
        num_envs=2048,
        gamma=0.995,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(num_steps=480, anneal_lr=False, entropy_coef=1e-2, num_epochs=16,
                      policy_hidden_dim=(128, 128, 128, 128), state_dependent_std=True),
    ),
    # Go2 Warp locomotion — same PPO recipe, unitree MJCF via Warp backend.
    # PPO hit 132 (entropy collapse); FastSAC preferred (see FAST_SAC_PRESETS).
    "Go2WarpJoystickFlat": TrainConfig(
        env_name="Go2WarpJoystickFlat",
        total_timesteps=100_000_000,
        num_envs=4096,
        gamma=0.97,
        lr=3e-4,
        reward_scaling=1.0,
        episode_length=1000,
        ppo=PPOConfig(
            num_steps=20,
            num_minibatches=32,
            num_updates_per_batch=4,
            num_epochs=4,
            entropy_coef=1e-2,
            max_grad_norm=1.0,
            policy_hidden_dim=(512, 256, 128),
            value_hidden_dim=(512, 256, 128),
        ),
    ),
}
PRESETS["Go2WarpJoystickFlatTorqueSpeed"] = dataclasses.replace(
    PRESETS["Go2WarpJoystickFlat"], env_name="Go2WarpJoystickFlatTorqueSpeed"
)

# Curriculum variants — same hyperparams as Flat + per_step reset mode
# (required: curriculum logic lives in TerrainCurriculumDRWrapper which
# env_setup only applies when reset_mode == "per_step").
PRESETS["Go2WarpJoystickCurriculum"] = dataclasses.replace(
    PRESETS["Go2WarpJoystickFlat"],
    env_name="Go2WarpJoystickCurriculum",
    reset_mode="per_step",
)
PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = dataclasses.replace(
    PRESETS["Go2WarpJoystickFlat"],
    env_name="Go2WarpJoystickCurriculumTorqueSpeed",
    reset_mode="per_step",
)

# Bongo handstand — matches PPO4 config from 2026-04-03 (eval 46.9 @ 80M, FS=3).
# Source: checkpoints/20260403_094124_ppo_go2bongohandstand_seed0/meta.json.
PRESETS["Go2BongoHandstand"] = TrainConfig(
    env_name="Go2BongoHandstand",
    total_timesteps=100_000_000,
    num_envs=256,
    gamma=0.99,
    lr=3e-4,
    reward_scaling=1.0,
    episode_length=250,
    n_frame_stack=3,
    ppo=PPOConfig(
        clip_eps=0.3,
        entropy_coef=0.01,
        gae_lambda=0.95,
        num_epochs=4,
        num_minibatches=32,
        num_steps=64,
        num_updates_per_batch=1,
        policy_hidden_dim=(32, 32, 32, 32),
        value_hidden_dim=(256, 256, 256, 256, 256),
        activation="swish",
        squash=True,
        anneal_lr=True,
        max_grad_norm=None,
    ),
)
# Contraction variant — same hyperparams, observe_contraction=True via env registry.
PRESETS["Go2BongoHandstandContraction"] = dataclasses.replace(
    PRESETS["Go2BongoHandstand"],
    env_name="Go2BongoHandstandContraction",
)


# SAC presets — matching MuJoCo Playground dm_control_suite_params.brax_sac_config()
# Reference HPs: lr=1e-3, batch_size=512, grad_updates_per_step=8, q_layer_norm=True
# num_envs=128, min_buffer=8192, max_buffer=4M, num_timesteps=5M
_SAC_BASE_CFG = TrainConfig(
    total_timesteps=5_000_000,
    num_envs=128,
    episode_length=1000,
    lr=1e-3,
    reward_scaling=1.0,
    gamma=0.99,
    handle_truncation=True,
)

_SAC_BASE_ALGO = SACConfig(
    tau=0.005,
    target_entropy_scale=0.5,
    alpha_lr=1e-3,
    buffer_size=4_194_304,
    min_buffer_size=8_192,
    batch_size=512,
    grad_updates_per_step=8,
    hidden_dim=(256, 256),
    activation="relu",
    q_layer_norm=True,
)

SAC_PRESETS: dict[str, tuple[TrainConfig, SACConfig]] = {
    "WalkerWalk": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="WalkerWalk"),
        _SAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="HumanoidRun"),
        _SAC_BASE_ALGO,
    ),
    "CheetahRun": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="CheetahRun"),
        _SAC_BASE_ALGO,
    ),
    "PandaPickCube": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="PandaPickCube",
                            episode_length=150, total_timesteps=10_000_000),
        _SAC_BASE_ALGO,
    ),
}


def get_sac_preset(env_name: str) -> tuple[TrainConfig, SACConfig]:
    """Return SAC preset (TrainConfig, SACConfig) for env, or a default."""
    if env_name in SAC_PRESETS:
        return SAC_PRESETS[env_name]
    return dataclasses.replace(_SAC_BASE_CFG, env_name=env_name), _SAC_BASE_ALGO


# TD3 presets — vanilla TD3 with 1:1 gradient ratio
_TD3_BASE_CFG = TrainConfig(
    total_timesteps=5_000_000,
    num_envs=128,
    episode_length=1000,
    lr=3e-4,
    reward_scaling=1.0,
    gamma=0.99,
    handle_truncation=True,
)

_TD3_BASE_ALGO = TD3Config(
    grad_updates_per_step=4,  # 128 envs need higher replay ratio than vanilla TD3's 1:1
    batch_size=256,
)

TD3_PRESETS: dict[str, tuple[TrainConfig, TD3Config]] = {
    "CheetahRun": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="CheetahRun"),
        _TD3_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="WalkerWalk"),
        _TD3_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="HumanoidRun"),
        dataclasses.replace(_TD3_BASE_ALGO, q_layer_norm=True),  # stability for high-dim
    ),
}


def get_td3_preset(env_name: str) -> tuple[TrainConfig, TD3Config]:
    """Return TD3 preset (TrainConfig, TD3Config) for env, or a default."""
    if env_name in TD3_PRESETS:
        return TD3_PRESETS[env_name]
    return dataclasses.replace(_TD3_BASE_CFG, env_name=env_name), _TD3_BASE_ALGO


# FastTD3 presets — Seo et al. 2025 (arXiv:2512.01996)
# Paper recipe: gamma=0.97, mixed noise U[0.01, 0.05], AdamW β2=0.95, wd=0.001
_FAST_TD3_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    reward_scaling=1.0,
    gamma=0.97,             # paper: 0.97 for locomotion
    num_eval_episodes=5,
    handle_truncation=True,
)

_FAST_TD3_BASE_ALGO = FastTD3Config(
    noise_min=0.01,         # paper: mixed noise σ ~ U[0.01, 0.05]
    noise_max=0.05,
)

# Paper uses v_min/v_max = [-20, 20] universally (now the default in FastTD3Config)
FAST_TD3_PRESETS: dict[str, tuple[TrainConfig, FastTD3Config]] = {
    "CheetahRun": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="CheetahRun"),
        _FAST_TD3_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="WalkerWalk"),
        _FAST_TD3_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="HumanoidRun"),
        _FAST_TD3_BASE_ALGO,
    ),
}


def get_fast_td3_preset(env_name: str) -> tuple[TrainConfig, FastTD3Config]:
    """Return FastTD3 preset (TrainConfig, FastTD3Config) for env, or a default."""
    if env_name in FAST_TD3_PRESETS:
        return FAST_TD3_PRESETS[env_name]
    return dataclasses.replace(_FAST_TD3_BASE_CFG, env_name=env_name), _FAST_TD3_BASE_ALGO


# FastSAC presets — Seo et al. 2025 (arXiv:2512.01996)
# Key differences from vanilla SAC: alpha_init=0.001, max_std=1.0, target_entropy=0,
# gamma=0.97 (locomotion), adam β2=0.95, weight_decay=0.001, Q averaging, C51 critic
_FAST_SAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,               # paper: 0.0003
    reward_scaling=1.0,
    gamma=0.97,             # paper: 0.97 for locomotion (NOT 0.99)
    num_eval_episodes=5,
    handle_truncation=True,
)

_FAST_SAC_BASE_ALGO = FastSACConfig(
    tau=0.125,                     # paper: 0.125 (fast target update for high UTD ratio)
    target_entropy_scale=0.0,      # target_entropy=0 (prevents alpha collapse at scale)
    alpha_lr=3e-4,
    alpha_init=0.001,              # start near-zero, not 1.0
    max_std=1.0,                   # cap pre-tanh std (log_std_max=0.0 → std=1.0)
    buffer_size=4_194_304,
    min_buffer_size=8_192,
    batch_size=8_192,
    grad_updates_per_step=8,           # paper: 8 (not 12)
    hidden_dim=(512, 256, 128),        # paper: tapered actor (actor_hidden_dim=512 → 512/256/128)
    critic_hidden_dim=(768, 384, 192), # paper: wider tapered critic (critic_hidden_dim=768)
    activation="swish",                # paper: SiLU (= swish)
    q_layer_norm=True,
    policy_delay=4,                    # paper: actor updates every 4th critic update
)

FAST_SAC_PRESETS: dict[str, tuple[TrainConfig, FastSACConfig]] = {
    "CheetahRun": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="CheetahRun"),
        _FAST_SAC_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="WalkerWalk"),
        _FAST_SAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="HumanoidRun"),
        _FAST_SAC_BASE_ALGO,
    ),
    # Go2 Warp — eval 276.5 @ 18M steps (seed 6001). Preferred over PPO for Go2.
    "Go2WarpJoystickFlat": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickFlat"),
        _FAST_SAC_BASE_ALGO,
    ),
}
FAST_SAC_PRESETS["Go2WarpJoystickFlatTorqueSpeed"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickFlatTorqueSpeed"),
    _FAST_SAC_BASE_ALGO,
)
FAST_SAC_PRESETS["Go2WarpJoystickCurriculum"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculum", reset_mode="per_step"),
    _FAST_SAC_BASE_ALGO,
)
FAST_SAC_PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculumTorqueSpeed", reset_mode="per_step"),
    _FAST_SAC_BASE_ALGO,
)

# (MuJoCo Warp Push{T,L,Circle,Plus} presets removed 2026-04-20 —
# `push_env.py` was junk, cross-shape work moved to vendored gym-pusht.)



def get_fast_sac_preset(env_name: str) -> tuple[TrainConfig, FastSACConfig]:
    """Return FastSAC preset (TrainConfig, FastSACConfig) for env, or a default."""
    if env_name in FAST_SAC_PRESETS:
        return FAST_SAC_PRESETS[env_name]
    return dataclasses.replace(_FAST_SAC_BASE_CFG, env_name=env_name), _FAST_SAC_BASE_ALGO



# FlashSAC presets — Kim et al. 2026 (arXiv:2604.04539)
# Key differences from FastSAC: inverted residual blocks, BatchNorm, weight norm,
# adaptive reward scaling, unified entropy target (sigma=0.15), Zeta noise repetition.
# Paper defaults: num_blocks=2, actor_hidden=128, critic_hidden=256, batch_size=2048,
# tau=0.01, UTD=1 (single-env). For parallel envs, scale UTD to compensate.
_FLASH_SAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    reward_scaling=1.0,         # raw rewards; FlashSAC normalizes adaptively
    gamma=0.97,                  # paper: 0.97 for locomotion
    num_eval_episodes=5,
    handle_truncation=True,
)

_FLASH_SAC_BASE_ALGO = FlashSACConfig(
    # Paper defaults (unchanged from FlashSACConfig defaults except UTD)
    grad_updates_per_step=8,     # compensate for 1024 parallel envs (paper uses UTD=1 @ 1 env)
)

FLASH_SAC_PRESETS: dict[str, tuple[TrainConfig, FlashSACConfig]] = {
    "CheetahRun": (
        dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="CheetahRun", gamma=0.99),
        _FLASH_SAC_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="WalkerWalk", gamma=0.99),
        _FLASH_SAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="HumanoidRun", gamma=0.99),
        _FLASH_SAC_BASE_ALGO,
    ),
    "Go2WarpJoystickFlat": (
        dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickFlat"),
        _FLASH_SAC_BASE_ALGO,
    ),
}
FLASH_SAC_PRESETS["Go2WarpJoystickFlatTorqueSpeed"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickFlatTorqueSpeed"),
    _FLASH_SAC_BASE_ALGO,
)
FLASH_SAC_PRESETS["Go2WarpJoystickCurriculum"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculum", reset_mode="per_step"),
    _FLASH_SAC_BASE_ALGO,
)
FLASH_SAC_PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculumTorqueSpeed", reset_mode="per_step"),
    _FLASH_SAC_BASE_ALGO,
)


def get_flash_sac_preset(env_name: str) -> tuple[TrainConfig, FlashSACConfig]:
    """Return FlashSAC preset (TrainConfig, FlashSACConfig) for env, or a default."""
    if env_name in FLASH_SAC_PRESETS:
        return FLASH_SAC_PRESETS[env_name]
    return dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name=env_name), _FLASH_SAC_BASE_ALGO


def get_preset(env_name: str) -> TrainConfig:
    """Return preset config for env, or a default with env_name set."""
    if env_name in PRESETS:
        return PRESETS[env_name]
    return TrainConfig(env_name=env_name, total_timesteps=3_000_000)


# -----------------------------------------------------------------------------
# TD-MPC2 presets (DMC single-task phase P1).
# -----------------------------------------------------------------------------
# Unlike other algos, TDMPC2Config bundles training-loop fields (total_steps,
# num_envs, eval_every, buffer_size) alongside algorithm hparams, so presets
# return TDMPC2Config directly rather than a (TrainConfig, AlgoConfig) tuple.

from jax_rl.configs.tdmpc2_config import TDMPC2Config, make_tdmpc2_config

TDMPC2_PRESETS: dict[str, TDMPC2Config] = {
    # episode_length=500 (wrapper steps) × action_repeat=2 = 1000 control steps,
    # matches source dmcontrol.py (Timeout(500) + hardcoded range(2)). Discount
    # auto-recomputes to 0.99 via compute_discount(500, denom=5).
    "CheetahRun": make_tdmpc2_config(action_dim=6, episode_length=500, task_name="CheetahRun"),
    "HumanoidRun": make_tdmpc2_config(action_dim=21, episode_length=500, task_name="HumanoidRun"),
    "HopperHop": make_tdmpc2_config(action_dim=4, episode_length=500, task_name="HopperHop"),
    "AcrobotSwingup": make_tdmpc2_config(action_dim=1, episode_length=500, task_name="AcrobotSwingup"),
    "CartpoleSwingup": make_tdmpc2_config(action_dim=1, episode_length=500, task_name="CartpoleSwingup"),
}


def get_tdmpc2_preset(env_name: str) -> TDMPC2Config:
    """Return TDMPC2Config for env. Raises KeyError if unknown
    (action_dim and episode_length are required env-specific values, so no safe default)."""
    if env_name not in TDMPC2_PRESETS:
        raise KeyError(
            f"No TDMPC2 preset for env '{env_name}'. Add one to TDMPC2_PRESETS in env_presets.py."
        )
    return TDMPC2_PRESETS[env_name]
