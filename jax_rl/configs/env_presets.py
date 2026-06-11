"""Per-environment training presets (MuJoCo Playground reference configs)."""

import dataclasses

from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS


def _resolve_go2_variant(env_name, base_cfg, base_algo, algo_name):
    """Resolve a Go2 Warp env preset from GO2_WARP_VARIANTS.

    Returns None for non-Go2 names and the excluded splitbelt family (caller
    falls through to its preset table / silent default). Raises ValueError for
    unknown ``Go2Warp*`` names — no silent fallback for the Go2 family.

    Override precedence: ``v.train`` (algo-agnostic TrainConfig deltas) first,
    then ``v.algo[algo_name]`` (algo-specific wins). Keys in ``v.algo[algo_name]``
    that are TrainConfig fields apply to the TrainConfig, the rest to the algo
    config — so a variant can carry per-algo train deltas without polluting
    ``train``.
    """
    v = GO2_WARP_VARIANTS.get(env_name)
    if v is None:
        if env_name.startswith("Go2Warp") and not env_name.startswith("Go2WarpSplitbelt"):
            raise ValueError(
                f"unknown Go2 Warp env {env_name!r}; known: {sorted(GO2_WARP_VARIANTS)}"
            )
        return None
    cfg = dataclasses.replace(base_cfg, env_name=env_name, **v.train)
    algo_overrides = dict(v.algo.get(algo_name, {}))
    train_fields = {f.name for f in dataclasses.fields(base_cfg)}
    cfg_overrides = {k: algo_overrides.pop(k) for k in list(algo_overrides)
                     if k in train_fields}
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    algo = dataclasses.replace(base_algo, **algo_overrides) if algo_overrides else base_algo
    return cfg, algo


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
}

# Go2 Warp PPO recipe — same hyperparams the legacy Go2Warp* PPO entries used.
# Go2 PPO presets resolve as this base + the variant's `train` overrides
# (see _resolve_go2_variant / GO2_WARP_VARIANTS).
# PPO hit 132 (entropy collapse); FastSAC preferred (see FAST_SAC_PRESETS).
_GO2_PPO_BASE_CFG = TrainConfig(
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
)

# Factory PegInsert — MJWarp dict-obs (state=25, privileged=72). Asymmetric AC
# auto-engages via _extract_obs. Contact budget caps num_envs ~128 on 16GB.
# v15 phased reward (Phase A above-bore align, Phase B below-bore aligned
# descent or dead-end penalty, +5 terminal success).
#
# Hparams story:
#   v15a (4 ep, ec=1e-2, squash=True): collapsed Return 1530→66 at iter 28,
#     KL spikes e6-e9 from tanh-Jacobian overflow when σ shrinks.
#   v15b (1 ep, ec=2e-2, rew=0.1, squash=True): recovered to Return 1300
#     by iter 139 then locked into squash-saturated regime (PLoss=1.8
#     sustained, KL=4e6 stuck) at iter 140-153.
#   v15c (this): squash=False so log_prob is plain Gaussian (no tanh
#     Jacobian → no overflow). state_dependent_std lets σ adapt per
#     state instead of one scalar collapsing to zero. lr=1e-4 for safer
#     updates. Actions out-of-range handled by env clip_to_bounds.
PRESETS["FactoryPegInsert"] = TrainConfig(
    env_name="FactoryPegInsert",
    total_timesteps=5_000_000,
    num_envs=128,
    episode_length=900,
    gamma=0.99,
    lr=1e-4,
    reward_scaling=0.1,
    reset_mode="per_step",
    handle_truncation=True,
    ppo=PPOConfig(
        num_steps=64,
        num_minibatches=32,
        num_updates_per_batch=1,
        num_epochs=4,
        entropy_coef=1e-2,
        clip_eps=0.2,
        max_grad_norm=0.5,
        anneal_lr=True,
        policy_hidden_dim=(256, 256),
        value_hidden_dim=(256, 256),
        squash=False,
        state_dependent_std=False,
    ),
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

# Splitbelt env (S§5.6) — PPO preset clones joystick + bumps episode_length to 1250
# to match splitbelt env default (25 s @ ctrl_dt=0.02). Per spec note: PPO is the
# secondary calibration smoke (FastSAC primary); both must clear go/no-go gates
# in Task 5.1 to validate algo-agnosticism. PPO entropy-collapse is a documented
# risk (zero-clip on negative-return steps); see spec §11.X caveat.
PRESETS["Go2WarpSplitbelt"] = dataclasses.replace(
    _GO2_PPO_BASE_CFG,
    env_name="Go2WarpSplitbelt",
    episode_length=1250,
    reset_mode="per_step",
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
    # Factory PegInsert — capsule peg + bore-tile ring substrate. Dict obs
    # (state=25, privileged_state=~72) → asymmetric AC auto-engages.
    # Bore-tile contact budget caps num_envs ~256 on 16GB. Phase 1 smoke at 50k.
    "FactoryPegInsert": (
        dataclasses.replace(_SAC_BASE_CFG,
                            env_name="FactoryPegInsert",
                            num_envs=256,
                            episode_length=900,
                            total_timesteps=5_000_000,
                            reset_mode="per_step",
                            handle_truncation=True),
        dataclasses.replace(_SAC_BASE_ALGO,
                            # target_entropy_scale=1.0 (default). v1..v5
                            # collapsed not because of entropy/EMA but
                            # because actuator_mode default was
                            # "position_pd" — OSC never engaged, actions
                            # ignored. Fixed by flipping default to "motor".
                            target_entropy_scale=1.0,
                            obs_normalization=True,
                            grad_updates_per_step=2),
    ),
}


def get_sac_preset(env_name: str) -> tuple[TrainConfig, SACConfig]:
    """Return SAC preset (TrainConfig, SACConfig) for env, or a default."""
    resolved = _resolve_go2_variant(env_name, _SAC_BASE_CFG, _SAC_BASE_ALGO, "sac")
    if resolved is not None:
        return resolved
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
    resolved = _resolve_go2_variant(env_name, _TD3_BASE_CFG, _TD3_BASE_ALGO, "td3")
    if resolved is not None:
        return resolved
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
    resolved = _resolve_go2_variant(env_name, _FAST_TD3_BASE_CFG, _FAST_TD3_BASE_ALGO, "fast_td3")
    if resolved is not None:
        return resolved
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
}
# Go2 Warp (non-splitbelt) presets resolve from GO2_WARP_VARIANTS via
# _resolve_go2_variant — FastSAC preferred over PPO for Go2 (eval 276.5 @ 18M,
# seed 6001, Go2WarpJoystickFlat).

# Splitbelt env (S§5.6) — FastSAC preset clones joystick base + bumps episode_length
# to 1250 to match splitbelt env default. per_step reset mode required for
# DomainRandWrapper auto-reset semantics.
FAST_SAC_PRESETS["Go2WarpSplitbelt"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpSplitbelt",
                        episode_length=1250, reset_mode="per_step"),
    _FAST_SAC_BASE_ALGO,
)
FAST_SAC_PRESETS["Go2WarpSplitbeltDR"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpSplitbeltDR",
                        episode_length=1250, reset_mode="per_step"),
    _FAST_SAC_BASE_ALGO,
)
FAST_SAC_PRESETS["Go2WarpSplitbeltPoseDR"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpSplitbeltPoseDR",
                        episode_length=1250, reset_mode="per_step"),
    _FAST_SAC_BASE_ALGO,
)

# (MuJoCo Warp Push{T,L,Circle,Plus} presets removed 2026-04-20 —
# `push_env.py` was junk, cross-shape work moved to vendored gym-pusht.)



def get_fast_sac_preset(env_name: str) -> tuple[TrainConfig, FastSACConfig]:
    """Return FastSAC preset (TrainConfig, FastSACConfig) for env, or a default."""
    resolved = _resolve_go2_variant(env_name, _FAST_SAC_BASE_CFG, _FAST_SAC_BASE_ALGO, "fast_sac")
    if resolved is not None:
        return resolved
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
}

# Factory PegInsert — Warp contact budget caps num_envs ~128. Scale UTD up
# (16 instead of paper's 8 at 1024 envs) since per-env data rate is 8× lower.
# sigma_target=0.5 (up from 0.15 default) raises target_entropy ≈ -2.9 → +4.4
# so alpha doesn't collapse to 0 and the policy keeps exploring instead of
# locking on a hover attractor (first v15.1 run plateaued at Return 1770 with
# Alpha=0 and Ent=-2.5 ≈ target).
FLASH_SAC_PRESETS["FactoryPegInsert"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG,
                        env_name="FactoryPegInsert",
                        num_envs=128,
                        # Insertion completes by ~125 steps; the remaining
                        # 775 of the original 900 were just "hold the seat."
                        # 450 = insertion + ~300 hold steps, plenty to learn
                        # maintain behavior, half the per-episode compute.
                        episode_length=450,
                        # 2.5M empirically sufficient: v15.6 hit Return 6545
                        # at step ~1M (eval 1024 eps) and plateaued 6500-6650
                        # through step 5M.
                        total_timesteps=2_500_000,
                        gamma=0.99,
                        # 100 eps ≈ every ~57k env steps ≈ 20 ckpts/2M run.
                        eval_every_n_episodes=100,
                        reset_mode="per_step"),
    dataclasses.replace(_FLASH_SAC_BASE_ALGO,
                        grad_updates_per_step=16,
                        # alpha_init=0.1 (10× default): keep entropy bonus
                        # alive through the critical 100-500k warmup. v15.4
                        # had alpha settle ~3× higher than v15.5 at matched
                        # steps and was the only run that broke past the
                        # hover attractor (Return 6150 vs 1700). Alpha gap
                        # came from GPU-nondeterminism rolling alpha lower
                        # during early estimates; higher init gives more
                        # margin before auto-tune crushes it.
                        alpha_init=0.1,
                        # PROBE v8: σ=0.30 → target_entropy=+1.29 @ D=6, keeps
                        # entropy alive for 6-DOF exploration. Pair with DR
                        # disabled (factory_peg_insert.py PROBE v8 block).
                        sigma_target=0.30),
)

# Factory GearMesh — Phase 1 scaffold (smoke-train hyperparams; tune after
# first 50k-500k results land).
FLASH_SAC_PRESETS["FactoryGearMesh"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG,
                        env_name="FactoryGearMesh",
                        num_envs=64,
                        episode_length=450,
                        total_timesteps=2_000_000,
                        gamma=0.99,
                        # mjx-Warp leaks ~1.7 MB / eval (JAX-side, unreclaimable).
                        # Budget = ~14 evals before OOM at step ~798k. Set
                        # eval_every high enough to stay under that across the
                        # full run. 2M steps × 64 envs / 450 ep_len ≈ 4500 eps
                        # total; eval_every=1500 → 3 evals over 2M, safely under
                        # the OOM cliff. See journal 2026-06-04-mjxwarp-leak.md.
                        eval_every_n_episodes=1500,
                        reset_mode="per_step"),
    dataclasses.replace(_FLASH_SAC_BASE_ALGO,
                        grad_updates_per_step=16,
                        alpha_init=0.1,
                        sigma_target=0.30),
)


def get_flash_sac_preset(env_name: str) -> tuple[TrainConfig, FlashSACConfig]:
    """Return FlashSAC preset (TrainConfig, FlashSACConfig) for env, or a default."""
    resolved = _resolve_go2_variant(env_name, _FLASH_SAC_BASE_CFG, _FLASH_SAC_BASE_ALGO, "flash_sac")
    if resolved is not None:
        return resolved
    if env_name in FLASH_SAC_PRESETS:
        return FLASH_SAC_PRESETS[env_name]
    return dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name=env_name), _FLASH_SAC_BASE_ALGO


def get_preset(env_name: str) -> TrainConfig:
    """Return preset config for env, or a default with env_name set.

    NOTE: returns a BARE TrainConfig (no algo tuple) — train_ppo.py callers
    depend on this signature. Go2 names resolve from GO2_WARP_VARIANTS using
    only the cfg half (PPO hparams live in _GO2_PPO_BASE_CFG.ppo).
    """
    resolved = _resolve_go2_variant(env_name, _GO2_PPO_BASE_CFG, None, "ppo")
    if resolved is not None:
        return resolved[0]
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
    # PushT (gym backend): action_dim=2 (xy pusher target), episode_length=300
    # (gym TimeLimit) × action_repeat=2 = 600 control steps. PushT control is
    # ~50Hz pymunk, so 600 steps ≈ 12 sec real time per episode. Discount
    # auto-recomputes via compute_discount(300, denom=5) ≈ 0.983.
    "PushT": make_tdmpc2_config(action_dim=2, episode_length=300, task_name="PushT"),
}


def get_tdmpc2_preset(env_name: str) -> TDMPC2Config:
    """Return TDMPC2Config for env. Raises KeyError if unknown
    (action_dim and episode_length are required env-specific values, so no safe default)."""
    if env_name not in TDMPC2_PRESETS:
        raise KeyError(
            f"No TDMPC2 preset for env '{env_name}'. Add one to TDMPC2_PRESETS in env_presets.py."
        )
    return TDMPC2_PRESETS[env_name]
