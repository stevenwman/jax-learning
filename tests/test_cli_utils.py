"""Tests for CLI override utility."""
import argparse
import pytest
from jax_rl.training.cli_utils import apply_cli_overrides
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.td3_config import TD3Config


def _base_args(**overrides):
    """Build a Namespace with all CLI fields defaulted to None/False."""
    defaults = dict(
        num_envs=None,
        total_timesteps=None,
        lr=None,
        reward_scaling=None,
        episode_length=None,
        n_frame_stack=None,
        action_delay_ms=None,
        reset_mode=None,
        batch_size=None,
        grad_updates_per_step=None,
        buffer_size=None,
        target_entropy_scale=None,
        exploration_noise=None,
        obs_norm=False,
        frame_stack=None,
        eval_every=None,
        action_delay_range_ms=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ── None args are ignored ────────────────────────────────────────────────


def test_none_args_leave_cfg_unchanged():
    args = _base_args()
    cfg = TrainConfig()
    algo_cfg = FastSACConfig()
    new_cfg, new_algo = apply_cli_overrides(args, cfg, algo_cfg)
    assert new_cfg == cfg
    assert new_algo == algo_cfg


# ── Direct cfg field overrides ───────────────────────────────────────────


def test_num_envs_override():
    args = _base_args(num_envs=512)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.num_envs == 512


def test_total_timesteps_override():
    args = _base_args(total_timesteps=2_000_000)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.total_timesteps == 2_000_000


def test_lr_override():
    args = _base_args(lr=1e-4)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.lr == 1e-4


def test_reward_scaling_override():
    args = _base_args(reward_scaling=0.5)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.reward_scaling == 0.5


def test_episode_length_override():
    args = _base_args(episode_length=500)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.episode_length == 500


def test_action_delay_ms_override():
    args = _base_args(action_delay_ms=120)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.action_delay_ms == 120


def test_reset_mode_override():
    args = _base_args(reset_mode="per_step")
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.reset_mode == "per_step"


# ── Direct algo field overrides ──────────────────────────────────────────


def test_batch_size_override():
    args = _base_args(batch_size=4096)
    _, new_algo = apply_cli_overrides(args, TrainConfig(), FastSACConfig())
    assert new_algo.batch_size == 4096


def test_grad_updates_per_step_override():
    args = _base_args(grad_updates_per_step=4)
    _, new_algo = apply_cli_overrides(args, TrainConfig(), FastSACConfig())
    assert new_algo.grad_updates_per_step == 4


def test_buffer_size_override():
    args = _base_args(buffer_size=500_000)
    _, new_algo = apply_cli_overrides(args, TrainConfig(), FastSACConfig())
    assert new_algo.buffer_size == 500_000


def test_target_entropy_scale_override():
    args = _base_args(target_entropy_scale=0.5)
    _, new_algo = apply_cli_overrides(args, TrainConfig(), FastSACConfig())
    assert new_algo.target_entropy_scale == 0.5


# ── Renamed algo fields ──────────────────────────────────────────────────


def test_exploration_noise_maps_to_exploration_noise_std():
    args = _base_args(exploration_noise=0.2)
    _, new_algo = apply_cli_overrides(args, TrainConfig(), TD3Config())
    assert new_algo.exploration_noise_std == 0.2


# ── Bool flags ───────────────────────────────────────────────────────────


def test_obs_norm_sets_algo_flag():
    args = _base_args(obs_norm=True)
    algo_cfg = FastSACConfig()
    assert algo_cfg.obs_normalization is False
    _, new_algo = apply_cli_overrides(args, TrainConfig(), algo_cfg)
    assert new_algo.obs_normalization is True


def test_obs_norm_false_leaves_algo_unchanged():
    args = _base_args(obs_norm=False)
    algo_cfg = FastSACConfig()
    _, new_algo = apply_cli_overrides(args, TrainConfig(), algo_cfg)
    assert new_algo.obs_normalization is False


# ── Special case: frame_stack -> n_frame_stack ───────────────────────────


def test_frame_stack_maps_to_n_frame_stack():
    args = _base_args(frame_stack=3)
    cfg = TrainConfig()
    assert cfg.n_frame_stack == 1
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.n_frame_stack == 3


def test_frame_stack_none_does_not_override():
    args = _base_args(frame_stack=None)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.n_frame_stack == cfg.n_frame_stack


# ── Special case: eval_every -> eval_every_n_episodes ────────────────────


def test_eval_every_maps_to_eval_every_n_episodes():
    args = _base_args(eval_every=5000)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.eval_every_n_episodes == 5000


# ── Special case: action_delay_range_ms -> tuple ─────────────────────────


def test_action_delay_range_ms_converts_to_tuple():
    args = _base_args(action_delay_range_ms=[40, 120])
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.action_delay_range_ms == (40, 120)
    assert isinstance(new_cfg.action_delay_range_ms, tuple)


def test_action_delay_range_ms_none_leaves_cfg_unchanged():
    args = _base_args(action_delay_range_ms=None)
    cfg = TrainConfig()
    new_cfg, _ = apply_cli_overrides(args, cfg, FastSACConfig())
    assert new_cfg.action_delay_range_ms is None


# ── Multiple overrides applied together ──────────────────────────────────


def test_multiple_overrides_applied():
    args = _base_args(
        num_envs=256,
        total_timesteps=5_000_000,
        lr=3e-4,
        frame_stack=3,
        eval_every=1000,
        batch_size=2048,
        obs_norm=True,
    )
    cfg = TrainConfig()
    algo_cfg = FastSACConfig()
    new_cfg, new_algo = apply_cli_overrides(args, cfg, algo_cfg)

    assert new_cfg.num_envs == 256
    assert new_cfg.total_timesteps == 5_000_000
    assert new_cfg.lr == 3e-4
    assert new_cfg.n_frame_stack == 3
    assert new_cfg.eval_every_n_episodes == 1000
    assert new_algo.batch_size == 2048
    assert new_algo.obs_normalization is True


# ── Returns are new objects, originals unchanged ─────────────────────────


def test_original_configs_not_mutated():
    args = _base_args(num_envs=1024, batch_size=256)
    cfg = TrainConfig()
    algo_cfg = FastSACConfig()
    original_num_envs = cfg.num_envs
    original_batch_size = algo_cfg.batch_size

    apply_cli_overrides(args, cfg, algo_cfg)

    assert cfg.num_envs == original_num_envs
    assert algo_cfg.batch_size == original_batch_size
