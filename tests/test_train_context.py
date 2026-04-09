"""Tests for TrainContext dataclass."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.checkpointing import CheckpointManager
from jax_rl.training.train_context import TrainContext


def _make_ctx(**overrides):
    defaults = dict(
        cfg=TrainConfig(),
        algo_cfg=object(),
        algo_name="sac",
        ckpt_dir="/tmp/test_ckpt",
        obs_dim=10,
        action_dim=4,
        metrics_log=[],
        ckpt_mgr=CheckpointManager("/tmp/test_ckpt"),
    )
    defaults.update(overrides)
    return TrainContext(**defaults)


def test_train_context_creation():
    """All fields are set correctly on construction."""
    cfg = TrainConfig(env_name="CartpoleBalance", num_envs=32)
    algo_cfg = object()
    ckpt_mgr = CheckpointManager("/tmp/ckpt")
    metrics = [{"total_steps": 1000}]

    ctx = TrainContext(
        cfg=cfg,
        algo_cfg=algo_cfg,
        algo_name="td3",
        ckpt_dir="/tmp/ckpt",
        obs_dim=12,
        action_dim=6,
        metrics_log=metrics,
        ckpt_mgr=ckpt_mgr,
        resume="/tmp/prev_ckpt",
    )

    assert ctx.cfg is cfg
    assert ctx.algo_cfg is algo_cfg
    assert ctx.algo_name == "td3"
    assert ctx.ckpt_dir == "/tmp/ckpt"
    assert ctx.obs_dim == 12
    assert ctx.action_dim == 6
    assert ctx.metrics_log is metrics
    assert ctx.ckpt_mgr is ckpt_mgr
    assert ctx.resume == "/tmp/prev_ckpt"


def test_train_context_defaults():
    """resume defaults to None when not provided."""
    ctx = _make_ctx()
    assert ctx.resume is None
