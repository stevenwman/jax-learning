"""Tests for checkpoint save/load correctness.

Verifies:
- Directory structure (meta.json, metrics.csv, actor_params.npy, orbax/ subfolder)
- meta.json contains full train_config
- metrics.csv has correct rows and columns
- actor_params.npy loads and params match
- norm stats in actor_params.npy are correct
- Orbax restore (for training resume) works and params match
"""

import csv
import json
import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import _save_checkpoint
from jax_rl.algos.ppo import PPO
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.normalization import NormalizationState, init as norm_init


OBS_DIM = 5
ACTION_DIM = 2
HIDDEN = (8, 8)


def make_ppo_and_state():
    cfg = PPOConfig(
        encoder=EncoderConfig(obs_dim=OBS_DIM, hidden_dim=HIDDEN),
        critic_encoder=EncoderConfig(obs_dim=OBS_DIM, hidden_dim=HIDDEN),
        policy_head=PolicyHeadConfig(action_dim=ACTION_DIM),
        num_envs=1,
    )
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3))
    ppo = PPO(cfg, OBS_DIM, ACTION_DIM, opt, opt)
    training_state = ppo.init(jax.random.PRNGKey(42))
    return ppo, training_state


def make_norm_state():
    ns = norm_init(OBS_DIM)
    # Simulate some updates so stats are non-trivial
    fake_obs = jnp.ones((4, OBS_DIM)) * 2.0
    from jax_rl.utils.normalization import update as norm_update
    return norm_update(ns, fake_obs)


def make_metrics_log(n=3):
    return [
        {
            "iteration": i,
            "total_steps": (i + 1) * 1000,
            "episodes": i * 10,
            "avg_return": float(i * 100.0),
            "min_return": float(i * 80.0),
            "max_return": float(i * 120.0),
            "policy_loss": -0.01,
            "value_loss": 1.5,
            "entropy": 1.2,
            "approx_kl": 0.01,
            "clip_fraction": 0.05,
            "log_std_mean": -0.3,
            "log_std_min": -0.5,
            "log_std_max": -0.1,
            "sps": 50000,
            "iter_time": 0.7,
        }
        for i in range(n)
    ]


@pytest.fixture
def checkpoint(tmp_path):
    """Save a checkpoint and return the directory path."""
    _, training_state = make_ppo_and_state()
    norm_state = make_norm_state()
    cfg = TrainConfig(env_name="CartpoleBalance", total_timesteps=100_000)
    metrics_log = make_metrics_log(n=3)

    ckpt_dir = str(tmp_path / "test_run")
    _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, OBS_DIM, ACTION_DIM, metrics_log, resume=None)
    return ckpt_dir, training_state, norm_state, cfg, metrics_log


def test_directory_structure(checkpoint):
    ckpt_dir, *_ = checkpoint
    assert os.path.isfile(os.path.join(ckpt_dir, "meta.json")), "meta.json missing"
    assert os.path.isfile(os.path.join(ckpt_dir, "metrics.csv")), "metrics.csv missing"
    assert os.path.isfile(os.path.join(ckpt_dir, "actor_params.npy")), "actor_params.npy missing"
    assert os.path.isdir(os.path.join(ckpt_dir, "orbax")), "orbax/ subfolder missing"


def test_meta_json(checkpoint):
    ckpt_dir, _, _, cfg, _ = checkpoint
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    assert meta["obs_dim"] == OBS_DIM
    assert meta["action_dim"] == ACTION_DIM
    assert "train_config" in meta, "train_config missing from meta.json"

    tc = meta["train_config"]
    assert tc["env_name"] == cfg.env_name
    assert tc["total_timesteps"] == cfg.total_timesteps
    assert "ppo" in tc, "nested PPOConfig missing from train_config"


def test_metrics_csv(checkpoint):
    ckpt_dir, _, _, _, metrics_log = checkpoint
    with open(os.path.join(ckpt_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(metrics_log), f"expected {len(metrics_log)} rows, got {len(rows)}"
    assert int(rows[0]["iteration"]) == 0
    assert int(rows[-1]["iteration"]) == len(metrics_log) - 1
    assert float(rows[1]["avg_return"]) == pytest.approx(100.0)
    # All expected columns present
    for col in ("iteration", "total_steps", "avg_return", "policy_loss", "value_loss", "entropy"):
        assert col in rows[0], f"column '{col}' missing from metrics.csv"


def test_actor_params_npy_content(checkpoint):
    ckpt_dir, training_state, norm_state, _, _ = checkpoint
    saved = np.load(os.path.join(ckpt_dir, "actor_params.npy"), allow_pickle=True).item()

    assert "actor_params" in saved
    assert "norm_mean" in saved
    assert "norm_mean_of_squares" in saved
    assert "norm_count" in saved

    # norm stats match
    np.testing.assert_allclose(saved["norm_mean"], np.array(norm_state.mean), rtol=1e-5)
    np.testing.assert_allclose(saved["norm_mean_of_squares"], np.array(norm_state.mean_of_squares), rtol=1e-5)
    assert int(saved["norm_count"]) == int(norm_state.count)

    # actor_params are not all zeros (real init)
    leaves = jax.tree.leaves(saved["actor_params"])
    assert any(np.any(leaf != 0) for leaf in leaves), "actor_params appear uninitialized"


def test_actor_params_npy_shapes_match(checkpoint):
    ckpt_dir, training_state, _, _, _ = checkpoint
    saved = np.load(os.path.join(ckpt_dir, "actor_params.npy"), allow_pickle=True).item()

    orig_leaves = jax.tree.leaves(training_state.actor_params)
    saved_leaves = jax.tree.leaves(saved["actor_params"])
    assert len(orig_leaves) == len(saved_leaves)
    for orig, saved_l in zip(orig_leaves, saved_leaves):
        assert orig.shape == saved_l.shape, f"shape mismatch: {orig.shape} vs {saved_l.shape}"


def test_orbax_restore(checkpoint):
    ckpt_dir, training_state, norm_state, _, _ = checkpoint
    orbax_dir = os.path.join(ckpt_dir, "orbax")

    target = {"training_state": training_state, "norm_state": norm_state}
    restored = ocp.StandardCheckpointer().restore(os.path.abspath(orbax_dir), target=target)

    # Actor params match
    orig_leaves = jax.tree.leaves(training_state.actor_params)
    rest_leaves = jax.tree.leaves(restored["training_state"].actor_params)
    for orig, rest in zip(orig_leaves, rest_leaves):
        np.testing.assert_allclose(np.array(orig), np.array(rest), rtol=1e-5)

    # Norm state matches
    np.testing.assert_allclose(
        np.array(restored["norm_state"].mean),
        np.array(norm_state.mean),
        rtol=1e-5,
    )


def test_metrics_csv_resume_appends(tmp_path):
    """Saving a second checkpoint with resume should prepend prior rows."""
    _, training_state = make_ppo_and_state()
    norm_state = make_norm_state()
    cfg = TrainConfig(env_name="CartpoleBalance")
    ckpt_dir = str(tmp_path / "run")

    first_log = make_metrics_log(n=3)
    _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, OBS_DIM, ACTION_DIM, first_log, resume=None)

    second_log = make_metrics_log(n=2)
    _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, OBS_DIM, ACTION_DIM, second_log, resume=ckpt_dir)

    with open(os.path.join(ckpt_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(first_log) + len(second_log), \
        f"expected {len(first_log) + len(second_log)} rows after resume, got {len(rows)}"
