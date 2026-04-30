"""Smoke tests for run_offpolicy_loop.

The CPU stub-env test (test_run_offpolicy_loop_stub_env_cpu) is the GATE
for committing the helper — it runs in <5s without GPU, exercises the
full training loop body (env step → buffer → gradient updates → logging),
and catches copy-paste / glue bugs that would otherwise survive until
the deferred GPU smoke tests.

The slow real-env test (test_run_offpolicy_loop_sac_cheetah) is the
end-to-end validation, deferred to when GPU is free.
"""
import os
import sys

import optax
import pytest

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.algos.sac import SAC
from jax_rl.algos.td3 import TD3
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.training import run_offpolicy_loop
from tests._loop_helpers import _common_cfg, _patch_eval, _stub_env_bundle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run_stub_loop(tmp_path, monkeypatch, algo_cfg, algo, algo_name,
                   log_extra_fields=None, log_extra_keys=None):
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    explore = lambda p, o, k: algo.select_action(p, o, k)

    _patch_eval(monkeypatch)
    monkeypatch.chdir(tmp_path)

    run_offpolicy_loop(
        cfg=cfg,
        algo_cfg=algo_cfg,
        algo=algo,
        algo_name=algo_name,
        env_bundle=bundle,
        explore_fn=explore,
        log_extra_fields=log_extra_fields or [],
        log_extra_keys=log_extra_keys or [],
        seed=0,
        resume=None,
        use_wandb=False,
    )


def test_run_offpolicy_loop_stub_env_sac_cpu(tmp_path, monkeypatch):
    """SAC loop smoke with a CPU-only stub env."""
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = SACConfig(
        hidden_dim=(32, 32), batch_size=8,
        min_buffer_size=10, buffer_size=100,
        grad_updates_per_step=1,
    )

    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_opt,
        gamma=cfg.gamma, critic_obs_dim=None,
    )
    _run_stub_loop(
        tmp_path,
        monkeypatch,
        algo_cfg,
        algo,
        "sac",
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
    )


def test_run_offpolicy_loop_stub_env_td3_cpu(tmp_path, monkeypatch):
    """TD3 loop smoke with a CPU-only stub env."""
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = TD3Config(
        hidden_dim=(32, 32), batch_size=8,
        min_buffer_size=10, buffer_size=100,
        grad_updates_per_step=1,
    )
    actor_opt = optax.adam(cfg.lr)
    critic_opt = optax.adam(cfg.lr)
    algo = TD3(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        gamma=cfg.gamma, critic_obs_dim=None,
    )
    _run_stub_loop(tmp_path, monkeypatch, algo_cfg, algo, "td3")


def test_run_offpolicy_loop_stub_env_fast_sac_cpu(tmp_path, monkeypatch):
    """FastSAC loop smoke with a CPU-only stub env."""
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = FastSACConfig(
        hidden_dim=(32, 32), critic_hidden_dim=(32, 32),
        batch_size=8, min_buffer_size=10, buffer_size=100,
        grad_updates_per_step=1, num_atoms=11,
    )
    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = FastSAC(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_opt,
        gamma=cfg.gamma, critic_obs_dim=None,
    )
    _run_stub_loop(
        tmp_path,
        monkeypatch,
        algo_cfg,
        algo,
        "fast_sac",
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
    )


def test_run_offpolicy_loop_stub_env_fast_td3_cpu(tmp_path, monkeypatch):
    """FastTD3 loop smoke with a CPU-only stub env."""
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = FastTD3Config(
        hidden_dim=(32, 32), critic_hidden_dim=(32, 32),
        batch_size=8, min_buffer_size=10, buffer_size=100,
        grad_updates_per_step=1, num_atoms=11,
    )
    actor_opt = optax.adam(cfg.lr)
    critic_opt = optax.adam(cfg.lr)
    algo = FastTD3(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        gamma=cfg.gamma, critic_obs_dim=None,
    )
    _run_stub_loop(tmp_path, monkeypatch, algo_cfg, algo, "fast_td3")


@pytest.mark.gpu
@pytest.mark.slow
def test_run_offpolicy_loop_sac_cheetah(tmp_path, monkeypatch):
    """Run SAC + CheetahRun for 2000 env steps — helper should complete without error.

    This is a smoke test: it verifies the glue works end-to-end with a real
    env + real algo. It does NOT verify training quality (too few steps).

    Uses monkeypatch.chdir (pytest fixture) so cwd is restored even on failure —
    `run_offpolicy_loop` creates `checkpoints/<timestamp>_<algo>_<env>_<seed>/`
    relative to cwd and we don't want it in the repo.
    """
    import dataclasses
    import optax

    from jax_rl.algos.sac import SAC
    from jax_rl.configs.env_presets import get_sac_preset
    from jax_rl.training import make_env_bundle, run_offpolicy_loop

    cfg, algo_cfg = get_sac_preset("CheetahRun")
    cfg = dataclasses.replace(cfg, num_envs=4, total_timesteps=2000)
    algo_cfg = dataclasses.replace(algo_cfg, min_buffer_size=500, batch_size=64)

    monkeypatch.chdir(tmp_path)

    bundle = make_env_bundle(cfg, seed=0)

    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_opt,
        gamma=cfg.gamma, critic_obs_dim=bundle.critic_obs_dim,
    )

    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    # Should complete without exception.
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=None, use_wandb=False,
    )
