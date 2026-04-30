"""Behavioral guard: --resume-warmup policy uses loaded actor, not random uniform.

From `.context/lessons/offpolicy.md` §"Resume Warmup":

> Action selection branch now gates `use_random` on
> `is_warmup AND (start_step == 0 OR resume_warmup == "random")`.

When resuming an off-policy ckpt with the default `resume_warmup="policy"`,
the warmup phase must call `explore_fn` (loaded policy) — never
`jax.random.uniform`. Random-uniform refill of a converged ckpt's buffer
was the bug that tanked first post-resume eval (FlashSAC Cartpole
996→747, FastSAC Go2 268→254).

This test spies on `jax.random.uniform` inside `offpolicy_loop` and
asserts no calls with shape `(num_envs, action_dim)` during the resume
warmup window. Hermetic CPU-only — synthesizes a resume ckpt on disk
then runs `run_offpolicy_loop` once with that ckpt as `resume=`.

Closes coverage gap flagged in
`.context/audits/2026-04-27_test_suite_audit.md` §6 "Resume warmup
behavioral test".
"""
import jax
import jax.random
import optax

import jax_rl.training.offpolicy_loop as ol_module
from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.training import run_offpolicy_loop
from jax_rl.training.checkpointing import save_checkpoint
from jax_rl.utils.normalization import init as norm_init
from tests._loop_helpers import _common_cfg, _patch_eval, _stub_env_bundle


def _build_sac(bundle, cfg):
    """Standard tiny-SAC build for stub-env smoke tests."""
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
    return algo_cfg, algo


def _synthesize_resume_ckpt(ckpt_dir, cfg, algo_cfg, algo, bundle, total_steps=20):
    """Write a real shared-actor ckpt to ckpt_dir.

    Produces:
      - meta.json (via save_checkpoint stamp)
      - metrics.csv with one row, total_steps=`total_steps` so load_checkpoint
        sets start_step > 0 (the gate that kills random-warmup on resume)
      - orbax/ snapshot of training_state + norm_state

    Does NOT actually train — synthesizes from algo.init() output. The actor
    weights are random-init, but that's fine for behavioral testing: what
    matters is whether the *loaded* policy is used for warmup actions.
    """
    training_state = algo.init(jax.random.PRNGKey(123))
    norm_state = norm_init(bundle.obs_dim)
    metrics_log = [{
        "total_steps": total_steps,
        "ep_return_mean": 0.5,
        "sps": 100,
    }]
    save_checkpoint(
        ckpt_dir, training_state, norm_state, cfg, algo_cfg,
        algo_name="sac", obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        metrics_log=metrics_log, resume=None,
        critic_norm_state=None, env=bundle.env,
    )


def _install_uniform_spy(monkeypatch):
    """Patch jax.random.uniform via the offpolicy_loop module binding.

    Returns a `calls` list — each entry is the `shape` tuple passed to
    a uniform call. The warmup-action call site (offpolicy_loop.py:189)
    uses shape `(num_envs, action_dim)`; SAC/TD3 internals use other
    shapes, so filtering on (num_envs, action_dim) isolates the warmup
    branch.
    """
    real_uniform = jax.random.uniform
    calls = []

    def _spy_uniform(key, shape=(), *args, **kwargs):
        calls.append(tuple(shape) if hasattr(shape, "__iter__") else (shape,))
        return real_uniform(key, shape, *args, **kwargs)

    monkeypatch.setattr(ol_module.jax.random, "uniform", _spy_uniform)
    return calls


def test_resume_warmup_policy_does_not_call_random_uniform(tmp_path, monkeypatch):
    """resume_warmup='policy' (default) → warmup actions come from loaded
    policy, not jax.random.uniform.

    This is the behavioral claim of `--resume-warmup policy`. If a future
    refactor breaks the gate (e.g., flips the boolean, drops start_step
    check), this test catches it before it ships and tanks first
    post-resume eval.
    """
    bundle = _stub_env_bundle(num_envs=2, action_dim=2)
    cfg = _common_cfg(num_envs=2)
    algo_cfg, algo = _build_sac(bundle, cfg)

    ckpt_dir = str(tmp_path / "phase1_ckpt")
    _synthesize_resume_ckpt(ckpt_dir, cfg, algo_cfg, algo, bundle, total_steps=20)

    # Spy must be installed AFTER algo.init() / save_checkpoint above —
    # those legitimately use uniform during weight init.
    calls = _install_uniform_spy(monkeypatch)
    _patch_eval(monkeypatch)
    monkeypatch.chdir(tmp_path)

    explore = lambda p, o, k: algo.select_action(p, o, k)
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=ckpt_dir, resume_warmup="policy",
        use_wandb=False,
    )

    warmup_action_shape = (bundle.num_envs, bundle.action_dim)
    warmup_calls = [s for s in calls if s == warmup_action_shape]
    assert warmup_calls == [], (
        f"resume_warmup='policy' still called jax.random.uniform with the "
        f"warmup-action shape {warmup_action_shape}: {warmup_calls}"
    )


def test_resume_warmup_random_does_call_random_uniform(tmp_path, monkeypatch):
    """resume_warmup='random' (opt-in) → warmup actions DO come from
    jax.random.uniform.

    Verifies the gate works in both directions: the policy-mode test alone
    can't distinguish "policy mode works" from "uniform was never called
    anywhere." This catches a regression where someone accidentally hard-
    wires the no-random branch and silently breaks `--resume-warmup random`
    for users who explicitly want a buffer-distribution reset.
    """
    bundle = _stub_env_bundle(num_envs=2, action_dim=2)
    cfg = _common_cfg(num_envs=2)
    algo_cfg, algo = _build_sac(bundle, cfg)

    ckpt_dir = str(tmp_path / "phase1_ckpt")
    _synthesize_resume_ckpt(ckpt_dir, cfg, algo_cfg, algo, bundle, total_steps=20)

    calls = _install_uniform_spy(monkeypatch)
    _patch_eval(monkeypatch)
    monkeypatch.chdir(tmp_path)

    explore = lambda p, o, k: algo.select_action(p, o, k)
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=ckpt_dir, resume_warmup="random",
        use_wandb=False,
    )

    warmup_action_shape = (bundle.num_envs, bundle.action_dim)
    warmup_calls = [s for s in calls if s == warmup_action_shape]
    assert len(warmup_calls) > 0, (
        f"resume_warmup='random' did NOT call jax.random.uniform with the "
        f"warmup-action shape {warmup_action_shape}. Got shapes: {calls}"
    )
