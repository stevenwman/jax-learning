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

import pytest

pytestmark = pytest.mark.gpu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_run_offpolicy_loop_stub_env_cpu(tmp_path, monkeypatch):
    """CPU-only smoke test with a stub env. <5s. No GPU. The Task 4 commit gate.

    Constructs an EnvBundle manually with stub env callables that return JAX
    arrays of the right shape. SAC is real (exercises init/update/
    select_action/get_q_value), env is fake. eval_runner functions are
    monkeypatched to no-ops so we test ONLY the helper's loop body.

    What this catches:
        - Wrong arg order / missing kwargs in run_offpolicy_loop call sites
        - Broken closure / explore_fn signature mismatch
        - Incorrect dict-obs/has_privileged branching
        - Buffer add_batch shape mismatch
        - Truncation handling regression
        - JAX leak / tracer error
        - Field name typos in TrainContext / EnvBundle construction
    """
    import dataclasses
    from dataclasses import dataclass

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from jax_rl.algos.sac import SAC
    from jax_rl.configs.sac_config import SACConfig
    from jax_rl.configs.train_config import TrainConfig
    from jax_rl.training import EnvBundle, run_offpolicy_loop
    import jax_rl.training.offpolicy_loop as ol_module

    # ── Tiny dimensions ────────────────────────────────────────────────────
    NUM_ENVS = 2
    OBS_DIM = 4
    ACTION_DIM = 2

    # ── Stub env state + step ──────────────────────────────────────────────
    @dataclass
    class StubEnvState:
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray
        info: dict

    def stub_step(state, action):
        # Drift obs slightly each step so update_stats sees variation
        new_obs = state.obs + 0.01 * jnp.ones_like(state.obs)
        return StubEnvState(
            obs=new_obs,
            reward=jnp.ones((NUM_ENVS,)) * 0.5,
            done=jnp.zeros((NUM_ENVS,)),
            info={"truncation": jnp.zeros((NUM_ENVS,))},
        )

    initial_state = StubEnvState(
        obs=jnp.zeros((NUM_ENVS, OBS_DIM)),
        reward=jnp.zeros((NUM_ENVS,)),
        done=jnp.zeros((NUM_ENVS,)),
        info={"truncation": jnp.zeros((NUM_ENVS,))},
    )

    # eval_env is unused (eval runner is mocked) but must be a non-None object
    class StubEnv:
        action_size = ACTION_DIM
        def reset(self, keys): return initial_state
        def step(self, state, action): return stub_step(state, action)

    env = StubEnv()
    bundle = EnvBundle(
        env=env, env_step=stub_step, env_state=initial_state,
        eval_env=env, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        critic_obs_dim=None, has_privileged=False, dict_obs=False,
        key=jax.random.PRNGKey(0),
    )

    # ── Mock eval/checkpoint to no-ops ─────────────────────────────────────
    # The eval runner is fully tested elsewhere; here we want to exercise
    # only the loop body. The mock pulls last_eval_eps + key from kwargs
    # first, falling back to positional args 7/8 — that way the test still
    # works if the helper switches to keyword arguments later.
    def _noop_maybe_eval(*args, **kwargs):
        last_eval_eps = kwargs.get("last_eval_eps", args[7] if len(args) > 7 else 0)
        key = kwargs.get("key", args[8] if len(args) > 8 else None)
        return last_eval_eps, key

    def _noop_final_eval(*args, **kwargs):
        return None

    monkeypatch.setattr(ol_module, "maybe_eval_and_checkpoint", _noop_maybe_eval)
    monkeypatch.setattr(ol_module, "final_eval_and_checkpoint", _noop_final_eval)

    # ── Configs (intentionally tiny) ───────────────────────────────────────
    cfg = TrainConfig(
        env_name="StubEnv", num_envs=NUM_ENVS, total_timesteps=40,
        episode_length=100, eval_every_n_episodes=10**9,  # eval never fires
        gamma=0.99, lr=3e-4, reward_scaling=1.0, n_frame_stack=1,
        handle_truncation=True,
    )
    algo_cfg = SACConfig(
        hidden_dim=(32, 32), batch_size=8,
        min_buffer_size=10, buffer_size=100,
        grad_updates_per_step=1,
    )

    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(
        config=algo_cfg, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        optimizer=optimizer, alpha_optimizer=alpha_opt,
        gamma=cfg.gamma, critic_obs_dim=None,
    )

    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    monkeypatch.chdir(tmp_path)

    # Should complete without exception. Catches the bugs listed in docstring.
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=None, use_wandb=False,
    )


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
