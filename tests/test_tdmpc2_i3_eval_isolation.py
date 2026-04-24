"""I3 Test 27: eval/collect prev_mean isolation regression guard.

Verifies that run_eval does NOT mutate state.prev_mean. The claim (iter-4 isolation
requirement) is that eval uses a local eval_prev_mean tensor and never writes back to
state.prev_mean. Without this guard, a bug that accidentally used state.prev_mean inside
run_eval would silently shift the collect trajectory after every eval call.
"""
import os
import subprocess
import pytest


@pytest.mark.slow
def test_eval_does_not_mutate_collect_prev_mean():
    """Interleave: collect → eval → collect. Assert state.prev_mean after the second
    collect equals what it would be without the eval call.

    Two parallel builds from the same seed:
      Run A: collect → eval → collect
      Run B: collect → collect (no eval)

    If eval is properly isolated, final state.prev_mean must be identical in both runs.
    """
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)

    script = """
import sys
sys.path.insert(0, '/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl')
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.algos.tdmpc2 import make_plan_batched
from train_tdmpc2 import (
    build_modules, init_train_state, build_train_config_from_tdmpc2,
    run_warmup, run_eval, _pipe_obs,
)
from jax_rl.algos.tdmpc2 import make_update_step, build_world_model_optimizer, build_policy_optimizer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.training.env_setup import make_env_bundle

cfg = get_tdmpc2_preset('CheetahRun')
cfg = dataclasses.replace(
    cfg,
    num_envs=2, seed_steps=4, batch_size=4, utd=1,
    num_eval_envs=1, eval_every=4, total_steps=20,
    mppi_iterations=1, num_samples=8, num_elites=2, num_pi_trajs=1,
)


def build():
    train_cfg = build_train_config_from_tdmpc2(cfg, 'CheetahRun', 20, 0)
    env_bundle = make_env_bundle(train_cfg, 0)
    buffer = JaxReplayBuffer(
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        max_size=cfg.buffer_size,
    )
    state, wm_opt, pol_opt = init_train_state(cfg, env_bundle.obs_dim, 0)
    modules = build_modules(cfg)
    encoder, dynamics, reward_net, q_ensemble, policy = modules
    update_step = make_update_step(
        cfg, wm_opt, pol_opt,
        encoder=encoder, dynamics=dynamics,
        reward_net=reward_net, q_ensemble_net=q_ensemble,
        policy_net=policy,
    )
    plan_fn = make_plan_batched(
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )
    key = jax.random.PRNGKey(42)
    state, env_state, key, episode_ids, prev_done = run_warmup(
        state, env_bundle, buffer, update_step, cfg, key,
    )
    return state, env_bundle, buffer, plan_fn, modules, env_state, key, episode_ids, prev_done


def collect_step(state, env_bundle, plan_fn, modules, env_state, key, prev_done):
    # Single collect step. Advances env and updates state.prev_mean. Buffer write skipped.
    encoder = modules[0]
    dict_obs = env_bundle.dict_obs
    env_step = env_bundle.env_step

    obs = _pipe_obs(env_state.obs, dict_obs)
    z_0 = encoder.apply(state.encoder_params, obs)
    plan_params = {
        'encoder': state.encoder_params,
        'dynamics': state.dynamics_params,
        'reward': state.reward_params,
        'q_ensemble': state.q_ensemble_params,
        'policy': state.policy_params,
    }
    key, plan_key = jax.random.split(key)
    plan_keys = jax.random.split(plan_key, cfg.num_envs)
    action, new_prev_mean = plan_fn(
        plan_params, z_0, state.prev_mean, prev_done, cfg, plan_keys, False,
    )
    state = state.replace(prev_mean=new_prev_mean)
    env_state = env_step(env_state, action)
    next_done = (
        env_state.done.astype(jnp.bool_)
        | env_state.info.get('truncation', jnp.zeros_like(env_state.done)).astype(jnp.bool_)
    )
    return state, env_state, key, next_done


# ---- Run A: collect → eval → collect ----
print('Run A: collect, eval, collect')
state_a, env_bundle_a, buffer_a, plan_fn_a, modules_a, env_state_a, key_a, _, pd_a = build()

state_a, env_state_a, key_a, pd_a = collect_step(
    state_a, env_bundle_a, plan_fn_a, modules_a, env_state_a, key_a, pd_a,
)
key_a, eval_key = jax.random.split(key_a)
_ = run_eval(state_a, env_bundle_a, plan_fn_a, modules_a, cfg, eval_key)
state_a, env_state_a, key_a, pd_a = collect_step(
    state_a, env_bundle_a, plan_fn_a, modules_a, env_state_a, key_a, pd_a,
)
final_prev_a = np.asarray(state_a.prev_mean)

# ---- Run B: collect → collect (no eval) ----
print('Run B: collect, collect (no eval)')
state_b, env_bundle_b, buffer_b, plan_fn_b, modules_b, env_state_b, key_b, _, pd_b = build()

state_b, env_state_b, key_b, pd_b = collect_step(
    state_b, env_bundle_b, plan_fn_b, modules_b, env_state_b, key_b, pd_b,
)
# No eval call here — key not consumed for eval either, so key_b stays identical to key_a
# at this point. We need to advance key_b by the same number of splits as Run A consumed
# for the eval key split (one jax.random.split call).
key_b, _discarded = jax.random.split(key_b)  # mirror the eval key split in Run A
state_b, env_state_b, key_b, pd_b = collect_step(
    state_b, env_bundle_b, plan_fn_b, modules_b, env_state_b, key_b, pd_b,
)
final_prev_b = np.asarray(state_b.prev_mean)

# ---- Compare ----
max_diff = np.abs(final_prev_a - final_prev_b).max()
if np.allclose(final_prev_a, final_prev_b, atol=1e-6):
    print('ISOLATION_OK')
else:
    print(f'ISOLATION_BROKEN: max_diff={max_diff:.2e}')
    print(f'  final_prev_a shape: {final_prev_a.shape}')
    print(f'  final_prev_b shape: {final_prev_b.shape}')
"""

    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
        env=env,
        capture_output=True,
        timeout=1200,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert "ISOLATION_OK" in stdout, (
        f"Eval mutated collect prev_mean!\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}"
    )
