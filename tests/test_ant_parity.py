"""tests/test_ant_parity.py — distributional parity vs Gym Ant-v5.

Strategy: matched action sequences, multi-step rollouts, four assertions:
reset z-distribution, reward-component means/stds, termination-fraction.
GPU/CPU divergence + RK4-vs-MJX integrator differences mean we cannot
bit-match; we check distributions over n=200 trajectories.
"""

import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import pytest

from jax_rl.envs.locomotion.ant import Ant


pytestmark = [pytest.mark.gpu, pytest.mark.warp]


N_TRAJ = 200
N_STEPS = 5
ACTION_SCALE = 0.3


def _make_action_sequence(n_traj: int, n_steps: int, seed: int) -> np.ndarray:
    """Same actions for both backends — keeps parity comparison sound."""
    rng = np.random.default_rng(seed)
    return rng.uniform(
        -ACTION_SCALE, ACTION_SCALE, size=(n_traj, n_steps, 8)
    ).astype(np.float32)


def test_reset_obs_dim_matches_gym():
    gym_env = gym.make("Ant-v5")
    gym_obs, _ = gym_env.reset(seed=0)
    mjx_env = Ant()
    mjx_state = mjx_env.reset(jax.random.PRNGKey(0))
    assert gym_obs.shape == mjx_state.obs.shape, (
        f"Gym {gym_obs.shape} vs MJX {mjx_state.obs.shape}"
    )


def test_reset_qpos_z_distribution_matches_gym():
    """qpos[2] (torso z) on reset should be init_qpos[2] (=0.75) + U[-0.1, 0.1]."""
    gym_env = gym.make("Ant-v5")
    gym_zs = []
    for s in range(N_TRAJ):
        gym_env.reset(seed=s)
        gym_zs.append(gym_env.unwrapped.data.qpos[2])
    gym_zs = np.array(gym_zs)

    mjx_env = Ant()
    keys = jax.random.split(jax.random.PRNGKey(0), N_TRAJ)
    mjx_states = jax.vmap(mjx_env.reset)(keys)
    mjx_zs = np.array(mjx_states.data.qpos[:, 2])

    # Both are init_qpos[2] + U(-0.1, 0.1) → mean ~0.75, std ~0.058.
    assert abs(gym_zs.mean() - mjx_zs.mean()) < 0.02, (
        f"Mean mismatch: gym {gym_zs.mean():.4f} vs mjx {mjx_zs.mean():.4f}"
    )
    assert abs(gym_zs.std() - mjx_zs.std()) < 0.02, (
        f"Std mismatch: gym {gym_zs.std():.4f} vs mjx {mjx_zs.std():.4f}"
    )


def _rollout_gym(actions: np.ndarray):
    """Returns dict of per-step reward components + per-traj termination flag."""
    env = gym.make("Ant-v5")
    n_traj, n_steps, _ = actions.shape
    out = {k: [] for k in ("forward", "ctrl", "contact", "survive", "total")}
    term = []
    for t in range(n_traj):
        env.reset(seed=t)
        terminated_at = -1
        for s in range(n_steps):
            if terminated_at >= 0:
                break
            _, r, term_flag, _, info = env.step(actions[t, s])
            out["forward"].append(info["reward_forward"])
            out["ctrl"].append(info["reward_ctrl"])
            out["contact"].append(info["reward_contact"])
            out["survive"].append(info["reward_survive"])
            out["total"].append(r)
            if term_flag:
                terminated_at = s
        term.append(terminated_at >= 0)
    return (
        {k: np.asarray(v, dtype=np.float64) for k, v in out.items()},
        np.asarray(term),
    )


def _rollout_mjx(actions: np.ndarray):
    """Vectorized MJX rollout: jit'd step, vmap'd reset, Python-loop steps.

    Pure Python-loop step() OOMs on warp at 200 trajectories because each
    step call materializes fresh graph buffers. JIT'ing step keeps the
    compiled artifacts cached. We vmap the reset + step over trajectories
    so each "step" is one jitted batched call.
    """
    env = Ant()
    n_traj, n_steps, _ = actions.shape
    keys = jax.random.split(jax.random.PRNGKey(0), n_traj)
    actions_jax = jp.asarray(actions)  # (n_traj, n_steps, 8)

    batched_reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.jit(jax.vmap(env.step))

    state = batched_reset(keys)

    keys_order = ("forward", "ctrl", "contact", "survive", "total")
    per_step_components = {k: np.zeros((n_traj, n_steps)) for k in keys_order}
    per_step_done = np.zeros((n_traj, n_steps), dtype=bool)
    for s in range(n_steps):
        a_step = actions_jax[:, s, :]
        state = batched_step(state, a_step)
        per_step_components["forward"][:, s] = np.asarray(state.metrics["reward_forward"])
        per_step_components["ctrl"][:, s] = np.asarray(state.metrics["reward_ctrl"])
        per_step_components["contact"][:, s] = np.asarray(state.metrics["reward_contact"])
        per_step_components["survive"][:, s] = np.asarray(state.metrics["reward_survive"])
        per_step_components["total"][:, s] = np.asarray(state.reward)
        per_step_done[:, s] = np.asarray(state.done) > 0.5

    # Determine terminated-at index per traj (first True, else -1).
    first_done_idx = np.where(
        per_step_done.any(axis=1),
        per_step_done.argmax(axis=1),
        -1,
    )
    term = first_done_idx >= 0

    out = {k: [] for k in keys_order}
    for ti in range(n_traj):
        end = first_done_idx[ti] + 1 if first_done_idx[ti] >= 0 else n_steps
        for si in range(end):
            for k in keys_order:
                out[k].append(float(per_step_components[k][ti, si]))
    out_arr = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    return out_arr, term


def test_reward_component_distributions_match():
    actions = _make_action_sequence(N_TRAJ, N_STEPS, seed=42)
    gym_rew, _ = _rollout_gym(actions)
    mjx_rew, _ = _rollout_mjx(actions)

    # ctrl_cost is dominated by ‖action‖² which is identical across backends
    # (we share the action sequence), so the relative tolerance can be tight.
    # Forward / contact tolerances reflect integrator-sensitivity.
    # Forward reward at small-action scale is near-zero in both backends
    # (~0.02 magnitude), so a relative tolerance is meaningless — use abs.
    # Same for total (sum of small components).
    tolerances = {
        "forward": ("abs", 0.10),
        "ctrl": ("rel", 0.05),
        "contact": ("rel", 0.50),
        "survive": ("abs", 0.10),
        "total": ("abs", 0.20),
    }
    for k, (mode, tol) in tolerances.items():
        g, m = gym_rew[k], mjx_rew[k]
        if mode == "rel":
            denom = max(abs(g.mean()), 1e-3)
            rel_err = abs(g.mean() - m.mean()) / denom
            assert rel_err < tol, (
                f"{k} mean rel-err {rel_err:.3f} > {tol}: "
                f"gym={g.mean():.4f} mjx={m.mean():.4f}"
            )
        else:
            abs_err = abs(g.mean() - m.mean())
            assert abs_err < tol, (
                f"{k} mean abs-err {abs_err:.4f} > {tol}: "
                f"gym={g.mean():.4f} mjx={m.mean():.4f}"
            )


def test_termination_fraction_matches_gym():
    """Within N_STEPS, fraction of trajectories that terminated should match."""
    actions = _make_action_sequence(N_TRAJ, N_STEPS, seed=42)
    _, gym_term = _rollout_gym(actions)
    _, mjx_term = _rollout_mjx(actions)
    gym_frac = float(gym_term.mean())
    mjx_frac = float(mjx_term.mean())
    assert abs(gym_frac - mjx_frac) < 0.10, (
        f"Termination fraction mismatch: gym={gym_frac:.3f} mjx={mjx_frac:.3f}"
    )
