"""scripts/plot_skill_xy.py — DIAYN-canonical xy-trajectory diversity figure.

Loads a skill-discovery checkpoint, rolls out N steps per skill from a fixed
initial state (no reset noise), captures torso CoM xy per step, and plots one
figure with all skills overlaid (color = skill index).

Numerical visual gate: max pairwise xy-endpoint distance > 3.0 m OR circular
std of skill endpoint headings > 30 deg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from jax_rl.envs.locomotion.ant import Ant, default_config
from jax_rl.training.checkpointing import load_actor_for_inference
from scripts.record_video import (
    _resolve_skill_vector,
    _SkillWrappedAlgo,
    _build_select_action,
)


def _make_env_for_ckpt(meta: dict) -> Ant:
    """Construct the Ant variant matching the trained checkpoint.

    The checkpoint's meta.json carries the env_name; AntMJXClassic and AntMJX
    differ only in `include_cfrc_ext_in_observation`.
    """
    env_name = meta.get("env_name") or meta.get("train_config", {}).get("env_name", "AntMJX")
    cfg = default_config()
    cfg.unlock()
    if env_name == "AntMJXClassic":
        cfg.include_cfrc_ext_in_observation = False
    return Ant(config=cfg)


def deterministic_reset(env: Ant) -> mjx_env.State:
    """Reset env to init_qpos / init_qvel WITHOUT random noise.

    Different from env.reset() which adds U(-0.1, 0.1) qpos noise +
    N(0, 0.1) qvel noise. We need shared init across skills/rollouts so
    the figure shows skill-driven variance, not reset-noise variance.
    """
    data = mjx_env.make_data(
        env.mj_model,
        qpos=env._init_qpos,
        qvel=env._init_qvel,
        impl=env.mjx_model.impl.value,
        naconmax=env._config.naconmax,
        njmax=env._config.njmax,
    )
    data = mjx.forward(env.mjx_model, data)
    info = {"rng": jax.random.PRNGKey(0),
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "distance_from_origin": jp.linalg.norm(data.qpos[0:2]),
            "x_velocity": jp.zeros(()),
            "y_velocity": jp.zeros(())}
    metrics = {k: jp.zeros(()) for k in
               ("reward_forward", "reward_ctrl", "reward_contact", "reward_survive")}
    reward, done = jp.zeros(2)
    obs = env._get_obs(data)
    return mjx_env.State(data, obs, reward, done, metrics, info)


def _evaluate_gate(per_skill_endpoints: dict[int, list]) -> dict:
    """Compute and print the numerical visual-gate metrics.

    Returns metrics dict for caller use (e.g. journal write).
    """
    skills = sorted(per_skill_endpoints.keys())
    mean_endpoints = np.stack([
        np.mean(per_skill_endpoints[k], axis=0) for k in skills
    ])  # (num_skills, 2)

    diffs = mean_endpoints[:, None, :] - mean_endpoints[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    max_pairwise = float(dists.max())

    headings = np.arctan2(mean_endpoints[:, 1], mean_endpoints[:, 0])
    R = np.sqrt(np.mean(np.cos(headings)) ** 2 + np.mean(np.sin(headings)) ** 2)
    circ_std_rad = float(np.sqrt(-2.0 * np.log(max(R, 1e-9))))
    heading_std_deg = float(np.degrees(circ_std_rad))

    hpd = np.abs(((headings[:, None] - headings[None, :] + np.pi)
                  % (2 * np.pi)) - np.pi)
    max_pairwise_heading_deg = float(np.degrees(hpd.max()))

    print(f"max pairwise xy-endpoint distance: {max_pairwise:.3f} m")
    print(f"circular std of skill endpoint headings: {heading_std_deg:.1f} deg")
    print(f"max pairwise heading separation: {max_pairwise_heading_deg:.1f} deg")

    gate_pass = (max_pairwise > 3.0) or (heading_std_deg > 30.0)
    print(f"VISUAL GATE: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "max_pairwise_dist_m": max_pairwise,
        "heading_circ_std_deg": heading_std_deg,
        "max_pairwise_heading_deg": max_pairwise_heading_deg,
        "gate_pass": gate_pass,
        "mean_endpoints": mean_endpoints.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts-per-skill", type=int, default=3)
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    meta, actor_params, norm_state, _ = load_actor_for_inference(args.checkpoint)
    total_skill_dim = int(meta["skill_discovery"]["total_skill_dim"])

    env = _make_env_for_ckpt(meta)
    raw_obs_dim = int(env.observation_size)
    augmented_obs_dim = raw_obs_dim + total_skill_dim
    action_dim = env.action_size

    algo, kind = _build_select_action(meta, augmented_obs_dim, action_dim)
    assert kind in ("offpolicy",), f"expected SAC family, got kind={kind}"

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 8))

    per_skill_endpoints: dict[int, list] = {}
    main_body_id = env._config.main_body_id
    rollout_length = args.rollout_length

    def make_rollout_fn(z_skill):
        wrapped = _SkillWrappedAlgo(inner=algo, skill_z=z_skill)

        def rollout_step(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            obs_b = state.obs[None, :]
            action = wrapped.select_action(actor_params, obs_b, sub, deterministic=True)
            new_state = env.step(state, action[0])
            xy = new_state.data.xpos[main_body_id, :2]
            return (new_state, key), xy

        @jax.jit
        def rollout(init_state, init_key):
            (final_state, _), xys = jax.lax.scan(
                rollout_step, (init_state, init_key), None, length=rollout_length
            )
            return xys, final_state

        return rollout

    for skill_idx in range(total_skill_dim):
        z = _resolve_skill_vector(meta, skill_idx, None)
        assert z is not None, "ckpt missing skill_discovery block"
        rollout = make_rollout_fn(z)

        per_skill_endpoints[skill_idx] = []

        for rollout_idx in range(args.rollouts_per_skill):
            key = jax.random.PRNGKey(rollout_idx + skill_idx * 1_000)
            state = deterministic_reset(env)
            init_xy = np.array(state.data.xpos[main_body_id, :2])
            xys_jax, _ = rollout(state, key)
            xys = np.concatenate([init_xy[None, :], np.asarray(xys_jax)], axis=0)
            per_skill_endpoints[skill_idx].append(xys[-1])
            ax.plot(xys[:, 0], xys[:, 1], color=cmap(skill_idx),
                    alpha=0.5, linewidth=1.0,
                    label=f"z{skill_idx}" if rollout_idx == 0 else None)

    ax.scatter([0], [0], marker="x", color="black", s=80, zorder=5, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    title = args.title or f"Ant DIAYN skill diversity ({total_skill_dim} skills × {args.rollouts_per_skill} rollouts)"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved figure to {args.output}")

    _evaluate_gate(per_skill_endpoints)


if __name__ == "__main__":
    main()
