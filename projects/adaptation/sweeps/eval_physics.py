"""Physics-metric sweep for splitbelt PoseDR ckpts: lag, sway, failure rate.

Reports cold-hard physical metrics per (vL, vR) pair instead of reward:
  - Mean final |drift_x| (lag)         — how far backward the robot ended up
  - Mean max  |drift_x| during episode — peak backward excursion
  - Mean max  |drift_y| (sway)         — peak lateral excursion
  - Termination rate                   — fraction of episodes that died early
  - Failure mode breakdown:
        term_cause=1 fall_torso  (torso contact)
        term_cause=2 off_belt    (foot left belt span)
        term_cause=3 tilt        (upvector_z<0.5 or base_z<0.18)
        term_cause=4 cross_belt  (grounded foot on opposite belt from spawn)
  - Mean survival steps                — episode length (max=episode_length)

Uses vmap to roll out batch_size envs in parallel for speed.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/eval_splitbelt_physics.py \\
      --checkpoint checkpoints/.../best \\
      --pairs 0.5,0.5 0.5,1.0 0.5,1.5 0.3,1.5 0.3,0.9 1.0,0.3 \\
      --num-episodes 16 --episode-length 1000
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from jax_rl.training.checkpointing import load_actor_for_inference
from jax_rl.utils.normalization import normalize as norm_normalize
import jax_rl.training.env_setup  # noqa: F401

from scripts.record_video import _build_select_action  # type: ignore


def make_env(schedule_kind, schedule_params, base_env_name="Go2WarpSplitbeltPoseDR"):
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from mujoco_playground import registry as pg_registry
    template = pg_registry.load(base_env_name)
    cfg = config_dict.ConfigDict(template._config.to_dict())
    cfg.unlock()
    cfg.schedule_kind = schedule_kind
    cfg.schedule_params = config_dict.create(**schedule_params)
    task_map = {
        "Go2WarpSplitbelt":               "splitbelt",
        "Go2WarpSplitbeltDR":             "splitbelt_dr",
        "Go2WarpSplitbeltPoseDR":         "splitbelt_pose_dr",
        "Go2WarpSplitbeltPosTrack":       "splitbelt_pos_track",
        "Go2WarpSplitbeltPosTrackTiedDR": "splitbelt_pos_track_tied_dr",
    }
    return Go2WarpSplitbeltEnv(task=task_map[base_env_name], config=cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-env", default="Go2WarpSplitbeltPoseDR")
    args = p.parse_args()

    meta, actor_params, norm_state, batch_stats = load_actor_for_inference(args.checkpoint)
    print(f"Loaded {meta.get('algo')} from {args.checkpoint}")

    # Probe one env for dims.
    probe = make_env("split_constant", {"vL": 0.5, "vR": 0.5}, args.base_env)
    s0 = probe.reset(jax.random.PRNGKey(0))
    obs = s0.obs["state"] if isinstance(s0.obs, dict) else s0.obs
    obs_dim, action_dim = int(obs.shape[-1]), int(probe.action_size)

    algo, _ = _build_select_action(meta, obs_dim, action_dim)
    init_key = jax.random.PRNGKey(args.seed)
    training_state = algo.init(init_key)
    if actor_params is not None:
        training_state = training_state.replace(actor_params=actor_params)
    if batch_stats is not None and hasattr(training_state, "actor_batch_stats"):
        training_state = training_state.replace(actor_batch_stats=batch_stats)
        algo._default_actor_bs = batch_stats

    use_obs_norm = bool(meta.get("fast_sac_config", {}).get("obs_normalization", False))

    def eval_pair(env, n_eps, ep_len):
        @jax.jit
        def reset_batch(key):
            keys = jax.random.split(key, n_eps)
            return jax.vmap(env.reset)(keys)

        @jax.jit
        def scan_step(carry, step_key):
            state, ep_done = carry
            obs = state.obs
            flat = obs["state"] if isinstance(obs, dict) else obs
            if use_obs_norm and norm_state is not None:
                flat = norm_normalize(flat, norm_state)
            keys = jax.random.split(step_key, n_eps)
            action = jax.vmap(
                lambda p, o, k: algo.select_action(p, o, k, deterministic=True),
                in_axes=(None, 0, 0),
            )(actor_params, flat, keys)
            new_state = jax.vmap(env.step)(state, action)
            done = new_state.done.astype(jnp.bool_)
            new_done = ep_done | done
            sb = new_state.info["splitbelt"]
            per_step = {
                "drift_x":   sb["drift_xy"][:, 0],
                "drift_y":   sb["drift_xy"][:, 1],
                "base_z":    sb["base_pos_world"][:, 2],
                "term_cause": sb["term_cause"],
                "done":      done,
                "active":    ~ep_done,
            }
            return (new_state, new_done), per_step

        key = jax.random.PRNGKey(args.seed)
        key, reset_key = jax.random.split(key)
        state = reset_batch(reset_key)
        init_done = jnp.zeros(n_eps, dtype=jnp.bool_)
        step_keys = jax.random.split(key, ep_len)
        (final_state, _), data = jax.lax.scan(scan_step, (state, init_done), step_keys)

        # Move to numpy
        d = {k: np.asarray(v) for k, v in data.items()}
        return d

    print(f"\nPhysics sweep on {args.base_env}")
    print(f"  episodes/pair: {args.num_episodes}, episode_length: {args.episode_length}")
    print(f"  Train range: vL∈[0.3,1.5], ratio∈[0.5,2.0]\n")
    print(f"  (vL,vR)        ratio   | term%  fall  off  tilt  cross | "
          f"surv_steps  finalDrx  maxDrx  maxDry")
    print("-" * 100)

    for spec in args.pairs:
        vL, vR = (float(x) for x in spec.split(","))
        env = make_env("split_constant", {"vL": vL, "vR": vR}, args.base_env)
        d = eval_pair(env, args.num_episodes, args.episode_length)

        # Active mask (episode still alive at step t).  Shape (T, B).
        active = d["active"]
        T, B = active.shape

        # Survival steps: count of active steps per episode (0 to T).
        survival = active.sum(axis=0)

        # Termination cause: take the cause at the FIRST done step (or 0 if survived).
        done = d["done"]
        any_done = done.any(axis=0)
        first_done_idx = np.argmax(done, axis=0)  # 0 if no done
        cause_at_term = d["term_cause"][first_done_idx, np.arange(B)]
        cause_at_term = np.where(any_done, cause_at_term, 0)

        n_fall  = int((cause_at_term == 1).sum())
        n_off   = int((cause_at_term == 2).sum())
        n_tilt  = int((cause_at_term == 3).sum())
        n_cross = int((cause_at_term == 4).sum())
        n_term  = int(any_done.sum())
        term_pct = 100.0 * n_term / B

        # Drift stats: mask inactive steps with NaN, then take per-episode max/last.
        dx_mask = np.where(active, d["drift_x"], np.nan)
        dy_mask = np.where(active, d["drift_y"], np.nan)
        with np.errstate(invalid="ignore"):
            max_abs_dx = np.nanmax(np.abs(dx_mask), axis=0)
            max_abs_dy = np.nanmax(np.abs(dy_mask), axis=0)
        # Final drift: last active value per episode.
        final_dx = np.array([
            float(dx_mask[max(0, int(survival[i]) - 1), i])
            for i in range(B)
        ])

        ratio = vR / vL if vL > 0 else float("inf")
        print(f"  ({vL:.2f}, {vR:.2f})   {ratio:>5.2f}x | "
              f"{term_pct:>4.0f}%  {n_fall:>3d}  {n_off:>3d}  {n_tilt:>3d}  {n_cross:>3d}  | "
              f"{survival.mean():>9.0f}  {final_dx.mean():>+8.2f}  "
              f"{np.nanmean(max_abs_dx):>6.2f}  {np.nanmean(max_abs_dy):>6.3f}")


if __name__ == "__main__":
    main()
