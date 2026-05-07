"""A1-protocol probe: roll out a trained ckpt under tied_split_tied schedule,
save per-step trajectory + sidecar, run offline `splitbelt_analysis`.

Default schedule: tied(0.5) for first 200 steps, split(vL=0.5, vR=1.0) for
600 steps, tied(0.5) for the remainder. Total episode 1500 steps.

Reports step-length asymmetry per phase + recovery-time + after-effect:
- Asymmetry baseline (Phase 1)
- Asymmetry start of split (early Phase 2)
- Asymmetry end of split (late Phase 2)         ← adaptation
- Asymmetry early Phase 3                       ← after-effect
- Asymmetry late Phase 3                        ← recovery

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/eval_splitbelt_a1.py \\
      --checkpoint checkpoints/.../best \\
      --episode-length 1500 --t1 200 --t2 600 \\
      --vL-split 0.5 --vR-split 1.0 --v-warm 0.5
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
from jax_rl.envs.locomotion import splitbelt_analysis as analysis

from scripts.record_video import _build_select_action  # type: ignore


def make_a1_env(v_warm, vL_split, vR_split, t1, t2,
                base_env_name="Go2WarpSplitbeltPoseDR"):
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from mujoco_playground import registry as pg_registry

    template = pg_registry.load(base_env_name)
    cfg = config_dict.ConfigDict(template._config.to_dict())
    cfg.unlock()
    cfg.schedule_kind = "tied_split_tied"
    cfg.schedule_params = config_dict.create(
        v_warm=float(v_warm),
        vL_split=float(vL_split),
        vR_split=float(vR_split),
        t1=int(t1),
        t2=int(t2),
    )
    task_map = {
        "Go2WarpSplitbelt":       "splitbelt",
        "Go2WarpSplitbeltDR":     "splitbelt_dr",
        "Go2WarpSplitbeltPoseDR": "splitbelt_pose_dr",
    }
    return Go2WarpSplitbeltEnv(task=task_map[base_env_name], config=cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode-length", type=int, default=1500)
    p.add_argument("--v-warm", type=float, default=0.5)
    p.add_argument("--vL-split", type=float, default=0.5)
    p.add_argument("--vR-split", type=float, default=1.0)
    p.add_argument("--t1", type=int, default=200)
    p.add_argument("--t2", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-env", default="Go2WarpSplitbeltPoseDR")
    p.add_argument("--out-prefix", default=None,
                   help="If set, save sidecar to <prefix>_splitbelt_traj.npz "
                        "(default: <ckpt>/a1_<timestamp>)")
    args = p.parse_args()

    meta, actor_params, norm_state, batch_stats = load_actor_for_inference(args.checkpoint)
    print(f"Loaded {meta.get('algo')} from {args.checkpoint}")

    env = make_a1_env(args.v_warm, args.vL_split, args.vR_split, args.t1, args.t2,
                      args.base_env)
    state = env.reset(jax.random.PRNGKey(args.seed))
    obs0 = state.obs["state"] if isinstance(state.obs, dict) else state.obs
    obs_dim, action_dim = int(obs0.shape[-1]), int(env.action_size)

    algo, _ = _build_select_action(meta, obs_dim, action_dim)
    key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    training_state = algo.init(init_key)
    if actor_params is not None:
        training_state = training_state.replace(actor_params=actor_params)
    if batch_stats is not None and hasattr(training_state, "actor_batch_stats"):
        training_state = training_state.replace(actor_batch_stats=batch_stats)
        algo._default_actor_bs = batch_stats

    use_obs_norm = bool(meta.get("fast_sac_config", {}).get("obs_normalization", False))

    @jax.jit
    def step_fn(state, key):
        obs = state.obs
        flat = obs["state"] if isinstance(obs, dict) else obs
        if use_obs_norm and norm_state is not None:
            flat = norm_normalize(flat, norm_state)
        action = algo.select_action(actor_params, flat, key, deterministic=True)
        next_state = env.step(state, action)
        return next_state, action

    # Per-step capture
    foot_pos_world = np.zeros((args.episode_length, 4, 3), dtype=np.float32)
    foot_in_contact = np.zeros((args.episode_length, 4), dtype=bool)
    belt_vel = np.zeros((args.episode_length, 2), dtype=np.float32)
    base_pos = np.zeros((args.episode_length, 3), dtype=np.float32)
    rewards = np.zeros(args.episode_length, dtype=np.float32)
    dones = np.zeros(args.episode_length, dtype=bool)

    print(f"Rolling out {args.episode_length} steps "
          f"[warm 0..{args.t1}, split {args.t1}..{args.t1+args.t2}, "
          f"recovery {args.t1+args.t2}..{args.episode_length}]")
    for t in range(args.episode_length):
        key, sub = jax.random.split(key)
        state, action = step_fn(state, sub)
        sb = state.info["splitbelt"]
        foot_pos_world[t] = np.asarray(sb["foot_pos_world"])
        foot_in_contact[t] = np.asarray(sb["foot_in_contact"])
        belt_vel[t] = np.asarray(sb["belt_vel"])
        base_pos[t] = np.asarray(sb["base_pos_world"])
        rewards[t] = float(state.reward)
        dones[t] = bool(state.done)
        if dones[t]:
            print(f"  Early termination at t={t} (term_cause={int(sb.get('term_cause', -1))})")
            break

    last_t = int(np.argmax(dones)) if dones.any() else args.episode_length
    print(f"  Total reward: {rewards[:last_t].sum():.1f}, ran {last_t} steps")

    events = analysis.detect_step_events(foot_in_contact[:last_t])

    # NOTE: step_length_asymmetry from splitbelt_analysis assumes a forward
    # walking gait. PoseDR is stationary (cmd=0); step lengths are dominated
    # by belt-drift, not voluntary stride. We report touchdown rates and
    # body-x drift instead — these have actual signal in a stationkeeping
    # quadruped under belt drag.

    phases = {
        "Phase 1 (warm tied 0..t1)":         (0, args.t1),
        "Phase 2 early (split start)":       (args.t1, args.t1 + args.t2 // 4),
        "Phase 2 late  (split end)":         (args.t1 + 3 * args.t2 // 4, args.t1 + args.t2),
        "Phase 3 early (recovery start)":    (args.t1 + args.t2, args.t1 + args.t2 + args.t2 // 4),
        "Phase 3 late  (recovery end)":      (max(0, last_t - args.t2 // 4), last_t),
    }

    POLICY_DT = 0.02  # 4 substep × 0.005s timestep — splitbelt default
    print(f"\nPer-phase metrics (PoseDR is stationary; touchdown rate proxies stride freq):")
    print(f"{'Phase':<42s}  {'L_rate':>7s} {'R_rate':>7s}  {'asym':>7s}  "
          f"{'Δx body (m)':>14s}  {'reward':>8s}")
    for label, (s, e) in phases.items():
        if e <= s:
            continue
        n_left = sum(1 for f in (0, 2) for t in events["touchdown_steps"][f] if s <= t < e)
        n_right = sum(1 for f in (1, 3) for t in events["touchdown_steps"][f] if s <= t < e)
        dt_phase = (e - s) * POLICY_DT
        l_rate = n_left / dt_phase
        r_rate = n_right / dt_phase
        denom = (l_rate + r_rate)
        rate_asym = (r_rate - l_rate) / denom if denom > 0 else 0.0
        x0 = base_pos[s, 0] if s < last_t else float("nan")
        x1 = base_pos[min(e, last_t) - 1, 0]
        dx = float(x1 - x0)
        rwd = float(rewards[s:e].sum())
        print(f"  {label:40s}  {l_rate:>7.2f} {r_rate:>7.2f}  {rate_asym:>+7.3f}  "
              f"{dx:>+14.3f}  {rwd:>8.1f}")

    # Save sidecar
    out_prefix = args.out_prefix
    if out_prefix is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_prefix = f"{args.checkpoint.rstrip('/')}/a1_{ts}"
    np.savez_compressed(
        f"{out_prefix}_splitbelt_traj.npz",
        foot_pos_world=foot_pos_world[:last_t],
        foot_in_contact=foot_in_contact[:last_t],
        belt_vel=belt_vel[:last_t],
        base_pos=base_pos[:last_t],
        rewards=rewards[:last_t],
        dones=dones[:last_t],
        t1=args.t1, t2=args.t2,
        v_warm=args.v_warm, vL_split=args.vL_split, vR_split=args.vR_split,
    )
    print(f"\nSidecar: {out_prefix}_splitbelt_traj.npz")


if __name__ == "__main__":
    main()
