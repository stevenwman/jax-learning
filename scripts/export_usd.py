"""Export a saved trajectory to USD for offline rendering in Blender / Omniverse.

Phase 1 — sanity check. Loads the env's mj_model and replays a `_traj.npz`
qpos sequence through `mujoco.usd.exporter.USDExporter`, writing a USD
file you can open in usdview / Blender / Omniverse.

Usage:
  uv run python scripts/export_usd.py \\
      --traj projects/adaptation/videos/posedr_differential_2026_05_07_HD/vL0.50_vR0.20_ratio0.40x_absOOD_ratioOOD_traj.npz \\
      --env Go2WarpSplitbeltPoseDR \\
      --out projects/adaptation/videos/usd_exports/posedr_vL0.5_vR0.2

If your traj is a splitbelt rollout, the belt slide-joint qpos entries are
included automatically (qpos shape matches model.nq).

Smoke check after export:
  usdcat <out_dir>/scene.usd | head -50      # text dump
  usdchecker <out_dir>/scene.usd             # structural validity
  usdview <out_dir>/scene.usd                # interactive viewer
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import mujoco
from mujoco.usd import exporter as usd_exporter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True,
                    help="Path to a `_traj.npz` file containing 'qpos' (T, nq).")
    ap.add_argument("--env", required=True,
                    help="Env name to look up XML (e.g. Go2WarpSplitbeltPoseDR, "
                         "G1WarpJoystickHoloClearance).")
    ap.add_argument("--out", required=True,
                    help="Output directory for USD + assets (will be created).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap number of frames written (default: full trajectory).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Sample every Nth frame (default 1).")
    ap.add_argument("--height", type=int, default=480,
                    help="USD camera default render height (per-frame poses unaffected).")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    # ── Load env to get mj_model ──────────────────────────────────────────
    import jax_rl.training.env_setup  # noqa: F401 — registers custom envs
    from mujoco_playground import registry as pg_registry
    from jax_rl.training.env_backends.mjx_backend import maybe_load_custom_env

    env = maybe_load_custom_env(args.env)
    if env is None:
        env = pg_registry.load(args.env)
    # Unwrap to base (wrappers may not have mj_model directly).
    base = env
    while hasattr(base, "env"):
        base = base.env
    mj_model = base.mj_model
    print(f"[env] {args.env}  nq={mj_model.nq}  nv={mj_model.nv}  nbody={mj_model.nbody}")

    # ── Load trajectory ───────────────────────────────────────────────────
    traj = np.load(args.traj)
    if "qpos" not in traj.files:
        raise SystemExit(f"--traj must have 'qpos' array; got {traj.files}")
    qpos_seq = np.asarray(traj["qpos"])
    qvel_seq = np.asarray(traj["qvel"]) if "qvel" in traj.files else None
    T = qpos_seq.shape[0]
    if qpos_seq.shape[1] != mj_model.nq:
        raise SystemExit(
            f"qpos shape {qpos_seq.shape} mismatches model.nq={mj_model.nq}. "
            f"Check --env matches the env that produced the trajectory."
        )
    if args.max_frames is not None:
        T = min(T, args.max_frames)
    indices = list(range(0, T, args.stride))
    print(f"[traj] T={qpos_seq.shape[0]}  → exporting {len(indices)} frames "
          f"(stride={args.stride}, max={args.max_frames})")

    # ── Set up USD exporter ───────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    out_root, out_dir = os.path.split(os.path.abspath(args.out.rstrip("/")))
    exp = usd_exporter.USDExporter(
        model=mj_model,
        height=args.height,
        width=args.width,
        output_directory=out_dir,
        output_directory_root=out_root,
        verbose=False,
    )

    # ── Replay loop ───────────────────────────────────────────────────────
    mj_data = mujoco.MjData(mj_model)
    for t in indices:
        mj_data.qpos[:] = qpos_seq[t]
        if qvel_seq is not None and qvel_seq.shape[1] == mj_model.nv:
            mj_data.qvel[:] = qvel_seq[t]
        mujoco.mj_forward(mj_model, mj_data)
        exp.update_scene(data=mj_data)
        if t % 100 == 0:
            print(f"  frame {t}/{T}")

    exp.save_scene(filetype="usd")
    print(f"\n[done] USD written to {args.out}/")
    print(f"  Sanity check:  usdchecker {args.out}/scene.usd")
    print(f"  Inspect:       usdview {args.out}/scene.usd")


if __name__ == "__main__":
    main()
