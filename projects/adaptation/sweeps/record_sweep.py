"""Record a folder of videos at multiple (vL, vR) speed pairs.

For each pair, registers a one-off Go2WarpSplitbeltPoseDR variant with
split_constant(vL, vR) schedule and runs record_video.record() against it.
Output filename encodes vL, vR, ratio, and OOD class.

Usage:
  MUJOCO_GL=egl uv run python scripts/record_splitbelt_sweep.py \\
      --checkpoint checkpoints/.../best \\
      --pairs 0.5,0.5 0.5,1.0 0.5,1.5 0.3,1.5 0.3,0.9 1.0,0.3 0.5,0.2 \\
      --out-dir videos/posedr_differential_2026_05_07
"""

from __future__ import annotations

import argparse
import functools
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _register_pair(vL: float, vR: float, base_env: str = "Go2WarpSplitbeltPoseDR"):
    from mujoco_playground import locomotion as pg_locomotion
    from ml_collections import config_dict
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    import jax_rl.training.env_setup  # noqa: F401

    name = f"{base_env}_split_{vL:g}_{vR:g}"
    if name in pg_locomotion._envs:
        return name

    def cfg_factory():
        cfg = pg_locomotion._cfgs[base_env]()
        cfg.unlock()
        cfg.schedule_kind = "split_constant"
        cfg.schedule_params = config_dict.create(vL=float(vL), vR=float(vR))
        return cfg

    task_map = {
        "Go2WarpSplitbeltPoseDR":         "splitbelt_pose_dr",
        "Go2WarpSplitbeltPosTrack":       "splitbelt_pos_track",
        "Go2WarpSplitbeltPosTrackTiedDR": "splitbelt_pos_track_tied_dr",
    }
    task = task_map.get(base_env, "splitbelt_pos_track")
    pg_locomotion.register_environment(
        name,
        functools.partial(Go2WarpSplitbeltEnv, task=task),
        cfg_factory,
    )
    from scripts import record_video
    record_video.ENV_DEFAULTS[name] = ((1280, 720), "splitbelt_side_iso")
    return name


def _classify(vL: float, vR: float) -> str:
    """Return short tag describing OOD type for filename."""
    abs_ood = (vL < 0.3 or vL > 1.5) or (vR < 0.3 or vR > 1.5)
    ratio = vR / vL if vL > 0 else float("inf")
    ratio_ood = ratio < 0.5 or ratio > 2.0
    if vL == vR:
        return "tied"
    parts = []
    if abs_ood:   parts.append("absOOD")
    if ratio_ood: parts.append("ratioOOD")
    if not parts: parts.append("inDist")
    return "_".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="vL,vR space-separated (e.g. 0.5,1.5 0.3,1.5)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-steps", type=int, default=1250)
    p.add_argument("--camera", default=None,
                   help="default: splitbelt_side_iso")
    p.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                   metavar=("W", "H"),
                   help="Render resolution. HD=1280 720, FullHD=1920 1080.")
    p.add_argument("--video-quality", type=int, default=8,
                   help="imageio quality 1-10 (default 8).")
    p.add_argument("--base-env", default="Go2WarpSplitbeltPoseDR",
                   help="Splitbelt env to clone for each pair.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from scripts.record_video import record

    print(f"Output: {args.out_dir}/\n")
    for spec in args.pairs:
        vL, vR = (float(x) for x in spec.split(","))
        env_name = _register_pair(vL, vR, args.base_env)
        ratio = vR / vL if vL > 0 else float("inf")
        tag = _classify(vL, vR)
        fname = f"vL{vL:.2f}_vR{vR:.2f}_ratio{ratio:.2f}x_{tag}.mp4"
        out_path = os.path.join(args.out_dir, fname)
        print(f"\n=== Recording ({vL}, {vR}) ratio={ratio:.2f}x  [{tag}] ===")
        record(
            env_name=env_name,
            checkpoint=args.checkpoint,
            out=out_path,
            max_steps=args.max_steps,
            camera=args.camera,
            no_early_term=True,
            resolution=tuple(args.resolution),
            video_quality=args.video_quality,
        )
        print(f"  → {out_path}")

    print(f"\nDone. {len(args.pairs)} videos in {args.out_dir}/")


if __name__ == "__main__":
    main()
