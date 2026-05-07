"""Record a splitbelt rollout at a fixed belt speed (override env schedule).

Use to demo a trained PoseDR ckpt at a clean in-dist speed (e.g. v=0.5 where
the policy stationkeeps reliably).

Usage:
  MUJOCO_GL=egl uv run python scripts/record_splitbelt_at_speed.py \\
      --checkpoint checkpoints/.../best --speed 0.5 --max-steps 1250
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools

# Pre-register a tied-speed PoseDR variant BEFORE importing record_video
# (which loads via registry).
def _register_at_speed(speed: float):
    from mujoco_playground import locomotion as pg_locomotion
    from mujoco_playground import registry as pg_registry  # noqa: F401
    from ml_collections import config_dict
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv

    # Trigger the standard PoseDR registration first.
    import jax_rl.training.env_setup  # noqa: F401

    name = f"Go2WarpSplitbeltPoseDR_v{speed:g}"
    if name in pg_locomotion._envs:
        return name

    def cfg_factory():
        # Reuse PoseDR default config from the playground cfg registry.
        cfg_fn = pg_locomotion._cfgs["Go2WarpSplitbeltPoseDR"]
        cfg = cfg_fn()
        cfg.unlock()
        cfg.schedule_kind = "tied"
        cfg.schedule_params = config_dict.create(v=float(speed))
        return cfg

    pg_locomotion.register_environment(
        name,
        functools.partial(Go2WarpSplitbeltEnv, task="splitbelt_pose_dr"),
        cfg_factory,
    )
    # Patch record_video's ENV_DEFAULTS so the iso side cam is picked.
    from scripts import record_video
    record_video.ENV_DEFAULTS[name] = ((640, 480), "splitbelt_side_iso")
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=1250)
    p.add_argument("--camera", default=None,
                   help="default: splitbelt_side_iso")
    args = p.parse_args()

    env_name = _register_at_speed(args.speed)
    print(f"Registered: {env_name}")

    from scripts.record_video import record
    record(
        env_name=env_name,
        checkpoint=args.checkpoint,
        max_steps=args.max_steps,
        camera=args.camera,
        no_early_term=True,
    )


if __name__ == "__main__":
    main()
