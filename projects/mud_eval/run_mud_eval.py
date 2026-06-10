"""M1 eval — run a jax-learning FastSAC Go2 policy in the Newton triple-mud sim.

Swaps the example's torch `Go2Policy` for `MudJaxPolicy` (real jax_rl actor + a
Newton-state obs adapter), forces a forward command, steps headless, and reports
whether the robot walks (torso displacement) + stays finite.

Run (from projects/mud_eval/, PYTHONPATH=<jax-learning>):
    PYTHONPATH=/home/stevenman/Desktop/Work/Research/jax-learning \
      .venv/bin/python run_mud_eval.py <ckpt_dir> [num_frames] [voxel] [mpm_iters]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))   # vendored newton

import numpy as np                # noqa: E402
import warp as wp                 # noqa: E402
import torch                      # noqa: E402  (vendored scene-build interop)
import newton.examples            # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config   # noqa: E402


def main(ckpt, num_frames=60, voxel_size=0.05, mpm_iters=10, command="fwd"):
    ex.Go2Policy = MudJaxPolicy            # drop-in: real jax_rl policy + obs adapter
    cfg = patched_config(ckpt, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                         "/tmp/mud_cfg_patched.yaml")   # spawn @ policy default pose + Kp/Kd

    sys.argv = [
        "run_mud_eval", "--viewer", "null", "--num-frames", str(num_frames),
        "--policy-path", str(ckpt), "--config", cfg,
        "--voxel-size", str(voxel_size), "--max-iterations", str(mpm_iters),
    ]
    parser = newton.examples.create_parser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--voxel-size", "-dx", type=float, default=None)
    parser.add_argument("--max-iterations", "-it", type=int, default=None)
    parser.add_argument("--tolerance", "-tol", type=float, default=None)
    parser.add_argument("--policy-path", "-cp", type=str, default=None)
    parser.add_argument("--precompute-frames", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--debug-forces", action="store_true")
    parser.add_argument("--plot-actions", type=str, default=None)
    parser.add_argument("--plot-forces", type=str, default=None)
    parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
    parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
    viewer, args = newton.examples.init(parser)

    example = ex.Example(viewer, args)
    fwd = (command == "fwd")
    example._auto_forward = fwd            # fwd: command=[1,0,0]; else [0,0,0]
    if command == "hold":
        example.policy.hold = True        # zero policy action — isolate spawn/contact
    print(f"[M1] policy={Path(ckpt).name} command={command}", flush=True)
    # where is the robot vs the mud? (free-fall suspicion: robot spawns off the collider)
    pq = np.asarray(example.state_0.particle_q.numpy())
    rb = np.asarray(example.state_0.body_q.numpy())[0]
    print(f"[M1] robot xyz={rb[:3].round(2)} | mud x[{pq[:,0].min():.1f},{pq[:,0].max():.1f}] "
          f"y[{pq[:,1].min():.1f},{pq[:,1].max():.1f}] z[{pq[:,2].min():.2f},{pq[:,2].max():.2f}]",
          flush=True)

    def torso():
        return np.asarray(example.state_0.body_q.numpy())[0]
    x0, z0 = float(torso()[0]), float(torso()[2])
    for f in range(num_frames):           # manual loop (vs newton.examples.run) for per-frame logs
        example.step()
        if f % 4 == 0 or f == num_frames - 1:
            t = torso(); an = float(np.linalg.norm(example.policy.last_act))
            print(f"  f{f:3d} z={t[2]:+.3f} x={t[0]:+.3f} |act|={an:.2f}", flush=True)
            if not np.isfinite(t).all():
                print("  non-finite — stop"); break
    example.viewer.close()
    bq = np.asarray(example.state_0.body_q.numpy())
    x1, z1 = float(bq[0][0]), float(bq[0][2])
    finite = np.isfinite(bq).all()
    fell = z1 < 0.18
    print(f"[M1] torso: x {x0:.3f}->{x1:.3f} (Δx={x1-x0:+.3f})  z {z0:.3f}->{z1:.3f}  "
          f"finite={finite} fell={fell}")
    walked = finite and (not fell) and (x1 - x0) > 0.1
    print("[M1] PASS — policy ran, robot upright + moved forward" if walked else
          "[M1] ran (inspect: %s)" % ("fell/non-finite" if (fell or not finite) else "little forward motion"))
    sys.exit(0 if finite else 1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print("usage: run_mud_eval.py <ckpt_dir> [num_frames] [voxel] [mpm_iters]")
        sys.exit(2)
    main(a[0],
         num_frames=int(a[1]) if len(a) > 1 else 60,
         voxel_size=float(a[2]) if len(a) > 2 else 0.05,
         mpm_iters=int(a[3]) if len(a) > 3 else 10,
         command=a[4] if len(a) > 4 else "fwd")
