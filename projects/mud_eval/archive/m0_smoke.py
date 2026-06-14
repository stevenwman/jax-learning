"""M0 physics smoke — confirm the Newton triple-mud Go2 scene runs headless in
our dedicated venv, before we own the control loop.

Builds the vendored Example (robot + thin/medium/thick MPM mud + two-way
coupling) and steps it headless with a HOLD-POSE stub controller (no trained
policy, no .pt checkpoint), then lets newton's `--test` runner assert state
finiteness. This isolates "does the physics run here?" from any policy work.

Run (from projects/mud_eval/):
    .venv/bin/python m0_smoke.py            # default fast smoke
    .venv/bin/python m0_smoke.py 20 0.04 10 # frames, voxel_size, mpm_iters
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))   # vendored newton (self-sufficient)

import warp as wp           # noqa: E402
import torch                # noqa: E402  (vendored scene-build uses torch interop)
import newton.examples      # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402


class _HoldPolicy:
    """Stub replacing the torch RSL-RL Go2Policy: holds the initial joint pose
    (zero action) so __init__ needs no checkpoint / MLP — just the physics."""

    def __init__(self, checkpoint_path, device, joint_pos_initial,
                 action_scale=0.25, search_relative_to=None, **kw):
        self.jpi = joint_pos_initial                  # (1, 12) torch, initial pose
        self.last_action = torch.zeros_like(joint_pos_initial)
        self._pad = torch.zeros(6, device=device)     # free-joint padding

    @torch.no_grad()
    def compute_joint_targets(self, state, command):
        padded = torch.cat([self._pad, self.jpi.squeeze(0)])   # hold initial pose
        return wp.from_torch(padded, dtype=wp.float32, requires_grad=False)


def main(num_frames=15, voxel_size=0.04, mpm_iters=12):
    # Swap the torch checkpoint policy for the hold-pose stub.
    ex.Go2Policy = _HoldPolicy

    # NOTE: no --test — the Example registers a *forward-motion* test
    # ("robot went in the right direction", q[1]>0.9 over 100 frames) that a
    # hold-pose stub over a few frames can't satisfy. We do our own finiteness
    # check below instead.
    sys.argv = [
        "m0_smoke", "--viewer", "null", "--num-frames", str(num_frames),
        "--voxel-size", str(voxel_size), "--max-iterations", str(mpm_iters),
    ]
    parser = newton.examples.create_parser()
    # Re-add the example's own args (defined only in its __main__).
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
    parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"],
                        default="magnitude")
    viewer, args = newton.examples.init(parser)

    example = ex.Example(viewer, args)
    print(f"[M0] scene built | particles={example.model.particle_count} "
          f"bodies={example.model.body_count} dofs={example.model.joint_dof_count}",
          flush=True)
    newton.examples.run(example, args)   # steps num_frames headless (viewer=null)

    import numpy as np
    body_q = np.asarray(example.state_0.body_q.numpy())
    part_q = np.asarray(example.state_0.particle_q.numpy())
    finite = np.isfinite(body_q).all() and np.isfinite(part_q).all()
    z = float(body_q[0][2])              # torso z after the run
    print(f"[M0] body_q finite={np.isfinite(body_q).all()} "
          f"particle_q finite={np.isfinite(part_q).all()} | torso_z={z:.3f}")
    print("[M0] PASS — physics stepped %d frames headless, all finite" % num_frames
          if finite else "[M0] FAIL — non-finite state")
    sys.exit(0 if finite else 1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(
        num_frames=int(a[0]) if len(a) > 0 else 15,
        voxel_size=float(a[1]) if len(a) > 1 else 0.04,
        mpm_iters=int(a[2]) if len(a) > 2 else 12,
    )
