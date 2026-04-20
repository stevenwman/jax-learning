"""Visually calibrate the coverage metric.

Directly places the T at perturbed poses from the "identity" state
(block.position where coverage is maximal), measures coverage, renders
a labeled grid. So we can see what 95/90/85/80/75/70% coverage looks
like visually.

Note: pymunk body.center_of_gravity for the T is offset (0, 45) from
body origin. Setting block.position = (256, 256) does NOT put the T
at the goal pose. The "identity" state must be discovered empirically —
we grid-search for it below.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from jax_rl.envs.manipulation.pusht import PushTEnv


GOAL_YAW = np.pi / 4


def measure(env, block_x, block_y, block_yaw):
    state = np.array([50.0, 50.0, block_x, block_y, block_yaw], dtype=np.float32)
    env.reset(seed=0, options={"reset_to_state": state.tolist()})
    return float(env._get_coverage()), env.render()


def find_identity(env):
    """Grid search (block_x, block_y) at fixed goal_yaw to find max-coverage state."""
    best = (0, 0, 0)
    # Coarse then fine. Top bar of T is ~120px, fine gradient.
    for dx_coarse in np.linspace(-50, 50, 21):
        for dy_coarse in np.linspace(-50, 50, 21):
            c, _ = measure(env, 256 + dx_coarse, 256 + dy_coarse, GOAL_YAW)
            if c > best[2]:
                best = (dx_coarse, dy_coarse, c)
    # Fine refinement around best coarse
    cx, cy, _ = best
    for dx in np.linspace(cx - 3, cx + 3, 31):
        for dy in np.linspace(cy - 3, cy + 3, 31):
            c, _ = measure(env, 256 + dx, 256 + dy, GOAL_YAW)
            if c > best[2]:
                best = (dx, dy, c)
    return 256 + best[0], 256 + best[1], best[2]


def find_perturbation_for_target(env, ident_x, ident_y, target_cov, axis, max_iter=40):
    def cov_at(mag):
        if axis == "x":
            return measure(env, ident_x + mag, ident_y, GOAL_YAW)[0]
        elif axis == "y":
            return measure(env, ident_x, ident_y + mag, GOAL_YAW)[0]
        elif axis == "yaw":
            return measure(env, ident_x, ident_y, GOAL_YAW + mag)[0]
        elif axis == "diag":
            return measure(env, ident_x + mag/np.sqrt(2), ident_y + mag/np.sqrt(2), GOAL_YAW)[0]
    lo, hi = 0.0, 2.0 if axis == "yaw" else 150.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        c = cov_at(mid)
        if c > target_cov:
            lo = mid
        else:
            hi = mid
        if abs(c - target_cov) < 0.002:
            break
    return mid, c


def label(img, text):
    pil = Image.fromarray(img).copy()
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 38], fill=(0, 0, 0, 220))
    draw.text((10, 4), text, fill=(255, 255, 255))
    return np.asarray(pil)


def main():
    out = Path(__file__).parent.parent / ".temp"
    out.mkdir(exist_ok=True)
    env = PushTEnv(obs_type="state", render_mode="rgb_array")

    print("Searching for identity state (max coverage)...")
    ident_x, ident_y, ident_cov = find_identity(env)
    print(f"  identity block-state: ({ident_x:.1f}, {ident_y:.1f}, {GOAL_YAW:.4f})  cov={ident_cov:.4f}")

    target_covs = [ident_cov, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    print(f"\n{'axis':>6} {'target':>8} {'mag':>10} {'actual_cov':>12}")
    rows_by_axis = {}
    for axis_name, axis in [("x (px)", "x"), ("diag (px)", "diag"), ("yaw (°)", "yaw")]:
        images = []
        for tc in target_covs:
            if abs(tc - ident_cov) < 0.001:
                cov, frame = measure(env, ident_x, ident_y, GOAL_YAW)
                mag_disp = "identity"
                lbl = f"cov={cov:.3f} (ident.)"
            else:
                mag, cov = find_perturbation_for_target(env, ident_x, ident_y, tc, axis)
                if axis == "x":
                    cov, frame = measure(env, ident_x + mag, ident_y, GOAL_YAW)
                    mag_disp = f"{mag:+.1f}px"
                elif axis == "diag":
                    cov, frame = measure(env, ident_x + mag/np.sqrt(2), ident_y + mag/np.sqrt(2), GOAL_YAW)
                    mag_disp = f"{mag:+.1f}px"
                elif axis == "yaw":
                    cov, frame = measure(env, ident_x, ident_y, GOAL_YAW + mag)
                    mag_disp = f"{np.degrees(mag):+.1f}°"
                lbl = f"cov={cov:.3f} ({mag_disp})"
            images.append(label(frame, lbl))
            print(f"{axis_name:>6} {tc:>8.3f} {mag_disp:>10} {cov:>12.4f}")
        rows_by_axis[axis_name] = images

    # Save individual per-cell images (viewable in .temp)
    for axis_name, imgs in rows_by_axis.items():
        axis_tag = axis_name.split()[0]
        for tc, img in zip(target_covs, imgs):
            fname = f"pusht_cov_{axis_tag}_{int(tc*100):02d}.png"
            Image.fromarray(img).save(out / fname)
    # Per-axis row mosaics
    for axis_name, imgs in rows_by_axis.items():
        axis_tag = axis_name.split()[0]
        row = np.concatenate(imgs, axis=1)
        Image.fromarray(row).save(out / f"pusht_cov_row_{axis_tag}.png")
    # Full grid
    rows = []
    for axis_name, imgs in rows_by_axis.items():
        row = np.concatenate(imgs, axis=1)
        pil = Image.fromarray(row)
        padded = ImageOps.expand(pil, border=(160, 0, 0, 0), fill=(20, 20, 20))
        draw = ImageDraw.Draw(padded)
        draw.text((5, padded.height // 2 - 10), axis_name, fill=(255, 255, 255))
        rows.append(np.asarray(padded))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "pusht_coverage_calibration.png")
    print(f"\nsaved grid + per-cell images to {out}/")


if __name__ == "__main__":
    main()
