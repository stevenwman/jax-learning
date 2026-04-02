"""CMA-ES optimization for Go2 handstand initial pose on flat ground.

Finds the joint angles + base orientation that maximize time-before-fall
with zero actuation. Uses CPU MuJoCo with multiprocessing for parallel eval.

Usage:
    uv run python tools/handstand_optimize.py
    uv run python tools/handstand_optimize.py --generations 100 --render
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

# Ensure project root is on sys.path (for background/remote execution).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cma
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


FLAT_SCENE_XML = """
<mujoco model="go2 handstand optimization">
  <include file="unitree_go2/go2.xml"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".3 .3 .3" specular="1 1 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" markrgb="0 0 0"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5"/>
  </asset>

  <worldbody>
    <geom name="floor" size="5 5 0.01" type="plane" material="groundplane"
      friction="0.8 0.005 0.001" contype="1" conaffinity="1"/>
  </worldbody>

  <sensor>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
  </sensor>
</mujoco>
"""

# Module-level model for multiprocessing workers (loaded once per worker).
_worker_model = None


def _init_worker():
    global _worker_model
    from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
    assets = get_warp_assets()
    _worker_model = mujoco.MjModel.from_xml_string(FLAT_SCENE_XML, assets=assets)
    for i in range(_worker_model.nu):
        _worker_model.actuator_forcerange[i] = _worker_model.actuator_ctrlrange[i]


def load_model():
    from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
    assets = get_warp_assets()
    mj_model = mujoco.MjModel.from_xml_string(FLAT_SCENE_XML, assets=assets)
    for i in range(mj_model.nu):
        mj_model.actuator_forcerange[i] = mj_model.actuator_ctrlrange[i]
    return mj_model


def params_to_qpos(params, nq):
    """Convert CMA-ES parameter vector to full qpos (numpy).

    Parameters (7 total, left-right symmetric, base_z computed from feet):
        params[0]: extra pitch beyond 90deg (radians, how far past vertical)
        params[1:4]: front leg joints (hip, thigh, calf) — same for FL and FR
        params[4:7]: rear leg joints (hip, thigh, calf) — same for RL and RR
    """
    extra_pitch = params[0]
    front_hip, front_thigh, front_calf = params[1], params[2], params[3]
    rear_hip, rear_thigh, rear_calf = params[4], params[5], params[6]

    # Y=90 (pitch into handstand) + Z=90 (yaw, fixed — irrelevant on flat ground)
    rot = R.from_euler(
        'yz',
        [90 + np.degrees(extra_pitch), 90],
        degrees=True,
    )
    q = rot.as_quat()  # xyzw
    quat_wxyz = [q[3], q[0], q[1], q[2]]

    qpos = np.zeros(nq)
    qpos[0] = 0.0
    qpos[1] = 0.0
    qpos[2] = 0.5  # temporary — will be adjusted
    qpos[3:7] = quat_wxyz

    # Left-right symmetric: hip mirrored (both legs splay outward), thigh/calf shared
    qpos[7:10] = [front_hip, front_thigh, front_calf]     # FL
    qpos[10:13] = [-front_hip, front_thigh, front_calf]   # FR (hip mirrored)
    qpos[13:16] = [rear_hip, rear_thigh, rear_calf]       # RL
    qpos[16:19] = [-rear_hip, rear_thigh, rear_calf]      # RR (hip mirrored)

    return qpos


def _solve_base_z(qpos, model):
    """Compute base_z so front feet sit on the ground (z=0),
    then lift if any non-foot body part would contact the ground.
    """
    data = mujoco.MjData(model)

    # Step 1: put feet on ground
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    fl_z = data.site_xpos[model.site("FL_foot").id][2]
    fr_z = data.site_xpos[model.site("FR_foot").id][2]
    feet_z = (fl_z + fr_z) / 2.0
    qpos[2] -= feet_z

    # Step 2: check if any non-foot collision geom is below ground, lift if so
    foot_geom_ids = {
        model.geom("FL").id, model.geom("FR").id,
        model.geom("RL").id, model.geom("RR").id,
    }
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    min_nf_z = float("inf")  # min non-foot z
    for i in range(model.ngeom):
        if i in foot_geom_ids or model.geom(i).name == "floor":
            continue
        if model.geom_group[i] != 3:
            continue
        min_nf_z = min(min_nf_z, data.geom_xpos[i][2])

    if min_nf_z < 0.08:
        # Lift the whole robot so the lowest non-foot part is at 8cm
        # Aggressive clearance to prevent head-on-ground tripod exploit
        qpos[2] += (0.08 - min_nf_z)

    return qpos


def _clamp_joints(qpos, model):
    for i in range(model.njnt):
        if model.jnt_type[i] == 3:  # hinge
            qposadr = model.jnt_qposadr[i]
            lo, hi = model.jnt_range[i]
            if lo != hi:
                qpos[qposadr] = np.clip(qpos[qposadr], lo, hi)
    return qpos


def evaluate(params, max_steps=2000):
    """Evaluate a single candidate. Uses worker-local model.

    Key design: non-foot floor contact at ANY step = immediate termination.
    This prevents the tripod cheat (head + 2 feet = stable but not a handstand).
    Height is heavily rewarded to push toward vertical poses.
    """
    model = _worker_model
    qpos = params_to_qpos(params, model.nq)
    qpos = _clamp_joints(qpos, model)
    qpos = _solve_base_z(qpos, model)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    torso_id = model.body("base_link").id
    fl_site = model.site("FL_foot").id
    fr_site = model.site("FR_foot").id

    # PD hold targets and act_to_joint mapping
    target_q = qpos[7:19].copy()
    Kp, Kd = 20.0, 0.5
    act_to_joint = np.zeros(model.nu, dtype=int)
    for i in range(model.nu):
        act_to_joint[i] = model.actuator_trnid[i, 0] - 1

    # Precompute geom sets for contact checking
    floor_id = model.geom("floor").id
    foot_geom_ids = {
        model.geom("FL").id, model.geom("FR").id,
        model.geom("RL").id, model.geom("RR").id,
    }
    robot_body_geoms = set()
    for i in range(model.ngeom):
        if model.geom_group[i] == 3 and i not in foot_geom_ids:
            gname = model.geom(i).name
            if gname and gname != "floor":
                robot_body_geoms.add(i)

    # Pre-check: simulate 5 steps and detect if non-foot body contacts floor.
    # If it does, this is a tripod pose. Penalize by how fast contact happens
    # (fewer steps = worse) but don't reject — give CMA-ES gradient.
    # Tripod detection runs AFTER the main sim — see below.

    # Initial metrics
    init_base_z = data.xpos[torso_id][2]

    # Joint limit cost (from initial pose)
    joint_limit_cost = 0.0
    for i in range(model.njnt):
        if model.jnt_type[i] == 3:
            qposadr = model.jnt_qposadr[i]
            lo, hi = model.jnt_range[i]
            if lo == hi:
                continue
            q = data.qpos[qposadr]
            margin = min(q - lo, hi - q) / (hi - lo)
            if margin < 0.15:
                joint_limit_cost += np.exp(5.0 * (1.0 - margin / 0.15)) - 1.0

    # Initial head height (lowest torso collision geom)
    torso_geoms_z = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == torso_id and model.geom_group[i] == 3:
            torso_geoms_z.append(data.geom_xpos[i][2])
    head_z = min(torso_geoms_z) if torso_geoms_z else init_base_z

    # Simulate
    survived = 0
    sum_base_z = 0.0
    sum_com_offset = 0.0
    body_contact_steps = 0
    settled_base_z = init_base_z  # updated at step 100

    for step in range(max_steps):
        cur_q = data.qpos[7:19]
        cur_dq = data.qvel[6:18]
        tau = Kp * (target_q - cur_q) + Kd * (0.0 - cur_dq)
        data.ctrl[:] = tau[act_to_joint]
        mujoco.mj_step(model, data)

        # Check non-foot body contact with floor EVERY step.
        # Terminate immediately — no tripod cheating.
        has_body_contact = False
        for c in range(data.ncon):
            g1 = data.contact[c].geom1
            g2 = data.contact[c].geom2
            if (g1 == floor_id and g2 in robot_body_geoms) or \
               (g2 == floor_id and g1 in robot_body_geoms):
                has_body_contact = True
                break

        if has_body_contact:
            body_contact_steps += 1
            break  # terminate on any non-foot floor contact

        base_z = data.xpos[torso_id][2]
        # Let PD settle for 100 steps (~0.2s), then terminate if dropping.
        # Use settled height as reference, not initial (which is pre-fall).
        if step == 100:
            settled_base_z = base_z
        if step > 100 and base_z < settled_base_z - 0.01:
            break

        survived += 1

        com = data.subtree_com[torso_id]
        fl_pos = data.site_xpos[fl_site]
        fr_pos = data.site_xpos[fr_site]
        feet_center = (fl_pos + fr_pos) / 2.0
        com_offset = np.sqrt(
            (com[0] - feet_center[0]) ** 2 + (com[1] - feet_center[1]) ** 2
        )
        sum_base_z += base_z
        sum_com_offset += com_offset

    avg_base_z = sum_base_z / max(survived, 1)
    avg_com_offset = sum_com_offset / max(survived, 1)

    # Tripod detection: if non-foot body contacts floor at the END,
    # the pose settled into a head-on-ground tripod. Reject.
    for c in range(data.ncon):
        g1 = data.contact[c].geom1
        g2 = data.contact[c].geom2
        if (g1 == floor_id and g2 in robot_body_geoms) or \
           (g2 == floor_id and g1 in robot_body_geoms):
            return 1e6

    # Cost terms
    body_contact_penalty = body_contact_steps * 10000.0
    height_reward = avg_base_z * 5000.0
    head_penalty = max(0.2 - head_z, 0.0) * 20000.0

    cost = (
        -survived
        + avg_com_offset * 500.0
        + joint_limit_cost * 50.0
        + body_contact_penalty
        + head_penalty
        - height_reward
    )

    evaluate._last_breakdown = {
        "survived": survived,
        "com_offset": avg_com_offset * 500.0,
        "joint_limit": joint_limit_cost * 50.0,
        "body_contact": body_contact_penalty,
        "head_penalty": head_penalty,
        "height_reward": height_reward,
        "avg_base_z": avg_base_z,
        "head_z": head_z,
        "cost": cost,
    }
    return cost


def _eval_wrapper(args):
    params, max_steps = args
    return evaluate(params, max_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-all", type=str, default=None,
                        help="Save every new best to this JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mj_model = load_model()
    nq = mj_model.nq

    # Seed pose (7 params, left-right symmetric, x/y/yaw fixed, z computed from feet)
    x0 = [
        0.0,    # extra pitch (beyond 90deg, how far past vertical)
        0.0,    # front hip (abduction, outward only)
        0.0,    # front thigh
        -0.84,  # front calf (most extended)
        0.0,    # rear hip (abduction, outward only)
        1.5,    # rear thigh (tucked)
        -1.8,   # rear calf (tucked)
    ]

    sigma0 = 0.3

    bounds = [
        [-1.5, 1.5],     # extra pitch
        [0.0, 0.5],      # front hip (outward only)
        [-1.5, 3.0],     # front thigh
        [-2.7, -0.84],   # front calf
        [0.0, 0.5],      # rear hip (outward only)
        [-0.5, 4.0],     # rear thigh
        [-2.7, -0.84],   # rear calf
    ]

    opts = {
        "popsize": args.popsize,
        "maxiter": args.generations,
        "seed": args.seed,
        "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
        "tolfun": 1e-6,
        "verbose": -1,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    param_names = [
        "pitch", "f_hip", "f_thigh", "f_calf",
        "r_hip", "r_thigh", "r_calf",
    ]

    print(f"CMA-ES: {args.generations} gens, popsize={args.popsize}, "
          f"max_steps={args.max_steps}, workers={args.workers}, params={len(x0)}")
    print(f"Seed: {x0}")
    print(f"Cost = -survived + com_offset*500 + body_ground*2000 "
          f"+ joint_limit*50")
    print(f"{'=' * 90}\n", flush=True)

    best_score = float("inf")
    best_params = None
    all_bests = []  # list of (gen, score, params) for --save-all
    gen = 0

    pool = Pool(processes=args.workers, initializer=_init_worker)

    try:
        while not es.stop():
            solutions = es.ask()
            gen += 1

            # Parallel eval across CPU cores
            work = [(s, args.max_steps) for s in solutions]
            costs_list = pool.map(_eval_wrapper, work)
            costs_np = np.array(costs_list)

            es.tell(solutions, costs_np.tolist())

            gen_best_idx = np.argmin(costs_np)
            gen_best = costs_np[gen_best_idx]
            gen_mean = np.mean(costs_np)
            gen_worst = np.max(costs_np)
            improved = gen_best < best_score

            if improved:
                best_score = gen_best
                best_params = solutions[gen_best_idx]

            marker = " *NEW BEST*" if improved else ""
            print(f"Gen {gen:3d} | best={-best_score:7.1f} | "
                  f"gen: best={-gen_best:7.1f} mean={-gen_mean:7.1f} "
                  f"worst={-gen_worst:7.1f} | sigma={es.sigma:.4f}{marker}",
                  flush=True)

            if improved:
                parts = " ".join(
                    f"{n}={v:+.3f}" for n, v in zip(param_names, best_params)
                )
                print(f"         {parts}", flush=True)
                # Re-eval best in main process to get breakdown
                global _worker_model
                if _worker_model is None:
                    _init_worker()
                evaluate(best_params, args.max_steps)
                bd = evaluate._last_breakdown
                print(f"         survived={bd['survived']:.0f} com={bd['com_offset']:.0f} "
                      f"jlim={bd['joint_limit']:.0f} contact={bd['body_contact']:.0f} "
                      f"head={bd['head_penalty']:.0f} "
                      f"height={bd['height_reward']:.0f} base_z={bd['avg_base_z']:.3f} "
                      f"head_z={bd['head_z']:.3f}", flush=True)
                all_bests.append({
                    "gen": gen,
                    "score": float(-best_score),
                    "params": {n: float(v) for n, v in zip(param_names, best_params)},
                    "params_list": [float(v) for v in best_params],
                })
                # Flush to disk immediately so we never lose results
                if args.save_all:
                    import json
                    with open(args.save_all, "w") as f:
                        json.dump(all_bests, f, indent=2)

            # Snapshot every 10 generations
            if gen % 10 == 0 and best_params is not None:
                _render_snapshot(mj_model, best_params, gen)

    finally:
        pool.close()
        pool.join()

    # Save all bests to JSON if requested.
    if args.save_all and all_bests:
        import json
        with open(args.save_all, "w") as f:
            json.dump(all_bests, f, indent=2)
        print(f"\nSaved {len(all_bests)} best results to {args.save_all}")

    print(f"\n{'=' * 90}")
    print(f"Best cost: {best_score:.1f}")
    print(f"Best params: {[f'{p:.4f}' for p in best_params]}")

    # Final CPU evaluation
    _init_worker()
    qpos = params_to_qpos(best_params, nq)
    qpos = _clamp_joints(qpos, mj_model)
    qpos = _solve_base_z(qpos, mj_model)

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)

    torso_id = mj_model.body("base_link").id
    fl = data.site_xpos[mj_model.site("FL_foot").id]
    fr = data.site_xpos[mj_model.site("FR_foot").id]
    com = data.subtree_com[torso_id]
    feet_c = (fl + fr) / 2
    print(f"\nFL foot: {fl}")
    print(f"FR foot: {fr}")
    print(f"Base: {data.xpos[torso_id]}")
    print(f"CoM: {com}")
    print(f"CoM offset from feet: ({com[0] - feet_c[0]:.4f}, {com[1] - feet_c[1]:.4f})")

    print(f"\nqpos[0:7] (base): {list(qpos[0:7])}")
    print(f"qpos[7:19] (joints): {list(qpos[7:19])}")

    # CPU survival test
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(mj_model, data)
    survived = 0
    for _ in range(args.max_steps):
        mujoco.mj_step(mj_model, data)
        survived += 1
        if data.xpos[torso_id][2] < 0.1:
            break
    print(f"CPU survival: {survived}/{args.max_steps} steps "
          f"({survived * mj_model.opt.timestep:.2f}s)")

    if args.render:
        _render_final(mj_model, qpos, args.max_steps)


def _render_snapshot(mj_model, params, gen):
    """Render a single PNG of the current best pose."""
    import mediapy

    qpos = params_to_qpos(params, mj_model.nq)
    qpos = _clamp_joints(qpos, mj_model)
    qpos = _solve_base_z(qpos, mj_model)

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)

    renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    out = Path("/home/stevenman/Desktop/Work/Research/jax-learning/tmp_videos")
    out.mkdir(exist_ok=True)

    for suffix, az, el, dist, look_z in [
        ("side", 90, 0, 1.2, 0.1),
        ("iso", 135, -25, 1.5, 0.25),
    ]:
        cam = mujoco.MjvCamera()
        cam.azimuth = az
        cam.elevation = el
        cam.distance = dist
        cam.lookat[:] = [0, 0, look_z]
        renderer.update_scene(data, cam)
        frame = renderer.render()
        path = out / f"handstand_gen{gen:03d}_{suffix}.png"
        mediapy.write_image(str(path), frame)
    print(f"         snapshot: gen{gen:03d}_side.png + gen{gen:03d}_iso.png", flush=True)
    renderer.close()


def _render_final(mj_model, qpos, max_steps):
    """Render PNGs + rollout video of the final best pose."""
    import mediapy

    renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    out = Path("/home/stevenman/Desktop/Work/Research/jax-learning/tmp_videos")
    out.mkdir(exist_ok=True)

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)

    for vname, az, el in [("front", 180, -15), ("side", 90, -5), ("3q", 135, -20)]:
        cam = mujoco.MjvCamera()
        cam.azimuth = az
        cam.elevation = el
        cam.distance = 1.5
        cam.lookat[:] = [0, 0, 0.3]
        renderer.update_scene(data, cam)
        frame = renderer.render()
        path = out / f"handstand_optimized_{vname}.png"
        mediapy.write_image(str(path), frame)
        print(f"Saved: {path}")

    # Rollout video
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)

    frames = []
    fps = 50
    render_every = max(1, int(1.0 / (fps * mj_model.opt.timestep)))
    for i in range(max_steps):
        mujoco.mj_step(mj_model, data)
        if i % render_every == 0:
            cam = mujoco.MjvCamera()
            cam.azimuth = 135
            cam.elevation = -20
            cam.distance = 1.5
            cam.lookat[:] = [0, 0, 0.3]
            renderer.update_scene(data, cam)
            frames.append(renderer.render().copy())

    path = out / "handstand_optimized_rollout.mp4"
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"Saved: {path} ({len(frames)} frames, {len(frames) / fps:.1f}s)")

    renderer.close()


if __name__ == "__main__":
    main()
