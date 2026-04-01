"""Test bongo board physics — record videos of drop/disturbance scenarios.

Usage:
    MUJOCO_GL=egl uv run python tools/bongo_board_test.py
    MUJOCO_GL=egl uv run python tools/bongo_board_test.py --test disturbance
    MUJOCO_GL=egl uv run python tools/bongo_board_test.py --test all
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import mediapy

XML_PATH = Path(__file__).parent.parent / "jax_rl/envs/locomotion/xmls/bongo_test_scene.xml"
OUTPUT_DIR = Path("/tmp/bongo_test")


def load_model():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    return model, data


def setup_renderer(model, width=1280, height=720):
    renderer = mujoco.Renderer(model, height=height, width=width)
    return renderer


def record_video(model, data, renderer, filename, n_steps=2000, fps=50):
    """Step physics and record frames."""
    frames = []
    dt = model.opt.timestep
    render_every = max(1, int(1.0 / (fps * dt)))

    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if i % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"Saved: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return path


def test_drop(model, data, renderer):
    """Drop board from 30cm above ground — should land on roller and settle."""
    mujoco.mj_resetData(model, data)

    # Lift board 30cm above its resting position.
    board_jnt_id = model.joint("board_joint").id
    qpos_adr = model.jnt_qposadr[board_jnt_id]
    data.qpos[qpos_adr + 2] += 0.3  # z offset

    mujoco.mj_forward(model, data)
    print("\n=== Test: DROP (board falls 30cm) ===")
    print(f"  Board initial z: {data.qpos[qpos_adr + 2]:.3f}")
    return record_video(model, data, renderer, "drop.mp4", n_steps=3000)


def test_tilt(model, data, renderer):
    """Start with board tilted 15deg — should rock and settle."""
    mujoco.mj_resetData(model, data)

    board_jnt_id = model.joint("board_joint").id
    qpos_adr = model.jnt_qposadr[board_jnt_id]

    # Tilt 15 degrees around Y axis (board rocks along its long axis X).
    angle = np.radians(15)
    # Quaternion for rotation around Y: [cos(a/2), 0, sin(a/2), 0]
    data.qpos[qpos_adr + 3] = np.cos(angle / 2)  # w
    data.qpos[qpos_adr + 4] = 0.0                 # x
    data.qpos[qpos_adr + 5] = np.sin(angle / 2)   # y
    data.qpos[qpos_adr + 6] = 0.0                 # z

    mujoco.mj_forward(model, data)
    print("\n=== Test: TILT (15deg initial tilt around Y) ===")
    return record_video(model, data, renderer, "tilt.mp4", n_steps=3000)


def test_disturbance(model, data, renderer):
    """Let board settle, then apply lateral force impulse."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("\n=== Test: DISTURBANCE (settle 1s, then lateral push) ===")

    frames = []
    dt = model.opt.timestep
    fps = 50
    render_every = max(1, int(1.0 / (fps * dt)))
    total_steps = 4000  # 8 seconds at dt=0.002

    board_body_id = model.body("board").id

    for i in range(total_steps):
        # Apply a lateral force impulse at t=1.0s for 0.1s.
        t = i * dt
        if 1.0 <= t < 1.1:
            data.xfrc_applied[board_body_id, 0] = 15.0  # 15N in +X
        elif 3.0 <= t < 3.1:
            data.xfrc_applied[board_body_id, 0] = -20.0  # 20N in -X
        else:
            data.xfrc_applied[board_body_id, :] = 0.0

        mujoco.mj_step(model, data)

        if i % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "disturbance.mp4"
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"Saved: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return path


def test_roller_roll(model, data, renderer):
    """Apply force to roller area to make it roll along the board."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("\n=== Test: ROLLER ROLL (push roller to test slide+spin coupling) ===")

    frames = []
    dt = model.opt.timestep
    fps = 50
    render_every = max(1, int(1.0 / (fps * dt)))
    total_steps = 4000

    # Apply torque via the roller spin joint to test coupling.
    roller_spin_id = model.joint("roller_spin").id
    roller_slide_id = model.joint("roller_slide").id
    spin_qvel_adr = model.jnt_dofadr[roller_spin_id]
    slide_qpos_adr = model.jnt_qposadr[roller_slide_id]
    spin_qpos_adr = model.jnt_qposadr[roller_spin_id]

    for i in range(total_steps):
        t = i * dt

        # Give roller a spin impulse at t=1s.
        if 0.99 <= t < 1.0:
            data.qvel[spin_qvel_adr] = 5.0  # rad/s spin

        mujoco.mj_step(model, data)

        if i % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

            # Print slide/spin state periodically.
            if i % (render_every * 10) == 0:
                slide_pos = data.qpos[slide_qpos_adr]
                spin_pos = data.qpos[spin_qpos_adr]
                expected_slide = -0.05842 * spin_pos
                print(f"  t={t:.1f}s  slide={slide_pos:.4f}  spin={spin_pos:.4f}  "
                      f"expected_slide={expected_slide:.4f}  "
                      f"error={abs(slide_pos - expected_slide):.6f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "roller_roll.mp4"
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"Saved: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["drop", "tilt", "disturbance", "roller_roll", "all"])
    args = parser.parse_args()

    model, data = load_model()
    renderer = setup_renderer(model)

    # Print model info.
    print(f"Model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Bodies: {[model.body(i).name for i in range(model.nbody)]}")
    print(f"Joints: {[model.joint(i).name for i in range(model.njnt)]}")
    print(f"Geoms: {[model.geom(i).name for i in range(model.ngeom)]}")

    tests = {
        "drop": test_drop,
        "tilt": test_tilt,
        "disturbance": test_disturbance,
        "roller_roll": test_roller_roll,
    }

    if args.test == "all":
        for name, fn in tests.items():
            model, data = load_model()  # fresh state each test
            fn(model, data, renderer)
    else:
        tests[args.test](model, data, renderer)

    renderer.close()
    print(f"\nAll videos saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
