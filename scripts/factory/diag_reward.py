"""Diagnostic: reward signal across the Phase 2 scripted trajectory.

Verifies the reward shape is monotonic in peg descent: low at the
"settle" / lateral pushes, rising as the -z dive segment drives the
peg into the bore.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jp
import numpy as np

from jax_rl.envs.manipulation.factory.factory_peg_insert import (
    FactoryPegInsert,
    default_config,
)


def main():
    cfg = default_config()
    cfg.actuator_mode = "motor"
    env = FactoryPegInsert(cfg)
    state = env.reset(jax.random.PRNGKey(0))

    segments = [
        ("settle",   jp.zeros(6),                              30),
        ("+y push",  jp.array([0., 1., 0., 0., 0., 0.]),       30),
        ("-y push",  jp.array([0., -1., 0., 0., 0., 0.]),      30),
        ("-z dive",  jp.array([0., 0., -1., 0., 0., 0.]),      60),
    ]

    def measure(state, label, step_global):
        peg_z = float(state.data.xpos[env._peg_body_id, 2])
        fixed_z = float(state.info["fixed_pos"][2])
        clip_z = float(state.info["clip_anchor"][2])
        hole_top_z = fixed_z + cfg.asset_height
        engaged = hole_top_z - peg_z > cfg.asset_height * cfg.engage_threshold
        success = hole_top_z - peg_z > cfg.asset_height * cfg.success_threshold
        print(
            f"step={step_global:3d} [{label:>7s}] "
            f"reward={float(state.reward):+.4f}  "
            f"peg_z={peg_z:.4f}  hole_top_z={hole_top_z:.4f}  "
            f"clip_z={clip_z:.4f}  "
            f"engaged={int(engaged)} success={int(success)}"
        )

    measure(state, "reset", 0)
    step_global = 0
    for name, action, n in segments:
        for _ in range(n):
            state = env.step(state, action)
            step_global += 1
        measure(state, name, step_global)


if __name__ == "__main__":
    main()
