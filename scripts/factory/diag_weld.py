"""Diagnostic: peg-hand weld coherence across Phase 2 scripted rollout.

If weld holds: |peg_xyz - hand_xyz| stays near relpose magnitude (0.13m)
and the projection of (peg - hand) onto hand's body-z is constant at 0.13.

If weld is broken under OSC wrench: distance drifts.
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
        ("settle",  jp.zeros(6),                            30),
        ("+y",      jp.array([0., 1., 0., 0., 0., 0.]),     30),
        ("-y",      jp.array([0., -1., 0., 0., 0., 0.]),    30),
        ("-z dive", jp.array([0., 0., -1., 0., 0., 0.]),    60),
    ]

    def measure(state, label, step_global):
        peg = np.asarray(state.data.xpos[env._peg_body_id])
        hand = np.asarray(state.data.xpos[env._hand_body_id])
        diff = peg - hand
        dist = np.linalg.norm(diff)
        # Project diff onto hand body-z (column 2 of hand rotation matrix).
        hand_mat = np.asarray(state.data.xmat[env._hand_body_id]).reshape(3, 3)
        z_hand = hand_mat[:, 2]
        z_proj = float(np.dot(diff, z_hand))
        offset_residual = diff - z_proj * z_hand
        ortho_err = float(np.linalg.norm(offset_residual))
        print(
            f"step={step_global:3d} [{label:>7s}] "
            f"|peg-hand|={dist:.5f}  z_hand·(peg-hand)={z_proj:+.5f}  "
            f"ortho_err={ortho_err:.6f}"
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
