"""Diagnostic: full -z action for 900 steps. How far can peg descend?

If OSC controller can drive peg fully into bore with constant -z action, the
gradient signal is reachable and the SAC collapse is a policy-learning
problem (entropy/exploration). If even sustained -z can't insert the peg,
the issue is controller-limited and OSC gains / pos_threshold need tuning.
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("MUJOCO_GL", "egl")

import jax, jax.numpy as jp, numpy as np

from jax_rl.envs.manipulation.factory.factory_peg_insert import (
    FactoryPegInsert, default_config,
)


def main():
    cfg = default_config()
    cfg.actuator_mode = "motor"
    env = FactoryPegInsert(cfg)
    state = env.reset(jax.random.PRNGKey(0))

    # Match v4's collapsed-policy action: all axes saturated at -1.
    action = jp.array([-1., -1., -1., -1., -1., -1.])

    def peg_z(s):
        return float(s.data.xpos[env._peg_body_id, 2])
    def fingertip_z(s):
        return float(s.data.site_xpos[env._fingertip_site_id, 2])

    print(f"step=  0  peg_z={peg_z(state):.4f}  fingertip_z={fingertip_z(state):.4f}  reward={float(state.reward):.4f}")
    n = 900
    samples = (0, 30, 60, 90, 150, 300, 500, 700, n-1)
    total_reward = 0.0
    for i in range(n):
        state = env.step(state, action)
        total_reward += float(state.reward)
        if i in samples:
            print(
                f"step={i:3d}  peg_z={peg_z(state):.4f}  "
                f"fingertip_z={fingertip_z(state):.4f}  "
                f"reward={float(state.reward):.4f}  cum={total_reward:.2f}"
            )
    print(f"\nTotal return: {total_reward:.2f} over {n} steps "
          f"(target ≥ 0.5/step × 900 = 450)")


if __name__ == "__main__":
    main()
