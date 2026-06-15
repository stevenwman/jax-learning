"""Refactor smoke test — builds + steps representative Go2 Warp envs and asserts
they run clean. The fast oracle for behaviour-preserving refactors on the
`linen_refactor` branch (controller/terrain/force-field component moves).

Covers every low-level controller (JointPD / OSC / VarImpedance / VarImpedanceMass),
the analytic-mud force field, and the physical motor model — i.e. all four concerns
in go2_warp_components. It does NOT replace the pytest suite (parity/snapshot tests
are the numerical gate); it's the quick "did I break env construction or the
substep loop" check, and prints a per-env signature for eyeball comparison across
a change (loose — GPU math is non-deterministic run-to-run).

    uv run python scripts/smoke_refactor.py          # all envs, exit 1 on any failure

Run it BEFORE and AFTER a refactor commit; signatures should match to ~1e-3.
"""
from __future__ import annotations

import sys
import jax
import jax.numpy as jp
import numpy as np

# Representative envs by short alias — one per controller × the mud field, plus a
# joint-PD mud baseline. Aliases resolve to the long canonical variant names.
SMOKE_ENVS = [
    "flat-jointpd",   # JointPD + Flat + TorqueOnly
    "flat-varimp",    # VarImpedance (per-axis K+D) + Flat + MotorModel
    "flat-mass",      # VarImpedanceMass (virtual A) + Flat + MotorModel
    "mud-slowfirm",   # VarImpedance + MudField + MotorModel
    "mud-mass",       # VarImpedanceMass + MudField
    "mud-jointpd",    # JointPD + MudField (PD mud baseline)
]
N_STEPS = 8


def _build(alias: str):
    """Resolve an alias → canonical variant → host env instance (mirrors the
    registration in training/env_backends/mjx_backend.py)."""
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS, GO2_ENV_ALIASES
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick, WarpJoystickNoAccel
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.locomotion.go2_warp_flat_postrack import WarpFlatPosTrack

    cls_map = {"WarpJoystick": WarpJoystick, "WarpJoystickNoAccel": WarpJoystickNoAccel,
               "WarpJoystickCurriculum": WarpJoystickCurriculum,
               "WarpFlatPosTrack": WarpFlatPosTrack}
    name = GO2_ENV_ALIASES.get(alias, alias)
    v = GO2_WARP_VARIANTS[name]
    return cls_map[v.cls](task="flat_terrain", config=v.config())


def _signature(alias: str) -> dict:
    env = _build(alias)
    act_dim = int(env.action_size)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    action = jp.zeros(act_dim)
    for _ in range(N_STEPS):
        state = step(state, action)

    # obs may be a dict (asymmetric actor/critic) or a flat array.
    obs = state.obs
    obs_dim = (sum(int(np.prod(v.shape)) for v in obs.values())
               if isinstance(obs, dict) else int(np.prod(obs.shape)))
    base_z = float(state.data.qpos[2])
    reward = float(state.reward)

    # Hard asserts: construction + substep loop produced finite, sane output.
    assert np.isfinite(reward), f"{alias}: non-finite reward {reward}"
    assert np.isfinite(base_z), f"{alias}: non-finite base_z {base_z}"
    assert act_dim > 0 and obs_dim > 0, f"{alias}: bad dims act={act_dim} obs={obs_dim}"
    return {"alias": alias, "act": act_dim, "obs": obs_dim,
            "base_z": base_z, "reward": reward}


def main() -> int:
    print(f"smoke: building + stepping {len(SMOKE_ENVS)} envs ({N_STEPS} steps each) "
          f"on {jax.devices()[0]}\n")
    print(f"{'env':<16}{'act':>5}{'obs':>6}{'base_z':>12}{'reward':>12}   status")
    print("-" * 64)
    failures = []
    for alias in SMOKE_ENVS:
        try:
            s = _signature(alias)
            print(f"{s['alias']:<16}{s['act']:>5}{s['obs']:>6}"
                  f"{s['base_z']:>12.5f}{s['reward']:>12.5f}   ok")
        except Exception as e:  # noqa: BLE001 — smoke test: report, don't raise
            failures.append(alias)
            print(f"{alias:<16}{'':>5}{'':>6}{'':>12}{'':>12}   FAIL  {type(e).__name__}: {e}")
    print("-" * 64)
    if failures:
        print(f"\nFAILED: {len(failures)}/{len(SMOKE_ENVS)} envs — {failures}")
        return 1
    print(f"\nPASS: all {len(SMOKE_ENVS)} envs built + stepped clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
