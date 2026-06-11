"""Backend seam: force SolverMuJoCo(use_mujoco_cpu=True). The GPU mujoco_warp 0.0.2
collision path silently drops robot<->PLANE contacts (so the robot falls through
flat ground off the mud); the CPU mujoco backend resolves them — giving walkable
ground AND keeping the MPM mud coupling (forces apply through xfrc_applied). For a
single robot it's not slower (the MPM solve dominates; see HANDOFF). Requires graph
capture OFF (mud_costep._no_capture) — the CPU path does GPU->CPU copies.

    import mud_cpu; mud_cpu.enable()   # before constructing the example
"""
from __future__ import annotations

import newton

_orig_init = newton.solvers.SolverMuJoCo.__init__


def _cpu_init(self, model, *a, **kw):
    kw["use_mujoco_cpu"] = True
    return _orig_init(self, model, *a, **kw)


def enable():
    newton.solvers.SolverMuJoCo.__init__ = _cpu_init


def disable():
    newton.solvers.SolverMuJoCo.__init__ = _orig_init
