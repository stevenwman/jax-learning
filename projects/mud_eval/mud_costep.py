"""Co-stepping seam: integrate the robot AND the MPM mud at the SAME substep rate
(the controller rate), exchanging two-way coupling forces every substep — instead
of the example's decimated scheme (robot 200 Hz, mud 50 Hz, mud force HELD across
the 4 robot substeps and integrated once/frame).

WHY: the robot-mud contact is the physics this eval measures. Decimating the mud
to 50 Hz means the feet only push the mud 50×/s and the robot feels stale reaction
forces — the interaction we care about, under-resolved. Co-stepping resolves the
coupling at the controller rate (250 Hz). Cost: ~5× MPM solves (the expensive
part) — accepted for eval fidelity.

HOW: fold the MPM step into simulate_robot's substep loop (interleaved at sim_dt),
make simulate_sand a no-op. Forces are scaled by sim_dt (was frame_dt). Control is
held across substeps for joint-PD (the solver's PD recomputes torque each substep);
the OSC per-substep torque hook (M2) slots into the same loop.

Usage:
    import mud_costep
    mud_costep.enable(sim_substeps=5)          # 5 -> sim_dt 0.004 = 250 Hz
    example = ex.Example(viewer, args)
    mud_costep.apply(example)                  # post-build: eager + set rate
"""
from __future__ import annotations

import warp as wp
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex
from newton.examples.mpm.mpm_go2_multi.twoway_coupling_go2 import (
    compute_body_forces, subtract_body_force,
)

_CFG = {"sim_substeps": 5}


def _costep_simulate_robot(self):
    """Robot + mud co-stepped at sim_dt, two-way force exchange every substep."""
    for _ in range(self.sim_substeps):
        self.state_0.clear_forces()
        # mud -> body force (this substep's impulses; force = impulse / sim_dt)
        wp.launch(
            compute_body_forces,
            dim=self.collider_impulse_ids.shape[0],
            inputs=[
                self.sim_dt,
                self.collider_impulse_ids, self.collider_impulses,
                self.collider_impulse_pos, self.collider_body_id,
                self.state_0.body_q, self.model.body_com, self.state_0.body_f,
            ],
        )
        self.body_sand_forces.assign(self.state_0.body_f)   # save for anti-double-count
        self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        # mud co-step: remove the applied force from body vel (pre-force vel for MPM),
        # then advance the mud one sim_dt, then collect fresh impulses for next substep.
        if self.state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.state_0.body_q.shape[0],
                inputs=[
                    self.sim_dt,
                    self.state_0.body_q, self.state_0.body_qd, self.body_sand_forces,
                    self.model.body_inv_inertia, self.model.body_inv_mass,
                    self.state_0.body_q, self.state_0.body_qd,
                ],
            )
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.sim_dt)
        self._collect_collider_impulses()


def _costep_simulate_sand(self):
    pass  # folded into _costep_simulate_robot


def _no_capture(self):
    # The example graph-captures simulate_robot on CUDA; our version now calls
    # mpm_solver.step inside it, and the MPM grid build (nanovdb PointsToGrid)
    # can't run during graph capture (CUDA err 900). Run everything eager — the
    # MPM cost dominates anyway, so the robot-substep graph saved little.
    self.graph = None
    self.sand_graph = None


def enable(sim_substeps: int = 5):
    _CFG["sim_substeps"] = int(sim_substeps)
    ex.Example.simulate_robot = _costep_simulate_robot
    ex.Example.simulate_sand = _costep_simulate_sand
    ex.Example.capture = _no_capture          # no graph capture (MPM can't be captured)


def apply(example):
    """Post-build: set the co-step rate (graph already disabled via _no_capture)."""
    example.graph = None
    example.sand_graph = None
    example.sim_substeps = _CFG["sim_substeps"]
    example.sim_dt = example.frame_dt / _CFG["sim_substeps"]
    print(f"[COSTEP] robot+mud co-stepped: sim_substeps={example.sim_substeps} "
          f"sim_dt={example.sim_dt:.4f}s ({1.0 / example.sim_dt:.0f} Hz), mud now co-steps")
