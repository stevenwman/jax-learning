"""Static MPM-only containment walls.

Generates closed-box ``wp.Mesh`` walls that can be passed to
``MPMModel.setup_collider`` as static colliders to bound particles. The walls
are pure MPM colliders — they are never added to the Newton/MuJoCo model, so
the rigid-body solver never sees them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class WallSpec:
    """Axis-aligned box wall in world space (all values in metres).

    hx/hy/hz are half-extents; center is the wall centre.
    """
    hx: float
    hy: float
    hz: float
    center: tuple  # (cx, cy, cz)


def build_wall_mesh(spec: WallSpec, device) -> wp.Mesh:
    """Build a closed triangular box mesh for use as an MPM-only static collider."""
    cx, cy, cz = spec.center
    hx, hy, hz = spec.hx, spec.hy, spec.hz
    verts = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    # Outward-facing CCW winding from outside each face.
    indices = np.array([
        0, 3, 2,  0, 2, 1,   # -z (bottom)
        4, 5, 6,  4, 6, 7,   # +z (top)
        0, 1, 5,  0, 5, 4,   # -y
        3, 7, 6,  3, 6, 2,   # +y
        0, 4, 7,  0, 7, 3,   # -x
        1, 2, 6,  1, 6, 5,   # +x
    ], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )


class BoundingWalls:
    """Four static walls forming a rectangular fence around a particle bed.

    Holds Python references to the wp.Mesh objects so their GPU buffers stay
    alive for the lifetime of the simulation.
    """

    def __init__(self, specs, device):
        self.specs = list(specs)
        self.meshes = [build_wall_mesh(s, device) for s in self.specs]
        self._device = device

    @classmethod
    def around_bed(cls, *, x_range, y_range, height, thickness, device):
        """Build 4 walls bounding a rectangular bed.

        x_range, y_range: ``(lo, hi)`` world-space extents of the interior bed.
        height:    total wall height (walls span ``z ∈ [0, height]``).
        thickness: total wall thickness (walls sit just outside the bed).
        """
        x_lo, x_hi = x_range
        y_lo, y_hi = y_range
        t = thickness * 0.5
        h = height * 5.5
        cx_mid = 0.5 * (x_lo + x_hi)
        cy_mid = 0.5 * (y_lo + y_hi)
        x_half = 0.5 * (x_hi - x_lo)
        y_half = 0.5 * (y_hi - y_lo)

        specs = [
            WallSpec(t,      y_half, h, (x_hi + t, cy_mid, h)),   # +x face
            WallSpec(t,      y_half, h, (x_lo - t, cy_mid, h)),   # -x face
            WallSpec(x_half, t,      h, (cx_mid,   y_hi + t, h)), # +y face
            WallSpec(x_half, t,      h, (cx_mid,   y_lo - t, h)), # -y face
        ]
        return cls(specs, device)

    @classmethod
    def around_particles(cls, particle_q, *, device, height=None,
                         thickness=0.1, padding=0.0):
        """Build 4 walls that hug the XY axis-aligned bounding box of all
        particles in ``particle_q``.

        particle_q: positions of shape ``(N, 3)``. Accepts ``wp.array``,
                    ``np.ndarray``, or any sequence ``warp`` can convert.
        height:     total wall height (``z ∈ [0, height]``). Defaults to the
                    largest particle z plus ``padding`` so walls reach above
                    the tallest spawn.
        thickness:  total wall thickness; walls sit just outside the AABB.
        padding:    extra space added to the XY extents on each side (and to
                    the auto-computed height).
        """
        if hasattr(particle_q, "numpy"):
            pts = particle_q.numpy()
        else:
            pts = np.asarray(particle_q, dtype=np.float32)
        pts = pts.reshape(-1, 3)
        if pts.size == 0:
            raise ValueError("around_particles: no particles to bound")

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        if height is None:
            height = float(hi[2]) + padding

        return cls.around_bed(
            x_range=(float(lo[0]) - padding, float(hi[0]) + padding),
            y_range=(float(lo[1]) - padding, float(hi[1]) + padding),
            height=float(height),
            thickness=float(thickness),
            device=device,
        )

    def collider_meshes(self):
        return list(self.meshes)

    def collider_body_ids(self):
        """All walls are static — no associated Newton body."""
        return [None] * len(self.specs)

    def render_xforms(self):
        return wp.array(
            [wp.transform(p=wp.vec3(*s.center), q=wp.quat_identity()) for s in self.specs],
            dtype=wp.transform, device=self._device,
        )

    def render_scales(self):
        return [(s.hx, s.hy, s.hz) for s in self.specs]
