"""Block shape builders for PushTEnv cross-shape experiments.

Each builder returns `(body, [shapes])` matching `PushTEnv.add_tee` signature.
All shapes are decomposed into convex `pymunk.Poly` pieces (pymunk constraint).

Concave letters (S, U) built from annular sectors — fat ring segments where
every wedge is tangent to a circle, giving uniform curvature with no kink
artifacts. See `.context/lessons/manipulation.md` for design rationale.
"""
import numpy as np
import pygame
import pymunk

MASK = pymunk.ShapeFilter.ALL_MASKS()


def _make_poly_shape(body, verts, color):
    s = pymunk.Poly(body, verts)
    s.color = pygame.Color(color)
    s.filter = pymunk.ShapeFilter(mask=MASK)
    return s


def _multi_poly(space, polys, position, angle, color):
    inertia_total = sum(pymunk.moment_for_poly(1, vertices=p) for p in polys)
    body = pymunk.Body(1, inertia_total)
    shapes = [_make_poly_shape(body, p, color) for p in polys]
    body.angle, body.position = angle, position
    body.friction = 1
    space.add(body, *shapes)
    return body, shapes


def _ring_polys(center, inner_r, outer_r, theta_start, theta_end, n):
    """Annular sector → n convex wedge quads tangent to concentric circles."""
    cx, cy = center
    thetas = np.linspace(theta_start, theta_end, n + 1)
    polys = []
    for i in range(n):
        t0, t1 = thetas[i], thetas[i + 1]
        polys.append([
            (cx + inner_r * np.cos(t0), cy + inner_r * np.sin(t0)),
            (cx + outer_r * np.cos(t0), cy + outer_r * np.sin(t0)),
            (cx + outer_r * np.cos(t1), cy + outer_r * np.sin(t1)),
            (cx + inner_r * np.cos(t1), cy + inner_r * np.sin(t1)),
        ])
    return polys


def add_ellipse(space, position, angle, color="LightSlateGray",
                semi_major=60, semi_minor=40):
    """Single 32-vert convex polygon approximating an ellipse."""
    n = 32
    verts = [(semi_major * np.cos(t), semi_minor * np.sin(t))
             for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    inertia = pymunk.moment_for_poly(1, vertices=verts)
    body = pymunk.Body(1, inertia)
    shape = _make_poly_shape(body, verts, color)
    body.angle, body.position = angle, position
    body.friction = 1
    space.add(body, shape)
    return body, [shape]


def add_triangle(space, position, angle, color="LightSlateGray"):
    """Isoceles triangle — single 3-vert convex polygon."""
    verts = [(-60, -40), (60, -40), (0, 40)]
    inertia = pymunk.moment_for_poly(1, vertices=verts)
    body = pymunk.Body(1, inertia)
    shape = _make_poly_shape(body, verts, color)
    body.angle, body.position = angle, position
    body.friction = 1
    space.add(body, shape)
    return body, [shape]


def add_s(space, position, angle, color="LightSlateGray", n_per_ring=10):
    """Letter S — two 270° fat rings, rot-180 symmetric, overlapping mid-strip.

    Rings produce uniform curvature; end wedge trimmed for tapered tip.
    """
    s = 1.367   # 1.25³ × 0.7 — tuned proportions
    block_height = (38 - 22) * s
    up_shift = block_height * 3 / 14
    arc_trim = 30
    upper = _ring_polys(
        center=(0, 27 * s + up_shift), inner_r=22 * s, outer_r=38 * s,
        theta_start=np.radians(255),
        theta_end=np.radians(195 + 360 - arc_trim),
        n=n_per_ring - 1,
    )
    lower = _ring_polys(
        center=(0, -27 * s), inner_r=22 * s, outer_r=38 * s,
        theta_start=np.radians(75),
        theta_end=np.radians(15 + 360 - arc_trim),
        n=n_per_ring - 1,
    )
    return _multi_poly(space, upper + lower, position, angle, color)


SHAPE_BUILDERS = {
    "ellipse":  add_ellipse,
    "triangle": add_triangle,
    "s":        add_s,
    # "tee" dispatched via PushTEnv.add_tee directly (already in pusht.py).
}
