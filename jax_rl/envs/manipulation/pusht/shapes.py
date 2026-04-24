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


def _rotated_rect_verts(from_pt, to_pt, width):
    """Convex rect verts for a thick segment from `from_pt` to `to_pt`."""
    p0 = np.asarray(from_pt, dtype=float)
    p1 = np.asarray(to_pt, dtype=float)
    axis = p1 - p0
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    perp = np.array([-axis[1], axis[0]])
    hw = width / 2
    return [tuple(p0 + perp*hw), tuple(p1 + perp*hw),
            tuple(p1 - perp*hw), tuple(p0 - perp*hw)]


def add_l(space, position, angle, color="LightSlateGray"):
    """Letter L — vertical bar + horizontal foot. 2 convex rects."""
    s = 1.367
    stroke = 15 * s
    h = 90 * s
    foot_len = 60 * s
    v = [(-stroke, -h/2), (0, -h/2), (0, h/2), (-stroke, h/2)]
    f = [(-stroke, -h/2), (foot_len, -h/2),
         (foot_len, -h/2 + stroke), (-stroke, -h/2 + stroke)]
    return _multi_poly(space, [v, f], position, angle, color)


def add_k(space, position, angle, color="LightSlateGray"):
    """Letter K — vertical bar + 2 diagonal arms meeting on right edge of vertical."""
    s = 1.367
    stroke = 15 * s
    h = 90 * s
    v = [(-stroke, -h/2), (0, -h/2), (0, h/2), (-stroke, h/2)]
    junction = (0, 5 * s)
    upper_arm = _rotated_rect_verts(junction, (45 * s,  h/2), stroke)
    lower_arm = _rotated_rect_verts(junction, (45 * s, -h/2), stroke)
    return _multi_poly(space, [v, upper_arm, lower_arm], position, angle, color)


SHAPE_BUILDERS = {
    "ellipse":  add_ellipse,
    "triangle": add_triangle,
    "s":        add_s,
    "l":        add_l,
    "k":        add_k,
    # "tee" dispatched via PushTEnv.add_tee directly (already in pusht.py).
}


# ══════════════════════════════════════════════════════════════════════
# Skeleton keypoints per shape (shape-local coords, x-y pairs).
#
# Each shape has a list of spine endpoints. For fixed-dim obs, MAX_KEYPOINTS
# = 11 (S is the longest). Shorter shapes zero-pad the remaining slots.
# ══════════════════════════════════════════════════════════════════════

# Fixed N=10 per shape — no zero padding. Multi-branch proportional arc-length
# sampling so every shape has identical obs-slot semantics (no shape-ID leak
# via pad pattern). See .context/studies/2026-04-22_pusht_letter_matrix.md §D1/D2.
MAX_KEYPOINTS = 10


def _lin(p0, p1, n, include_start=True, include_end=True):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if include_start and include_end:
        ts = np.linspace(0, 1, n)
    elif include_start:
        ts = np.linspace(0, 1, n + 1)[:-1]
    elif include_end:
        ts = np.linspace(0, 1, n + 1)[1:]
    else:
        ts = np.linspace(0, 1, n + 2)[1:-1]
    return [tuple((1 - t) * p0 + t * p1) for t in ts]


_s_val = 1.367
_st_val = 15 * _s_val
_h_val = 90 * _s_val
_ft_val = 60 * _s_val
_jy_val = 5 * _s_val

# T: 5 bar pts (junction at pt 2) + 5 stem pts (excludes junction)
TEE_KEYPOINTS = (
    _lin((-60, 15), (60, 15), 5)
    + _lin((0, 15), (0, 120), 5, include_start=False)
)

# L: 5 vertical (corner at pt 4) + 5 foot pts (excludes corner)
_L_corner = (-_st_val / 2, -_h_val / 2 + _st_val / 2)
L_KEYPOINTS = (
    _lin((-_st_val / 2, _h_val / 2), _L_corner, 5)
    + _lin(_L_corner, (_ft_val - _st_val / 2, -_h_val / 2 + _st_val / 2), 5,
           include_start=False)
)

# K: 4 bar + junction + 3 upper-arm (excl junction) + 2 lower-arm (excl junction)
K_KEYPOINTS = (
    _lin((-_st_val / 2,  _h_val / 2), (-_st_val / 2, -_h_val / 2), 4)
    + [(-_st_val / 2, _jy_val)]
    + _lin((-_st_val / 2, _jy_val), (45 * _s_val,  _h_val / 2), 3, include_start=False)
    + _lin((-_st_val / 2, _jy_val), (45 * _s_val, -_h_val / 2), 2, include_start=False)
)


# S: 10 arc-length-parametrized samples along continuous midline (upper arc +
# bridge + lower arc). Precomputed from 2-ring geometry.
def _compute_s_keypoints(n=10):
    s = 1.367
    uc_y = 27*s + (38 - 22)*s * 3/14
    lc_y = -27*s
    mid_r = (22 + 38) / 2 * s
    arc_per_ring = np.radians(270) * mid_r
    up_end_xy = np.array([np.cos(np.radians(-105)) * mid_r,
                          uc_y + np.sin(np.radians(-105)) * mid_r])
    lo_start_xy = np.array([np.cos(np.radians(75)) * mid_r,
                            lc_y + np.sin(np.radians(75)) * mid_r])
    bridge = float(np.linalg.norm(up_end_xy - lo_start_xy))
    total = 2 * arc_per_ring + bridge

    def pt_at(s_val):
        if s_val <= arc_per_ring:
            frac = s_val / arc_per_ring
            theta = np.radians(165 - 270 * frac)
            return (np.cos(theta) * mid_r, uc_y + np.sin(theta) * mid_r)
        s2 = s_val - arc_per_ring
        if s2 <= bridge:
            frac = s2 / bridge
            xy = (1 - frac) * up_end_xy + frac * lo_start_xy
            return (float(xy[0]), float(xy[1]))
        s3 = s2 - bridge
        frac = s3 / arc_per_ring
        theta = np.radians(75 + 270 * frac)
        return (np.cos(theta) * mid_r, lc_y + np.sin(theta) * mid_r)

    return [pt_at(v) for v in np.linspace(0, total, n)]

S_KEYPOINTS = _compute_s_keypoints(10)


SHAPE_KEYPOINTS = {
    "tee":      TEE_KEYPOINTS,
    "l":        L_KEYPOINTS,
    "k":        K_KEYPOINTS,
    "s":        S_KEYPOINTS,
    "ellipse":  [],    # shape not in letter matrix; placeholder
    "triangle": [],
}
