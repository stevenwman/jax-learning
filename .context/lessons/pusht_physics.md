# Push-T Physics & Geometry Lessons

Physics-engine and shape-construction lessons from the planar pushing benchmark (`jax_rl/envs/manipulation/push_env.py` and `jax_rl/envs/manipulation/pusht/`). For obs/action design see `pusht_design.md`; for rewards see `pusht_rewards.md`; for eval/metrics see `pusht_eval.md`.

---

## S-shape tunneling with default pymunk solver (2026-05-11)

**What happened:** Visually observed agent half-clipping through S-shape
boundary in MPPI rollout videos. Single-shot eval cov=0.0% on an episode
where seed initialized agent already overlapping S geometry; policy
couldn't recover (no contact-feasible escape).

**Root cause:** S has **18 convex wedge polys** (per `_compute_s_keypoints`
+ `_ring_polys` in `shapes.py`) vs T/L/K's 2-3. pymunk defaults
(`iterations=10`, wall `radius=2`, `collision_slop=0.1`) handle a few
contacts well but choke on many fast small ones. The two overlapping
270° fat rings of S produce ~18 small contact normals.

**Diagnosis pattern:** If eval cov has a bimodal distribution per seed
(most ~90%, a few 0%), suspect physics-init clip, not policy failure.
Re-record with a fresh `--seed` to confirm. Multi-seed eval averages
hide this — single-episode logs reveal it.

**Knobs (untested, in priority):**
1. `space.iterations = 50` (default 10) — solver stiffness
2. Wall `radius = 5` (default 2 at `pusht.py:672-675`) — thicker walls
3. `collision_slop = 0.01` (default 0.1) — tighter penetration tol
4. `dt = 0.005` (default 0.01) + 2× substeps — best accuracy, 2× slower
5. Poly `radius=2` in shape builders — rounded-corner CCD on block

Caveat: any physics change breaks comparability with prior SAC V2 letter
matrix (trained on default physics). Either retrain SAC + TDMPC2 on
"v3 physics" or label TDMPC2 row separately.

---

## Cylinder-Box Collisions Require MuJoCo Warp (2026-04-17)

**What happened:** Built a shape-agnostic push env with cylinder pusher + box block geoms. `mjx.put_model(m, impl="jax")` raised:

```
NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.
```

**Root cause:** MJX JAX backend never implemented cylinder-vs-box narrowphase. Only sphere-sphere, sphere-box, sphere-capsule, box-box, and a few others.

**Fix:** Use `impl="warp"`. Warp's CCD (`naccdmax`-sized buffer) handles all geom pairs via GJK+EPA.

**Lesson:** Any env with cylinder pusher or tool interacting with non-sphere objects is Warp-only. This matches [mjx.md](mjx.md) §"MJX can't load all MJCFs". Don't plan for JAX-backend fallback — cylinder-box comes up naturally in manipulation.

**Workaround if Warp unavailable:** replace cylinder with sphere (radius ≈ height/2). Sphere-box is supported in JAX backend. Loses the flat-top contact profile but OK for most RL.

---

## Contact Penetration: Tighten solref When Soft Defaults Don't Fit (2026-04-17)

**What happened:** Early pusher-block contacts had visible penetration in rendered video — pusher disc overlapped block geoms by several mm for 3–5 frames before separating.

**Root cause:** MuJoCo default contact is `solref="0.02 1"` — 20ms time constant, critically damped. At `sim_dt=0.002` that means ~10 sim steps to resolve a penetration. Between resolutions, visible interpenetration.

**Fix:** Stiffen both geoms:

```xml
<geom solref="0.004 1" solimp="0.98 0.995 0.0005 0.5 2"/>
```

- `solref="0.004 1"` → 4ms time constant ≈ 2 sim steps. Visually clean.
- `solimp="0.98 0.995 ..."` → nearly rigid (default max is 0.95; bump to 0.995).
- Bump `<option iterations="50" ls_iterations="10"/>` to give Newton solver more iterations at the stiffer problem.

**Tradeoff:** stiffer contacts are slightly more expensive per step. Measurable but small (few percent sps hit). Worth it for manipulation where contact is the whole point.

**Lesson:** Default MuJoCo contact is tuned for locomotion (feet on floor, soft enough to avoid hard chatter). Manipulation wants hard contact with fast separation. Always check the contact quality in rendered video before assuming physics is correct.

---

## Slide-Joint Body Pos Is an Offset, Not a Starting Position (2026-04-17)

**What happened:** Wrote `qpos = [pusher_x, pusher_y, block_x, block_y, block_yaw]` in reset(), expecting pusher/block to appear at those world coordinates. On render, pusher was visibly *outside* the wall on the left — despite qpos values being inside the valid range.

**Root cause:** Pusher body had `<body pos="-0.15 0 0.015">` in the MJCF. Slide joints are **additive to body pose**: world_x = body_pos_x + qpos[0]. So qpos=-0.19 produced world_x = -0.15 + -0.19 = -0.34, past the wall at ±0.3. Same for block (body_pos 0.05). Visible confusion: policy obs said pusher at -0.19, physics had it at -0.34.

**Fix:** Set body pos to origin (`<body pos="0 0 0.015">`) so qpos directly = world XY. No offset bookkeeping.

**Lesson:** When using slide joints to express "XY position" of a body, always anchor the body at origin. Any nonzero `<body pos>` becomes a hidden offset baked into every qpos read/write. Check with `d.xpos[body_id]` vs `d.qpos[joint_qposadr]` during a forward pass — they should match for a pure slide body. Silent visual-only bug otherwise.

---

## Pymunk CoG Offset → `block.position` ≠ Block Goal Pose (2026-04-19)

**What happened:** Calibration script tried to set the T at the goal pose with `block.position = (256, 256), block.angle = π/4`. Coverage came out at 0.298, not 1.0. Debugging revealed `block.center_of_gravity = (0, 45)` in body frame. When you set `body.position = X` for a body with non-zero CoG, X is the **body origin location**, not the CoG location. World CoG = body.position + rotate(CoG, angle). Setting position then angle (per `_set_state` order) made the block end up at world (287.8, 269.2) — 32 pixels from where I asked.

**True identity state** (gives coverage 0.9995): `[agent_x, agent_y, 224.2, 242.8, π/4]` for `reset_to_state`. Discovered by 0.5-px grid search around (256, 256).

**Implications for any code using `block.position`:**
- Distance shaping `||block.position - goal_pose[:2]||` is measuring distance between body origin and goal *body origin*, NOT geometric centers. Off by ~30 px on the constant CoG offset, so still gives a useful gradient — but interpret with care.
- For exact pose matching (e.g. setting up a calibration test), use `pymunk_to_shapely(body, shapes).centroid` to get the actual world centroid.
- For RL training, the bias is a constant offset, so the policy learns to compensate. Not a hard bug, but a footgun for diagnostics.

**Lesson:** any pymunk body with non-zero `center_of_gravity` behaves "weirdly" under direct position+angle assignment because rotation pivots around CoG, not origin. For T-shapes, hexagons, asymmetric polygons — read `body.center_of_gravity` first; never assume `body.position` = world geometric center.

---

## Pymunk Shape Construction — Decompose Concave Letters Into Convex Rings (2026-04-20)

**What happened:** needed 4 new block shapes (ellipse, iso-triangle, letter S, letter U) for cross-shape benchmark. pymunk requires **convex** `Poly` geoms; letters S / U are concave. Early attempts at S used rotated rectangles along a bezier centerline — produced visible "fins" at hook tips (tangent rotates fast near tight curvature → quads poke perpendicular to curve).

**Fix:** build concave letters as sets of **annular sectors** (fat C's). Each sector decomposed into wedge quads via:

```python
def _ring_polys(center, inner_r, outer_r, theta_start, theta_end, n):
    thetas = np.linspace(theta_start, theta_end, n + 1)
    return [[
        (cx + inner_r * cos(t0), cy + inner_r * sin(t0)),
        (cx + outer_r * cos(t0), cy + outer_r * sin(t0)),
        (cx + outer_r * cos(t1), cy + outer_r * sin(t1)),
        (cx + inner_r * cos(t1), cy + inner_r * sin(t1)),
    ] for t0, t1 in zip(thetas[:-1], thetas[1:])]
```

- **Letter S**: two 270°-arc rings, rot-180 symmetric, overlapping in middle strip → smooth uniform-curvature S with 18 quads total. Thickness `outer_r - inner_r` controls stroke width; `inner_r` controls hook-interior radius (must be ≥ pusher radius for reachability).
- **Letter U**: 180° half-ring + 2 rectangles (arms).
- **Ellipse / triangle**: single convex `Poly` each (no decomposition needed).

Every wedge is tangent to a circle → uniform curvature across the shape, no kinks or tangent-mismatch artifacts.

**Design knobs:**
- `n_per_ring` = 10 (30°/wedge) looks clean; 8 acceptable, 6 chunky.
- Scale shape so `inner_r > pusher_radius + margin` (pusher 15 px, inner_r ≥ 22 → 7 px margin).
- For letter hooks: trim the END wedge of each ring (reduce `theta_end` by ~30°) for tapered tip instead of squared terminal.

**Lesson:** when pymunk's convexity constraint bites, reach for annular sectors before bezier ribbons. Circle-tangent decompositions produce much cleaner curves per poly count than curve-sampling approaches.

---

## Shapely TopologyException on Overlapping Convex Pieces (2026-04-20)

**What happened:** after adding multi-piece block shapes (S = 18 overlapping quads, U = 10+2 polys), `_get_coverage` crashed:

```
shapely.errors.GEOSException: TopologyException: side location conflict
at 343.55936998684041 274.20135325161755. This can occur if the
input geometry is invalid.
```

**Root cause:** `pymunk_to_shapely` builds a `sg.MultiPolygon` from the convex pieces and calls `.intersection(goal_geom).area`. When pieces overlap (S has two overlapping rings by design), the MultiPolygon has self-intersections at overlap boundaries. GEOS rejects the geometry as invalid.

**Fixes (in order of robustness):**
1. **Union first, then intersect:** replace `MultiPolygon` with `unary_union(polygons).buffer(0)` — heals overlap seams into a single valid polygon. `buffer(0)` is a known GEOS idiom for fixing self-intersecting geometries.
2. **Catch + stub:** try/except around `.intersection` → return 0. Fine for vibes-only tests where coverage doesn't drive learning. Used in `tools/record_pusht_shapes.py`.
3. **Redesign shape**: decompose such that convex pieces don't overlap. Doubles the convex-piece count and breaks the elegant ring construction.

**Lesson:** MultiPolygon is not the same as "polygon with holes" in GEOS. Overlapping convex pieces → invalid MultiPolygon. When porting T-specific coverage metric to multi-shape envs, wrap union with `buffer(0)` or catch the exception explicitly. Don't assume valid geometry from valid pymunk shapes.
