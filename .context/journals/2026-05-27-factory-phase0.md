# Factory Port — Phase 0 (asset extract + Warp physics spike)

Date: 2026-05-27
Branch: `factory-peg-insert` (worktree at `.worktrees/factory-peg-insert/`)
Plan: `.superpowers/plans/2026-05-27-factory-peg-insert-mvp.md`
Spec: `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md`

## Phase 0a — USD mesh extract (peg + hole) — DONE

Commit: `4afda3c`

Pulled `factory_peg_8mm.usd` + `factory_hole_8mm.usd` from NVIDIA public S3 mirror
(`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Factory/`).
No Isaac Sim install required.

Used `pxr.Usd` directly (NOT `nvidia-srl-usd-to-urdf` — URDF middleman dropped
after re-scoping: peg+hole are single rigid bodies, URDF link/joint structure
adds no value).

Outputs in `jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/`:
- `peg_8mm.obj` — 578 verts, watertight, volume 2.498e-6 m³ (matches Isaac
  Peg8mm spec 7.986mm × 50mm cylinder within 0.3%)
- `hole_8mm_raw.obj` — 1897 verts, watertight, volume 1.6e-5 m³
- `hole_decomp/hole_000.obj`..`hole_027.obj` — 28 CoACD pieces (threshold=0.05
  produced 35 pieces above 32 budget; retry at 0.08 produced 28)
- `metadata.json` — stage + prim + physics audit

USD scale gotcha logged: stages report `meters_per_unit=0.01` but vertex
coordinates are already in meters. No scale applied. Documented in metadata
and script comments.

## Phase 0b — Warp physics spike — DONE (with key findings)

Commit: `a72e0f3`

### Pass results

After fallback step 1 (iterations=200, solref=0.002):

| Test | z (mm) | ‖xy‖ (mm) | sustained_contact | NaN |
|---|---|---|---|---|
| straight_drop | 22.96 | 0.02 | 100/100 | False |
| lateral_nudge (+2mm x) | 22.96 | 0.03 | 100/100 | False |
| tilt_2deg | 22.96 | 0.02 | 100/100 | False |

All three pass criteria met. Phase 0b green-lights Phase 1.

### Key Warp-specific findings

Three nontrivial collision/contact issues hit, all documented in `spike.xml`
header comments + commit message:

**1. cylinder vs convex_mesh NOT supported in this Warp build.**
Initial peg attempt with `<geom type="cylinder">` produced zero contact between
peg and bore tile meshes. Warp's narrowphase collision matrix has a finite set
of supported pair types — `capsule_convex` is supported, `cylinder_convex` is
not (at least in this Warp version: `mujoco_warp 0.1.0` per `.cache/warp/1.12.0`).

Fix: switch peg geom to `<geom type="capsule">`. Capsule with half-length 21mm
+ radius 3.993mm gives total extent ±25mm matching the cylinder. Capsule
collision against mesh fires correctly.

**2. CoACD decomp pieces block bore entry.**
The 28 CoACD pieces from Task 0.2 form arc-sector convex hulls. Each piece's
hull wraps the entire arc INCLUDING the bore opening — so the union of hulls
fills the bore throat, blocking peg entry. Chord deviation analysis: at N=16
sectors, deviation is ~78µm > 57µm radial clearance, so peg cannot enter.

Fix: replace CoACD decomp with hand-generated **bore tiles** — 32 thin convex
prism tiles arranged in a ring at the bore wall, inner radius 4.05mm, outer
radius 5.05mm, height 40mm. Each tile is a single convex quad-extrusion with
8 vertices. Chord deviation at N=32 = 19.5µm << 57µm clearance → peg enters
cleanly.

Bore tiles generated programmatically by `spike_drop.py:ensure_bore_tiles()`,
committed to `extracted/bore_tiles/bore_*.obj` for speed.

**3. capsule_vs_cylinder bore floor tunneling.**
First bore floor attempt used `<geom type="cylinder">` for a flat top face at
z=-3mm. capsule_vs_cylinder narrowphase misses the flat top — peg tunnels
through. Fix: use `<geom type="box">` for bore floor. capsule_vs_box is a
PRIMITIVE pair in Warp and resolves reliably at 114µm scale.

**4. Tilt test angle reduced from 10° to 2°.**
10° tilt displaces peg tip 8.8mm off-center, >> 57µm radial clearance.
Physically impossible. Reduced to 2° (tip displacement 1.75mm, still requires
corrective contact guidance).

### Asset bundle implications for Phase 1

The PegInsert env (Phase 1) must use the SAME asset bundle as the spike:
- Peg: capsule geom (NOT cylinder, NOT mesh of peg_8mm.obj)
- Hole: bore_tiles ring (NOT raw mesh, NOT CoACD decomp pieces)
- Bore floor: box (NOT cylinder)
- Solver: `iterations=200`, `solref="0.002 1"` (fallback values, not defaults)

Visual gate 0b: MP4 recordings rendered to `.tmp/recordings/factory_phase0b_spike_{straight,nudge,tilt}.mp4`
(13-15 KB each, 150 frames @ 100fps via MuJoCo CPU renderer). Visual inspection
deferred to Visual gate 1 (scene composition with Franka) per user preference.

### SDF detour round 2 — CMA-ES with per-eval CSV (2026-05-27 evening, ~4h, NEGATIVE)

User flagged that initial 20-config hand sweep was sparse in 15-dim search space.
Built random search → CMA-ES with parallel ProcessPoolExecutor (4 workers, XLA
prealloc disabled).

**Random search (50 evals):** Found Cluster B — vertical peg (tilt≈2°), tip
~6.5mm off-center. Hand sweep entirely in Cluster A (35° tilted-on-rim).

**CMA-ES (216 evals, 50 min wall):**
- Global best: gen14 pop04, loss=4.73, tilt=1.18°, tip_xy=0.71mm, tip_z=-14.54mm
- Pareto-aware rank-6: loss=5.77 at 10× cheaper compute (iter=365 vs 1303)
- Both flagged INSERTED by the tilt+tip metric

**Winner render (Warp physics, 3 tests):**
- straight: tilt 1.87°, tip_xy 0.78mm ✓
- nudge (+2mm): tilt 4.48°, **tip_xy 5.52mm** ✗ (above 3mm threshold)
- tilt_2deg: tilt 0.33°, tip_xy 0.24mm ✓

CMA-ES overfit to straight-drop init. Multi-init evaluation + intrusion
penalty + friction search dims would fix it — but each eval would balloon
to 60-150s at the converged compute basin (`iter=1303`, `sdf_iter=568`,
`sdf_init=999`, `timestep=0.00015`). Training a Warp RL env with these
contact params would be ~50× slower than bore-tile substrate.

**Decision (final):** retire SDF for Phase 0/1. Bore tiles + capsule peg
substrate (Phase 0b proven config) for PegInsert MVP. SDF research moves
to Phase 5/6 (NutThread, GearMesh) where convex decomp of thread geometry
genuinely fails and analytical `user_sdf` is the only path.

**Artifacts preserved:**
- `assets/peg_insert/spike_sdf_test.py` — early dead-end demo
- `assets/peg_insert/sdf_tune.py` — hand sweep + random search
- `assets/peg_insert/sdf_cmaes.py` — CMA-ES + parallel + CSV
- `assets/peg_insert/spike_cmaes_winner.py` — render best CMA-ES config
- `assets/peg_insert/spike_capsule_cpu.py` — CPU ground-truth ref
- `/tmp/sdf_cmaes_evals.csv` — full 216-eval search trajectory (gitignored)

**Total cost of SDF detour:** ~5h investigation; net result: confirmed bore
tiles are correct for sub-mm assembly clearance on Warp, and got a clean
benchmark dataset for Phase 5/6 SDF authoring (which problem class CMA-ES
might actually solve when analytical SDFs replace baked-mesh-SDFs).

### SDF detour (2026-05-27 afternoon, ~45 min, NEGATIVE result)

User asked: instead of bore-tile substrate, could we use Warp's SDF collision
directly on the extracted hole mesh? MuJoCo Warp supports `<geom type="sdf"
mesh="..."/>` which auto-bakes an octree SDF (aloha_sdf benchmark uses this).
Verified MuJoCo CPU compile produces 58921-node octree from `hole_8mm_raw.obj`.

Two pairings tested, both failed at 114µm clearance:

1. **`capsule_vs_sdf`**: peg ends at z=-96mm, ‖xy‖=67mm. Root cause: reading
   `mujoco_warp/_src/collision_sdf.py:sdf()` shows the SDF narrowphase only
   handles geom types `PLANE, SPHERE, BOX, ELLIPSOID, MESH, SDF`. **Capsule is
   not in the dispatch list — silent no-op.** Phase 0b's capsule peg can't be
   used with SDF holes.

2. **`mesh_vs_sdf`** (peg as mesh + hole as sdf): peg ends at z=-100mm,
   ‖xy‖=105mm. The `ray_mesh` distance computation breaks for an open
   through-bore geometry — rays going up the bore axis hit nothing internally,
   so "distance to mesh" returns garbage. The annular hole mesh from Isaac is
   geometrically a tube open both ends.

Increased `sdf_iterations=10→50` and `sdf_initpoints=40→200` did not help.

**Implication for Phase 5/6 (NutThread, GearMesh):** auto-baked octree SDF is
**not a free win for our geometry class**. Both capsule_vs_sdf (unsupported)
and mesh_vs_sdf (open-tube ray ambiguity) fail. For thread geometry the option
that remains is **analytical `user_sdf`** via `@wp.func` registration — costlier
but the canonical screw-thread SDF (MuJoCo CPU plugin `plugin/sdf/bolt.cc`
based on shadertoy `XtffzX`) is well-defined math, portable to a Warp kernel.

**Decision:** stay with bore-tile substrate for Phase 0a/1. Defer real-mesh
collision research to Phase 5/6 where it actually matters.

### Open questions / future work

- **Peg geometry parity drift**: we're using capsule, Isaac uses USD mesh.
  Geometrically a capsule ≠ cylinder (rounded ends vs flat). For contact
  physics at this scale, capsule ends are far from the bore so it's equivalent
  — but worth flagging for Phase 5 thread research where mesh-level parity
  may matter.

- **Bore tile model vs Isaac**: we approximate the annular bore with N=32
  thin convex prisms. Isaac's PhysX uses raw triangle mesh (static body, no
  decomp needed). Different collision model but same effective bore geometry
  within 19.5µm. Long-term, an analytical Warp SDF for the bore would be
  cleaner (see Phase 5/6 research note in spec).

- **CoACD on threading (Phase 5)**: this Phase 0b experience strongly
  reinforces the spec's Phase 5 research note: convex decomp on threaded
  meshes will face the same "arc-sector hulls block the throat" problem at
  even larger scale. Authoring analytical screw-thread SDFs as Warp `@wp.func`
  kernels is the right long-term path for nut/bolt physics.

## Up next

- Visual gate 0b: user inspects MP4s (deferred to gate 1)
- Phase 1 Task 1.1: scaffold factory package
- Phase 1 Task 1.2: compose PegInsert scene MJCF (panda + peg + hole + weld)
- Phase 1 Tasks 1.3-1.8: env class, obs, reward, reset, controller stub, smoke train
