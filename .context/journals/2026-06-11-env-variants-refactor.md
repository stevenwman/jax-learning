# 2026-06-11 — Go2 env variants-as-data refactor + OSC extraction

Overnight autonomous refactor on branch `go2-osc-impedance` (worktree),
commits `af4bf7f..d15d026` (11 commits). Plan:
`.superpowers/plans/2026-06-11-go2-env-variants.md`. Spec:
`.superpowers/specs/2026-06-11-go2-env-variants-design.md`.

## Motivation (audit findings)

The Go2 Warp env family had grown to 29 named envs via four compounding
pathologies:

1. **Patch-chains** — each new variant was a subclass + a config-factory
   function copying and tweaking the previous one's overrides
   (`go2_warp_osc_joystick.py` → `go2_warp_osc_var_impedance.py` →
   `go2_warp_osc_rough.py`). Answering "what config does env X actually
   train with" required tracing 3–4 files.
2. **God function** — `mjx_backend.py` carried a ~175-line Go2 closure
   block, one hand-written registration per env, with research rationale
   buried in comments there instead of next to the env declarations.
3. **Silent preset fallback (the real bug)** — preset getters only knew
   the older Go2 names. Unknown `Go2Warp*` names fell through to the bare
   base config: **all OSC envs silently trained with NO domain
   randomization (`reset_mode` default, DomainRandWrapper never applied)
   and `eval_every_n_episodes=5000` (≈zero mid-run evals on short runs)**.
   Weeks of OSC runs trained under a different regime than intended.
4. **Half-extraction** — the 2026-06-09 composition refactor moved
   controller *selection* into components, but the OSC mechanics
   (`_run_osc`, `_feet_in_body`, `_compute_nominal_foot_body`) and all
   gains state still lived on the host env, planted there by the
   component at setup.

## What landed (per commit)

| Commit | What |
|---|---|
| `af4bf7f` | **Variants-as-data table** — `jax_rl/envs/locomotion/go2_warp_variants.py`: `EnvVariant` (config callable via `go2_config()` builder + host class name + train overrides + notes), 29 entries, pinned equal to legacy configs by transitional equality tests. |
| `95509a5` | Guard cross-axis `go2_config` kwargs (e.g. OSC gains on a joint-PD env raise); hoist OSC base gains to module constants. |
| `d303cbc` | `mjx_backend.py` Go2 block → a 6-line loop over `GO2_WARP_VARIANTS` + `_resolve_cls`. Research rationale (Kp-sweep design, hard-kick ladder, Jᵀ ablation, zero-shot-physical protocol, rough-hfield notes) migrated into the variants file. |
| `5ca7adc` | All 6 preset getters resolve Go2 names via `_resolve_go2_variant`; **unknown `Go2Warp*` names now RAISE** (`ValueError` with the known-names list). The silent fallback is dead. |
| `47c2df0` | `_resolve_go2_variant` splits algo-vs-train override keys; named error when a variant carries algo overrides but the getter has no algo config (PPO). |
| `b0df7f1` | **Behavior change (see below)** — 21 OSC/physical/rough variants get `_DR_TRAIN = {reset_mode: per_step, eval_every_n_episodes: 500}`. Also fixed stale `--eval-every` help text in the 5 off-policy train scripts. |
| `6596a2b` | **Legacy modules deleted**: `go2_warp_osc_joystick.py`, `go2_warp_osc_var_impedance.py`, `go2_warp_osc_rough.py`. `WarpOscJoystick` / `WarpOscVarImpedance` classes are gone — OSC envs are config presets over `WarpJoystick`. Snapshot pin: `tests/data/go2_warp_variants_snapshot.json`. |
| `fba9bad` | Regen env-presets + CLI reference docs; `gen_env_presets.py` resolves Go2 names through the getters (`_with_go2`) so the 29 variants stay listed. |
| `054ace4` | Snapshot regen one-liner in test docstring + cleanup. |
| `14a2f6f` | **OSC extraction completed** — `_run_osc` / `_feet_in_body` / `_compute_nominal_foot_body` + all gains state now live on the `OSC` component in `go2_warp_components.py`; the host env plants nothing. `mud_eval` callsite updated. |
| `d15d026` | Controller read-only env contract documented; stale comments fixed. |

Net: every Go2 Warp env is now ONE declaration in `go2_warp_variants.py`.
Adding a variant = adding one `EnvVariant(...)` entry.

## ⚠️ BEHAVIOR CHANGE — DR cut date 2026-06-11

**21 OSC / physical-motor / rough variants now default to
`reset_mode="per_step"` (DomainRandWrapper DR ON) +
`eval_every_n_episodes=500`.** Before this, the preset fallback bug meant
OSC envs trained with NO DR and almost no mid-run evals.

**Old OSC runs (anything before 2026-06-11) are NOT comparable to new
ones.** The joint-PD benchmark family (`Flat`, `TorqueSpeed`, `NoAccel`,
`Unitree`, joint HardKick, PosTrackProto) is unchanged; curriculum keeps
its historical per_step-only train dict. If OSC baselines matter for a
comparison, retrain them under the new defaults (TODO).

## Equivalence-gate methodology

- **Configs**: transitional equality tests pinned every variant's config
  byte-equal to legacy factory output BEFORE the legacy modules were
  deleted; then frozen as `tests/data/go2_warp_variants_snapshot.json`.
- **OSC extraction**: GPU/Warp is not bit-reproducible across processes
  (`.temp/REFACTOR_SANITY_CHECKS.md`), so the gate was: measure the
  cross-process Warp noise floor with **identity control** (same code,
  two processes), then show the refactor's old-vs-new trajectory
  deviation sits BELOW that floor. Refactor deviation is
  indistinguishable from run-to-run Warp noise.

## Smoke run (post-refactor end-to-end)

`Go2WarpOscFlatSoftPhysical`, FastSAC, 200k steps, 256 envs, `--eval-every
50` override (log `.temp/logs/smoke_osc_softphysical_1130.log`, ckpt
`checkpoints/20260611_113024_fast_sac_go2warposcflatsoftphysical_seed0`):

- **Best eval 87.5 ± 26.0** (@ 848 eps); final eval 32.3 ± 52.2.
- 929 episodes, online avg return (last 100 eps) 17.3; 17 evals fired
  (the eval cadence actually works now).
- Throughput ~600–690 sps at end (291 s wall) — healthy for OSC +
  physical motor + per_step DR at 256 envs.
- Q diagnostics sane: bias −0.19, RMSE 1.42, corr 0.72.

200k is a smoke (full runs are 5M+); the point is DR + eval cadence +
training all function end-to-end on a pure-preset OSC env.

## Test suite state

**836 passed / 46 skipped** on CPU + GPU OSC tests green. 6 failures are
PRE-EXISTING, not from this work:

1. `tests/test_go2_warp_curriculum_env.py::test_generated_scene_file_written`
   (GPU-marked) — 1 failure.
2. `tests/manipulation/factory/test_factory_action_chain.py` — 2 failures
   (`denormalize` got unexpected kwarg `unidirectional_rot`).
3. `tests/test_pusht_parity.py` — 3 failures (pymunk
   `add_collision_handler` API drift / `gym_pusht` import).

Filed in `.context/TODO.md`.

## Afternoon: first DR-era retrain (VarDampingAxis) + OOM saga

Steven picked `Go2WarpOscVarDampingAxisFlatPhysical` as the first retrain under
the new per_step-DR defaults (5M steps, 256 envs, seed 0, FastSAC).

**OOM at paper config.** The 4.19M-slot buffer (36-d action → ~6.4GB with dual
critic obs) + per_step DR wrapper + Warp does NOT fit on the 16GB GPU at any
`XLA_CLIENT_MEM_FRACTION` (0.75 default fails allocating 2.28GiB, 0.55 fails at
1.12GiB, 0.65 at 576MiB — all at init, 0 steps). Fraction tuning exhausted →
config change required.

**Decision (Steven): `--buffer-size 2097152` (2M) is the DR-era retrain
standard**, keeping 256 envs; launch with `XLA_CLIENT_MEM_FRACTION=0.65`. All
future ladder rungs use the same buffer for HP comparability. 2M still covers
40% of a 5M-step run's experience.

**Run healthy**: obs 72-d (last_act grows to 36), buffer 2,097,152 confirmed in
banner, first eval **97.2 ± 48.3 @ 500 eps**, wandb `go2-osc-impedance`.
Log: `.temp/logs/retrain_vardampaxis_5M_seed0.log`.

**Process note (cwd trap, again):** two launch attempts ran in the MAIN repo
instead of the worktree (no `cd` prefix; cwd does not reliably persist between
shell calls) and died with `Env not found` — masquerading as extra OOM data
points until the traceback was actually read. The `feedback_worktree_cwd`
memory rule exists for exactly this; `cd` EVERY command in worktrees.

## VarDampingAxis DR-era retrain: COMPLETE

5M steps in 23.5 min (~3,500 sps — the earlier 650-sps smoke estimate was
eval-cadence-bound, not compute-bound). **Best/final eval 280.5 ± 5.9** with
per_step DR active — matches the historical no-DR Go2 FastSAC range (~280),
i.e. the 8-axis model DR cost essentially nothing in final performance for
this variant. Eval curve: 97 → 91 → 166 → 255 → 262 → 267 → 275 → 270 → 273 →
277 → 280.5 (monotone-ish after 1k eps, tight ±4-10 variance from 2k eps).
No per-eval GPU memory drift (12,748 → 12,750 MiB across evals 5→8) — the
`--xla_gpu_enable_command_buffer=` mitigation holds; eval cadence no longer
needs memory budgeting.
Ckpt: `checkpoints/20260611_121135_fast_sac_go2warposcvardampingaxisflatphysical_seed0`
(wandb 4me6cddf). NOTE: a 2026-06-09 ckpt of the same env exists from the
pre-DR era — do not confuse them; the new one supersedes for DR-era comparisons.

## Newton force probe — the analytic mud is the WRONG physics for the bog

Question (from the mud previews): how does the force the robot feels in the analytic
field compare to Newton MPM? Built `projects/mud_eval/probe_forces.py` (read-only tap
of `body_sand_forces`, the MPM→rigid coupling force = impulse/sim_dt, on the calf
bodies; physics untouched). Ran the VarDampingAxis DR ckpt spawned in thick mud (90
frames, co-step 250 Hz). Gating video `recordings/probe_forces.mp4`.

**Result — Newton bogs the robot with ~100× LESS force than the analytic field:**

| field | total leg force | % bodyweight (150 N) | robot |
|---|---|---|---|
| analytic 1× (Isaac default) | 50 N | 34% | walks fine |
| analytic 8× (my "bog" probe) | 438 N | 294% | bogs |
| **Newton thick mud** | **3.4 N mean / 9.2 N peak** | **2% / 6%** | **bogs** |

Per calf: 0.5–1.2 N mean. The robot stands at z≈0.27 (normal height — dense ρ2000 mud
SUPPORTS it near the surface, calves only shallowly submerged) and is stuck at y≈0.59
the whole run despite a forward command.

**Mechanism — the two fields fail the robot in OPPOSITE ways:**
- Analytic (my port): mud = SOLID MJX floor + an added opposing foot force. The floor
  still gives full normal reaction + friction, so push-off always works; to bog the
  robot I had to crank resistance to ~3× bodyweight (brute force).
- Newton MPM: mud IS the substrate. The robot bogs not from large force but from
  TRACTION/PROPULSION LOSS — pushing off a yielding granular medium displaces mud
  instead of generating thrust (sand-treadmill). Force stays tiny; the robot just
  can't get purchase.

**Implication for train-on-mud:** matching the analytic field's force magnitude to
Newton is the WRONG calibration target — the analytic field models "overcome
resistance," Newton's bog is "loss of foothold." A force-coefficient sweep can never
make the analytic proxy reproduce the Newton bog. To proxy the real phenomenon you'd
model the SUBSTRATE yielding (reduce effective ground friction / normal-reaction /
add foot sink under the foot ∝ mud depth), NOT add an opposing wrench. The analytic
field still trains a real (different) skill — disturbance rejection — but it is NOT a
mud-traversal proxy. Newton MPM stays the only valid mud test. (Caveat: single ckpt,
single depth, quasi-static-ish; the force during a successful dynamic stride could be
higher — but this policy never achieves one in thick mud, which is itself the point.)

## Autonomous mud-training iteration (multi-hour, 2026-06-11 eve)

Metric: final y on Newton thin→thick traverse, max-forward (vx=1.5), 750 frames
(lower = deeper into mud = better). Thick y0-1, medium y1-2, thin y2-3.

### R0 — 1× Isaac-coeff mud DR (depth U[0.03,0.22]), both control arms
| arm | flat eval | Newton final y | posture z |
|---|---|---|---|
| var-impedance (per-axis + damping, 36-d) | 279.3 | **1.29** (deep medium) | ~0.30 |
| joint-PD (12-d) | 274.5 | **1.62** (early-mid medium) | ~0.24 (crouched) |

Var-impedance penetrates ~0.33 m deeper + taller posture — the mud_eval headline
(compliant beats stiff in mud) SURVIVES mud training. But BOTH cooked: neither reaches
thick (y<1); both bog in medium. Mud DR at 1× cost ~nothing on flat (both ~275-280),
consistent with the weak-disturbance force finding (analytic 1× = 34% bodyweight).
→ triggers R1 per the decision rule.

### R1 — mud DR expanded to span 1→4× Isaac coeffs (f 14→60, c1 9→40, c2 6→28)
| arm | flat eval | Newton final y | vs R0 | pitch |
|---|---|---|---|---|
| var-impedance 1-4× | 276.0 | **0.68 (THICK mud!)** | R0 1.29 → 0.68 | sustained +12-18° fwd |
| joint-PD 1-4× | (training) | — | R0 1.62 → ? | — |

**BREAKTHROUGH:** the wider/stronger mud DR pushed var-impedance from bogging in
medium (y1.29) all the way into THICK mud (y0.68) — first policy to reach thick on the
thin-first maxfwd traverse. So coeff-scaling DR DOES transfer to Newton, at least
partway — despite the force-physics mismatch, exposing the policy to stronger analytic
resistance taught it to drive through soft terrain. Flat perf barely moved (276 vs 279).

**Pitch-forward CONFIRMED (user's observation):** the successful var-4× policy holds a
+12-18° nose-down lean the ENTIRE traverse (not a transient) — a deliberate lean-and-
drive strategy under the max-forward command. Posture stays tall (z~0.30). Open
question: is this lean optimal, or a max-command lunge artifact? Tests queued:
(a) command-sensitivity vx=0.5 vs 1.5; (b) reward shaping toward slow-firm planting.

### R1 joint-PD result — DR scaling SPLITS the controllers
| arm | R0 (1×) | R1 (1-4×) | pitch | posture |
|---|---|---|---|---|
| var-impedance | 1.29 | **0.68 (thick) ✓** | +15° fwd (drive) | z~0.30 |
| joint-PD | 1.62 | **2.78 (stuck at thin edge!) ✗** | -9° back | z~0.23 crouch |

joint-PD 4× hit the first mud (y2.78) and FROZE for 670 frames — leaning back,
crouched, never crossing thin. Stronger mud DR made fixed-gain joint-PD MORE timid
(R0 1.62 → R1 2.78, a regression), while it broke variable-impedance THROUGH to thick
(1.29 → 0.68). Clean support for the variable-impedance thesis: only stiffness-
modulating control converts harder-mud training into traversal ability; joint-PD learns
brace-and-stall. Pitch signs mirror it — success leans INTO mud (+), failure leans away
(-). Caveat: n=1 seed each. Best policy so far: var-impedance 1-4× (y=0.68).

### Command-sensitivity probe — var-4× at vx=0.5 vs 1.5 (pitch artifact test)
| command | final y | depth | pitch | posture z |
|---|---|---|---|---|
| vx=1.5 (max) | 0.68 | thick | +15° | 0.30 |
| vx=0.5 (half) | 1.18 | medium | +11° | 0.32 |
Slower command → LESS penetration (not more); forward pitch PERSISTS at half speed
(+11°). So the lean-forward is the policy's real mud-driving mechanism (lean→push),
command-scaled, NOT a max-command artifact. User's "slower=firmer=better" intuition
does NOT hold for THIS policy (lean-and-drive trained at high command). → motivates R2.

### R2 — reward shaping toward FIRM PLANTING (the user's hypothesis, direct test)
New variant Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xFirm: var-impedance + 4× mud DR
+ feet_slip -0.1→-0.6 (firm planting) + orientation -5→-8 (discourage lean). Velocity
tracking left default. Head-to-head vs lean-and-drive var-4× (y=0.68). If deeper →
firm planting wins (validates intuition); if shallower → lean-and-drive was better.

### R2 RESULT — firm-planting reward shaping SOLVES the traverse 🎯
| policy | final y | depth | posture z | pitch |
|---|---|---|---|---|
| R1 var-4× lean-and-drive | 0.68 | deep thick (didn't exit) | 0.30 | +15° |
| **R2 var-4× + FIRM PLANT** | **−1.38** | **THROUGH the whole gradient, out the far side** | 0.33 (tall) | +4° at exit |

Full ladder (thin-first maxfwd vx=1.5, lower y = deeper; mud y0-3, y<0 = cleared):
R0 var 1.29 / joint-PD 1.62 → R1 var **0.68** / joint-PD **2.78 (regressed)** →
R2 var+firmplant **−1.38 (CLEARED THE MUD)**.

**Winning recipe = variable-impedance control + 4× mud DR + firm-planting reward**
(feet_slip -0.1→-0.6, orientation -5→-8). The policy crosses thin→medium→thick and
walks out, taller posture than lean-and-drive.

**Mechanistic insight (connects to the force-probe finding):** the Newton bog is
TRACTION LOSS on a yielding substrate (not resistance — force probe showed Newton bogs
at ~3N). Pure DR coeff-scaling (resistance) only got partway (var-4× to y0.68). The
feet_slip penalty directly targets TRACTION — it trains the foot to plant firmly and
not slide, which is exactly what the yielding mud destroys. Tell: the firm-plant policy
STILL pitches hard (+25° at the densest thick mud, f530) — MORE than lean-and-drive —
yet succeeds, because firm planting gives the foot grip to convert that lean into
forward thrust instead of slipping. So firm planting didn't remove the lean; it made
the lean EFFECTIVE. The user's "plant feet firmly" intuition was right, and for the
right reason (traction, the actual failure mode).

Caveats: n=1 seed per arm; the orientation vs feet_slip contributions aren't isolated
(pitch data suggests feet_slip dominates). Next: seed-replicate the firm-plant win;
then ablate feet_slip vs orientation. Best ckpt:
checkpoints/20260612_002529_fast_sac_go2warposcvardampingaxisflatphysicalmuddr4xfirm_seed0

### R3 — seed replication of firm-plant: PARTIAL (real but stability-limited)
| firm-plant | deepest y | final y | end |
|---|---|---|---|
| seed 0 | −1.38 | −1.38 | cleared ✓ |
| seed 1 | 0.205 (deep thick) | 0.78 | z=0.065 FELL |
Both seeds drive DEEP into thick mud (beat R1 lean-drive 0.68 and R0 1.29 at their
deepest), but seed-1 over-lunged to y0.205 then collapsed. So firm-planting reward
ROBUSTLY improves penetration; CLEAN full traversal is seed-dependent / at the
stability limit. Tempers the R2 headline: firm-plant is a clear improvement, not yet a
robust solution. Failure mode = over-lunging in deep thick mud (chasing vx=1.5).

### R4 — "slow + firm" (complete the user's hypothesis)
Firm planting kept FULL velocity-tracking pressure → policy lunges + falls. User's
intuition was "slower AND firmly plant" — add the slow half: reduce tracking_lin_vel
weight so the policy isn't punished for slowing in mud, trading lunge for stable steps.
Variant Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowFirm: feet_slip -0.6, orient -8,
tracking_lin_vel 10→4. Test if slow+firm gives ROBUST deep/clean traversal.

### R4 RESULT — slow+firm seed0: cleared the mud STABLY
| recipe | final y | deepest | stability |
|---|---|---|---|
| firm-plant s0 | −1.38 | −1.38 | cleared (fast) |
| firm-plant s1 | 0.78 | 0.205 | FELL (lunge) |
| slow+firm s0 | −0.36 | −0.36 | **cleared, z~0.32 throughout, no collapse** |
Reduced velocity pressure (tracking 10→4) + firm planting → clears the mud without the
lunge-and-fall. Still pitches +17-19° in deep thick mud but stays upright. Trades raw
distance (didn't sprint out the far side like firm s0) for stability. Supports the full
"slow + firm" intuition. Flat eval 168 (lower only because tracking weight cut, not a
worse policy). Decisive test pending: slow+firm seed1 (does it replicate where firm
didn't?). Best ckpt 20260612_012032_..._muddr4xslowfirm_seed0.

### R4 CONCLUSIVE — slow+firm REPLICATES (robust recipe) 🎯
| recipe | seed0 | seed1 | robust |
|---|---|---|---|
| firm-plant | −1.38 ✓ | 0.78 FELL | ✗ 1/2 |
| **slow+firm** | −0.36 ✓ | **−2.20 ✓** | **✓ 2/2 cleared, upright** |

Both slow+firm seeds clear the full Newton mud gradient (thin→medium→thick) and walk out
the far side UPRIGHT (z~0.34 throughout). The "slow" half (tracking 10→4) delivered the
robustness firm-plant alone lacked. **User's full "slower AND firmly plant" intuition
validated — both halves necessary: firm planting = traction (the real bog mechanism),
slowing = no destabilizing lunge.**

## ROBUST RECIPE for Newton mud traversal (the answer)
variable-impedance control + 4× mud DR + firm-planting reward (feet_slip -0.6, orient -8)
+ reduced velocity pressure (tracking_lin_vel 10→4). Variant:
Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowFirm. Ckpts seed0
20260612_012032_*, seed1 20260612_014631_*.

Open: is variable-impedance ESSENTIAL or does slow+firm rescue joint-PD too? (testing)

### R5 — slow+firm on joint-PD: TOTAL FAIL → variable-impedance is ESSENTIAL
joint-PD + slow+firm: final y=3.388 — never entered the mud (froze at spawn, drifted
backward, z0.23). The exact recipe that robustly clears the mud on var-impedance makes
joint-PD FREEZE. joint-PD's response to harder mud DR + conservative reward = barely
move (R1 jPD-4× y2.78 → R5 jPD slow+firm y3.39, progressively more timid). Variable
impedance uses the same signals to learn active stiffening + traversal. CONCLUSION:
stiffness modulation is NECESSARY for mud traversal; reward+DR alone can't rescue
fixed-gain control.

## ═══ AUTONOMOUS MUD-TRAINING RUN — FINAL SUMMARY ═══
Goal: improve training so the Go2 policy traverses Newton MPM mud (held-out test).
Metric: final y on thin→thick maxfwd (vx=1.5) Newton traverse; mud y0-3, y<0 = cleared.

Full ladder:
| round | recipe | final y | outcome |
|---|---|---|---|
| R0 | var-imp, 1× DR | 1.29 | bog (medium) |
| R0 | joint-PD, 1× DR | 1.62 | bog (medium) |
| R1 | var-imp, 4× DR | 0.68 | reached thick |
| R1 | joint-PD, 4× DR | 2.78 | REGRESSED (timid) |
| R2 | var, 4× + firm-plant | −1.38 (s0) / 0.78-fell (s1) | clears but seed-variant |
| R4 | var, 4× + slow+firm | −0.36 (s0) / −2.20 (s1) | **ROBUST: both clear upright** |
| R5 | joint-PD, slow+firm | 3.39 | froze at spawn |

ROBUST RECIPE = variable-impedance + 4× mud DR + firm-planting reward (feet_slip
-0.1→-0.6, orientation -5→-8) + reduced velocity pressure (tracking_lin_vel 10→4).

Three compounding levers, each addressing a distinct failure:
1. variable-impedance control (stiffness modulation) — ESSENTIAL; joint-PD can't and
   gets MORE timid with harder mud training.
2. 4× mud DR — pushes bog→thick (resistance exposure).
3. firm-planting reward (feet_slip) — targets TRACTION, the actual Newton bog mechanism
   (force probe: Newton bogs at ~3N, not resistance). Gets through thick mud.
4. reduced velocity pressure — trades the destabilizing lunge for stable steps (robust
   across seeds; firm-plant alone was a coin-flip).

User's "slower AND firmly plant feet" intuition VALIDATED, both halves necessary, and
for the right mechanistic reason (firm=traction, slow=stability). Pitch is a real
command-scaled driving mechanism (lean→push), not an artifact.

Best ckpts: var slow+firm seed0 20260612_012032_*, seed1 20260612_014631_*.
Videos in projects/mud_eval/recordings/: slowfirm_thinfirst_maxfwd, slowfirm_seed1_maxfwd,
firm_thinfirst_maxfwd, var4x_thinfirst_maxfwd, jointpd4x_thinfirst_maxfwd.

### R6 ablation — slow+slip-only (drop orientation penalty): orientation ALSO matters
slow + feet_slip ONLY (no orient): y=0.638 (deep thick, did NOT clear), pitch +22°
sustained. vs slow+firm (slip+orient) y=-0.36/-2.20 (cleared), +8° exit. So BOTH reward
terms contribute: feet_slip → traction/depth, orientation → pitch control → stability to
finish clearing. Earlier inference (feet_slip is THE lever, orient dispensable) was half
right — feet_slip drives depth but orient enables the clean exit. Recipe is near-minimal;
each lever earns its place. (n=1 ablation — suggestive.)

## ═══ AUTONOMOUS RUN COMPLETE ═══
Every component of the robust recipe (var-impedance + 4× DR + feet_slip + orientation +
reduced velocity) was shown necessary by removing it:
- drop var-impedance (joint-PD): R5 froze at spawn (y3.39).
- drop 4× DR (1×): R0 bog at medium (y1.29).
- drop reduced velocity (firm-plant): seed-variant, lunge-and-fall (R3 seed1 fell).
- drop orientation (slow+slip-only): R6 didn't clear (y0.64), over-pitches.
- drop feet_slip entirely (lean-drive R1): y0.68, bog in thick.
Robust winner stands: Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowFirm, both seeds
clear (y-0.36, -2.20). Goal met + fully characterized.

## Follow-ups (post-investigation, user "go for it")
### #1 stiffness readout — var-impedance thesis confirmed at mechanism
Winning slow+firm ckpt, commanded per-axis stiffness scale s vs FIXED mud depth (MJX):
| depth(m) | mean s |
|---|---|
| 0.00 | 0.460 |
| 0.05 | 0.482 |
| 0.10 | 0.519 |
| 0.15 | 0.547 |
| 0.22 | 0.551 |
Monotonic ramp 0.46→0.55 (~20%) — the policy STIFFENS stance in deeper mud (s scales
kp toward [6000,6000,8000]). Modest but consistent → variable-impedance is used as
designed (active stiffening), the mechanism behind why joint-PD can't.

### #3 thick-FIRST protocol — recipe crushes the original mud_eval headline
slow+firm winner, spawn y=-1 facing +Y → hits DENSEST mud (thick y0-1) immediately, no
warm-up. Final y=2.727 (powered thick→medium→thin, upright z~0.30-0.34 throughout).
Original mud_eval headline (thick-first, 2026-06-10): var-impedance bogged at y=0.34 in
thick. This recipe: y=2.73 — cleared thick AND medium into thin. ~8× deeper. Robust to
protocol (thin-first AND thick-first both work). Video slowfirm_THICKfirst_maxfwd.mp4.

### #5 — 1× DR + slow+firm: 4× DR is NECESSARY (not dispensable)
slow+firm reward at only 1× Isaac mud DR: Newton y=0.829 (deep thick, did NOT clear) vs
4× DR slow+firm y=-0.36/-2.20 (cleared). So the strong 4× DR matters — both the DR
strength AND the reward shaping contribute. (ckpt died at 4.44M/89%, plateaued ~167 eval
— undertrained but unlikely to flip the "didn't clear" conclusion.) Ablation COMPLETE:
every lever necessary (drop var-imp→froze; drop 4×→1× y0.83 no-clear; drop slow→fall;
drop orient→y0.64 no-clear; drop feet_slip→y0.68 bog).

### #8 forgetting check (winner vs flat-DR baseline on flat + rough)

## ═══ GAIT REFINEMENT — RMA-minimal reward (2026-06-12) ═══
**Motivation:** slow+firm winner walks an odd tripod-ish gait on flat (back-right foot
held up). Hypothesis (from RMA arXiv 2107.04034 minimal reward): our gait-shaping feet
terms cause it — `feet_air_time +0.1` REWARDS time-in-air (lets the policy park a leg),
and a lifted foot also dodges the contact-gated `feet_slip` penalty → tripod.

### RMA-minimal (drop air_time+clearance+height+pose+stand_still, feet_slip→-0.8)
Variant `...MudDR4xSlowFirmRMA`, ckpt `20260612_152650_..._seed0`. 5M, seed0.
- **Flat-forward gait (MJX):** walks 1.12 m/s (cmd 1.0), pitch flat (~0°, std 2.1),
  upright (z 0.318, no fall). Tripod PARTIALLY cleared: back-right (RR) leg now CYCLES
  (22 cycles vs 33-37 other legs) instead of fully parked — but still asymmetric, tucked
  band (RR thigh [0.50,0.93] vs others ~1.0-1.5; RR calf never extends past -1.76 vs
  others -1.1). Milder tripod, not a clean trot.
- **Newton mud (the critical test): BOGGED AT ENTRY.** y frozen +3.30→+3.41 (never
  entered mud), z sank 0.54→0.23, upright (pitch +3°) but stuck the entire 750 frames.
  slow+firm cleared to y=-0.36. **RMA-minimal DESTROYED the Newton traverse.**

**KEY FINDING — tension between flat-gait cleanliness and mud traverse:**
The feet terms RMA drops are LOAD-BEARING for mud. `feet_clearance -2.0` + `feet_height
-0.2` are the foot-EXTRACTION incentive (lift the foot to clear height) that pulls feet
OUT of mud each step. Zero them → feet plant and sink → bog. So the same lift-shaping
that causes the flat tripod is what lets the policy extract feet from mud. RMA-minimal is
too aggressive — a dead end for the actual goal (mud traverse).

### NoAir middle profile (slow+firm with feet_air_time→0 ONLY) — RUNNING
Diagnosis splits the feet terms: `feet_air_time` (REWARDS duration-in-air = park-a-leg =
tripod driver) vs `feet_clearance/-height` (penalties enforcing clearance = mud
extraction). Zero ONLY feet_air_time; keep clearance/height (mud) + firm-plant shaping.
Variant `...MudDR4xSlowFirmNoAir` (commit d47a8c9). Tests: is feet_air_time the sole
tripod driver, with the Newton traverse preserved? Training 5M seed0 in flight; eval
flat gait + Newton next.

NOTE: dropped the PD-RMA gait 2×2 baseline — RMA breaks mud (PD already can't do mud), so
the 2×2 became academic. Pivoted GPU to the NoAir middle profile (on-mandate: mud perf).

## ═══ SPAWN-PARITY BUG (found 2026-06-12, user-flagged) ═══
**The Newton eval spawned the robot in a different state than warp training.** User
noticed the RMA Newton video "doesn't even move forward, spawns in the air." Investigated:
- Newton (`record_traverse_maxfwd.py`) spawned the vendored example's `INITIAL_Q` pose
  (hip ±0.1, thigh 0.8/1.0, calf -1.5 — splayed, straight-ish) at base **z≈0.54**, then
  let it DROP. (`mud_model.set_home_pose` was SUPPOSED to override but silently doesn't;
  `initial_position` z is ignored too — the example FK-places the base from the pose.)
- Warp/MJX training ALWAYS starts SETTLED at the keyframe `home` pose **(0, 0.9, -1.8)**
  uniform, base **z=0.27**, feet on ground. So every Newton eval had an OOD drop-in
  transient from a splayed pose the policy never trained from.
**Fix** (record_traverse_maxfwd.py, post-build state override): overwrite
`state_0.joint_q[7:19]` = home pose, `[2]` = 0.27, then `newton.eval_fk`. Leaves yaw +
xy untouched. Home is uniform across legs so the policy↔Newton leg-order mismatch is moot.
**Validated SOUND** — winner (slow+firm) re-baselined under parity still CLEARS:
y 3.30→**-0.16** upright (was -0.36 with the drop-in; drop-in gave slightly more depth,
conclusion unchanged). RMA still FAILS under parity (drifts backward, collapses z→0.057).

## ═══ NoAir VERDICT — feet_air_time was the SOLE tripod driver 🎯 ═══
All three profiles, parity spawn, thin-first max-fwd (vx=1.5), seed0:
| profile | feet_air_time | clearance/height | flat gait | Newton final y |
|---|---|---|---|---|
| slow+firm (winner) | +0.1 | kept | TRIPOD (RR parked) | clears -0.16 |
| RMA-minimal | 0 | DROPPED | partial fix (RR tucked) | COLLAPSES (fail) |
| **NoAir** | **0** | **kept** | **CLEAN (4 legs cycle)** | **clears -2.42** |
- NoAir flat: 1.06 m/s, stance 0.340 (normal), all 4 legs cycle full-range (FR/FL/RR/RL
  cycles 20/37/30/31, thigh ranges 0.44-0.73) — NO parked leg. Mild FR asymmetry only.
- NoAir Newton: y -2.42 (DEEPER than winner -0.16). Tripod fixed AND traverse preserved.
**Conclusion:** `feet_air_time` (+0.1, REWARDS time-in-air → pays to park a leg) was the
sole tripod driver. `feet_clearance/-height` are the load-bearing mud-EXTRACTION terms —
RMA failed because it dropped THOSE too. NoAir kills only air_time → best of both.
NoAir is the new recommended recipe. CAVEAT: seed0 only (winner was 2-seed robust);
confirm seed1 before declaring bulletproof. Videos: noair_FLAT_fwd.mp4,
noair_PARITY_thinfirst.mp4; winner_PARITY_thinfirst.mp4, rma_PARITY_thinfirst.mp4.
