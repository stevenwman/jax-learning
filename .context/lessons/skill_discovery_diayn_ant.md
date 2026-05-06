# DIAYN on Ant (MJX/Warp port) — visual diversity gate closed

> Lesson from SD-B Wave D Gate 2 follow-up (2026-05-04 → 2026-05-05). DIAYN
> trained on `AntMJXClassic` (27d obs, no cfrc — Gym v4 style) and `AntMJX`
> (105d obs, with cfrc — Gym v5 style) closes the visual-diversity gate that
> CheetahRun couldn't deliver. Three seeds × 1M on Classic all PASS the
> numerical visual gate (max-pairwise xy > 3.0 m OR circular heading-std > 30°).
> Companion to `lessons/skill_discovery_diayn_cheetah.md`.

---

## §1. What Ant gets you that CheetahRun doesn't

- **A 2D xy plane to render skill trajectories.** Cheetah is planar with no
  lateral/yaw axis; "diversity" reduces to forward-vs-backward velocity which
  looks identical in renders. Ant's torso CoM xy is the canonical DIAYN
  visualization (Eysenbach 2018, Fig. 4 / App. D.3).
- **Two obs variants for side-by-side.** `AntMJXClassic` (no cfrc, 27d) and
  `AntMJX` (with cfrc, 105d). Side-by-side shows the canonical DIAYN-on-rich-obs
  failure mode (discriminator hides skill diffs in tiny contact-force minutiae)
  while Classic forces motion-based discriminability.

## §2. Pipeline numbers (RTX 5080, 16 GB VRAM, MJX Warp)

- **Wall-clock 1M:** ~56 min/seed (vs Cheetah's ~28 min). Ant runs at ~300 sps
  vs Cheetah's ~600 sps because frame_skip=5 substeps + richer body geometry.
- **GPU env vars:** use `XLA_PYTHON_CLIENT_PREALLOCATE=false` (NOT
  `XLA_CLIENT_MEM_FRACTION=0.55`). RK4 + Warp PTX module load OOMs at 0.55
  because JAX preallocation eats all VRAM before Warp loads kernels. Disabling
  preallocation lets Warp load lazily.
- **Buffer/env flags:** at 64 envs, use `--buffer-size 524288 --num-envs 64`
  for Classic (27d → ~1 GiB buffer). For v5 (105d), `--buffer-size 1048576`
  (~3.6 GiB buffer). Default 4M × 128 envs OOMs both.
- **`<framelinacc>` sensor required in ant.xml.** MJX/Warp gates `cfrc_ext`
  population on having an accelerometer/force/torque/framelinacc/frameangacc
  sensor present. Without it, `cfrc_ext` is identically zero, silently zeroing
  `reward_contact` and breaking parity with Gym Ant-v5. The sensor's value is
  unused — only its presence triggers the `rne_postconstraint` pass.
- **Backend dispatch.** AntMJX (and AntMJXClassic) bypass `pg_registry`. Use
  `maybe_load_custom_env(env_name)` helper at all three call sites:
  `mjx_backend.py` training-env (~line 190), eval-env (~line 215),
  `record_video.py` (~line 334). `detect_backend("AntMJX") == "mjx"` by default
  fallthrough; no `_GYM_ENV_NAMES` registration needed.

## §3. Per-seed results (1M, 8 skills, 3 seeds)

### AntMJXClassic (27d, no cfrc) — 3/3 seeds PASS visual gate

| Seed | DiscA final | Best skill | Worst skill | Spread | Gate (50% max-\|mean\|) | max-pairwise xy | heading-std |
|---|---|---|---|---|---|---|---|
| 0 | 0.998 | z5=-12 | z1=-1238 | 1226 | 619 (2.0×) | 0.431 m | 79.4° |
| 1 | 1.000 | **z6=+199** | z4=-700 | 900 | 350 (2.6×) | **1.197 m** | 81.8° |
| 2 | 1.000 | z3=-264 | z1=-1174 | 910 | 587 (1.5×) | 0.673 m | 96.1° |

**Seed 1 is the headline.** z6=+199.6 is the first DIAYN skill across ALL our
SD-B + Ant runs that produces *positive* task return — meaning that skill
actually learned to walk forward. max-pairwise 1.197 m vs 0.43/0.67 m on
seeds 0/2 confirms one trail visibly extends beyond the central cluster.

### AntMJX (105d, with cfrc) — seed 0 only

| Seed | DiscA final | Spread | max-pairwise xy | heading-std |
|---|---|---|---|---|
| 0 | 0.953 | 873 | 0.253 m | 110.0° |

v5 has tighter xy displacement (0.25 m vs Classic's 0.43 m) but **wider**
heading distribution (110° vs 79°). The pattern: discriminator separates
skills via cfrc minutiae, so policy doesn't need to MOVE far to be
discriminable; small obs perturbations suffice. Classic forces actual motion
variance.

## §4. The cfrc-obs failure mode (canonical-DIAYN-limit, refined)

- DIAYN's MI objective `I(s; z)` is bounded by `log(num_skills)` and is
  reachable via *any* discriminable s-vs-z mapping.
- **More obs dimensions = more places for the discriminator to hide.** With
  cfrc_ext (78 dims of contact forces) in the obs, the discriminator can
  classify skills via tiny contact-force differences without the policy ever
  making meaningfully different gross motions. Result: DiscA → 0.95+ but xy
  trajectories cluster within < 1 m of origin.
- **Removing cfrc forces motion-based discriminability.** Classic 27d obs
  (qpos[2:] + qvel only) makes the discriminator depend on body-frame
  configurations and velocities, which require actual locomotion to vary.
- This explains why Classic seed 1 produced a positive-locomotion skill
  while Cheetah's 3 seeds (both with similar dim counts but no cfrc to hide
  in) showed widely-varying max-skill values without ever producing positive
  forward locomotion. Cheetah's body morphology (planar, prone-to-fall) is
  the bottleneck, not obs structure.

## §5. Visual gate methodology

**Don't use linear `np.std(arctan2(...))` on circular heading data.** It treats
+179° and -179° as 358° apart instead of 2°. Use circular std from resultant
length R: `R = sqrt(mean(cos)² + mean(sin)²)`, then `circ_std = sqrt(-2 ln R)`.

`scripts/plot_skill_xy.py` rolls each skill `--rollouts-per-skill` times (3
default) for `--rollout-length` steps (500 default) from a *deterministic*
reset (no qpos/qvel noise — otherwise reset noise dominates trajectories at
short rollout lengths). Plots all rollouts overlaid in skill-color, plus a
star marker at origin. Numerical gate prints + emits PASS/FAIL.

`lax.scan` the rollout (not Python loop). Per `lessons/warp.md`, Python loops
calling `mjx.step` per iteration cause Warp contact-buffer OOMs because
allocations don't get pooled. JIT the entire `rollout_length` scan and a
single Warp graph compilation handles all steps.

## §6. Engineering gotchas (don't lose 1h to these again)

- **Background bash dies on session resume (SIGHUP).** First Classic 1M run
  (`brd8cen59`) ghost-completed with 0-byte log + no ckpt. The runtime fired
  `exit 0` notification when noticing the dead handle. Mitigation: stay in
  session for long runs OR use `setsid`/`nohup` wrapper. Always verify
  progress by filesystem (log size growing, ckpt timestamp, GPU mem usage)
  rather than trusting completion notifications.
- **`record_video.py` helper signatures:** `_resolve_skill_vector(meta,
  skill_index, skill_vector_path)`, `_build_select_action(meta, obs_dim,
  action_dim) → (algo, kind)`, `load_actor_for_inference(ckpt) → (meta,
  actor_params, norm_state, actor_batch_stats)` (4-tuple). `obs_dim` for
  `_build_select_action` must be the **augmented** dim (raw_obs + skill_z),
  not raw — saved actor was trained on the augmented obs.
- **`_SkillWrappedAlgo(inner=algo, skill_z=z)` for skill conditioning.** Its
  `select_action(actor_params, obs, key, deterministic=False)` accepts raw
  obs (B, raw_obs_dim) and concatenates z internally before delegating to
  `inner.select_action(actor_params, obs_aug, key, deterministic=...)`.
- **`deterministic_reset`** must construct a full `mjx_env.State` matching
  the field structure that `step()` produces. The training-loop wrapper's
  `lax.scan` over `action_repeat` requires reset (carry input) and step
  (carry output) to share pytree structure — info dict needs the same keys
  in both.
- **Don't trust env.observation_size to be lazy.** It's computed via
  `jax.eval_shape(self.reset, ...)` so calling it triggers a full env build
  + dummy reset. Cache it in your script if reusing across rollouts.

## §7. SD-B Wave D Gate 2 — CLOSED

3/3 AntMJXClassic seeds pass the numerical visual gate. v5 seed 0 also
passes via heading-std (relatively wide-fanning despite tighter xy spread).
Plan acceptance "≥2 of 3 seeds pass" met with margin.

Headline figures:
- `.context/figures/ant_classic_diayn_3seeds.png` — 3-panel Classic (the
  Wave D Gate 2 deliverable)
- `.context/figures/ant_classic_vs_v5_seed0.png` — side-by-side comparison
  showing the cfrc-obs effect

## §8. When to revisit this lesson

- Future SD-D / SD-E (METRA / D3) on AntMJXClassic should produce more
  positive-return skills than DIAYN's 1/24 here (only seed 1 z6 hit positive).
  Use this entry as the DIAYN-baseline contrast.
- If a future DIAYN regression on AntMJXClassic shows DiscA < 0.95 at 1M
  or zero seeds with positive-return skills, something broke. Bisect against
  ckpts referenced above.
- The cfrc-obs hide-out hypothesis (§4) is testable: train DIAYN with even
  larger obs (e.g. add raw sensor data, dummy random vectors). Prediction:
  DiscA still ≈ 1.0 but xy spread → 0. Confirms more dims = more hideout.
