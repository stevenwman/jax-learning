# Lessons Learned — Index

Navigable index of all project lessons. Each entry is a one-liner — click through for full context.
JAX/Flax fundamentals in `lessons/learner.md`.

---

## [PPO](lessons/ppo.md) — 19 lessons

- **Match reference EXACTLY before investigating** — diff the code first (found 2x gap in 5 min of reading)
- **Entropy coefficient is env-dependent** — CheetahRun needs 0.0; CartpoleBalance needs 0.01
- **LR annealing for continuous control** — linear schedule to 0, standard PPO practice
- **State-independent log_std for PPO** — state-dependent NaN'd on HumanoidRun (21-dim entropy ~30 nats)
- **Minibatch count, not size** — fix num_minibatches (32), derive minibatch_size from data volume
- **Batch-level advantage normalization** — normalize full batch before splitting into minibatches
- **Reshape crash on non-divisible batches** — truncate permutation to `num_minibatches * minibatch_size`
- **HumanoidRun diagnostic signals** — healthy ranges for clip fraction, entropy, VLoss, KL, log_std
- **Interleaved collect→update** — many short cycles (num_updates_per_batch=16) beats long rollouts
- **VLoss oscillation from synchronized resets** — interleaved short rollouts desynchronize episode phases
- **Truncation handling in auto-reset envs** — zero out TD error at truncation, don't try to correct bootstrap
- **Tanh squashing requires matched entropy** — must include Jacobian correction `log(1 - tanh²)`
- **Batch volume controls safe epochs** — more epochs need proportionally more data
- **ContractionPPO on dense-reward stabilization is ~neutral** — port is algorithmically faithful; no return gains on Go2BongoHandstand at ref HPs (paper's claim is wind-robustness, which we didn't test)
- **Recording rollouts must freeze norm + handle frame stack** — inference path must mirror training's preprocessing exactly; bongo FS=3 exposed the bug
- **Structural parity, not just HPs** — same HP value behaves differently in different structural contexts
- **Know when PPO is the wrong tool** — HumanoidRun is a known failure case, check literature first
- **PPO validation summary** — CartpoleBalance PASS, CheetahRun PASS (826), HumanoidRun PASS (matches Brax ~8-10)
- **Remaining performance gap (RESOLVED)** — fixed by 0.25x value loss scaling + full-batch advantage norm

## [Off-Policy (SAC / TD3)](lessons/offpolicy.md) — 11 lessons

- **Obs normalization: NEVER before buffer storage** — normalize at sample time with `--obs-norm` (Go2: 139 vs 97)
- **SAC validation results** — WalkerWalk 975, HumanoidRun 426 (vanilla) / 892 (FastSAC)
- **TD3 needs gradient clipping on high-dim tasks** — SAC's entropy provides implicit stability TD3 lacks
- **Replay ratio must scale with num_envs** — 1:1 for 128 envs = 1 grad step per 128 samples, use 4-8x
- **TD3 exploration fails at >10 action dims** — SAC's structured exploration wins on humanoid-class tasks
- **Target entropy = 0 for SAC at scale** — classic `-dim(A)` causes alpha collapse at 1024 envs / 100M steps
- **AdamW requires `params` in optimizer.update()** — weight decay needs the params themselves
- **Optimizer decoupling** — algorithms define what to optimize, not how (DI pattern)
- **Asymmetric critic: faster early learning, same ceiling** — A/B on Go2 FastSAC: ~2x faster to 270+ but final 276 vs 279 (noise). Actor obs bottlenecks convergence.
- **Frame stacking doesn't help locomotion with proprioceptive obs** — A/B on Go2 FastSAC: 276.5 (48d) vs 271.3 (144d stacked). `last_action` already provides temporal context.
- **Staged rewards need longer budgets** — gated rewards (box_target after reached_box) require 10M+ steps to discover full sequence; 2M plateau is stage 1, not convergence
- **Truncation: mask the loss, zero the bootstrap** — Brax convention (SAC/TD3). `target = r + γ(1-done)V_next`, `loss *= (1 - truncation)`. Fast*/Flash* were missing the mask — teaching Q=r at timeout steps, systematic underestimation on long-horizon.
- **Q bias is the cleanest diagnostic for truncation handling** — on long-horizon tasks, post-fix Q bias should be near zero. Strongly negative bias = fix isn't applied or wrapper doesn't populate `info["truncation"]`. Add `eval/q_bias` to smoke-test checklist.

## [Distributional RL (C51 / FastTD3 / FastSAC / FlashSAC)](lessons/distributional.md) — 12 lessons

- **FastTD3 scale matters** — 285 eval at 128 envs/5M steps, **880** at 1024 envs/86M steps
- **C51 V_min/V_max is critical** — distributional Q is hard-bounded, get it wrong and critic is blind
- **C51 cross-entropy NaN** — `log_softmax` → `-inf`, then `-inf * 0 = NaN`. Clamp to -30.
- **SAC variants can't match FastTD3 on low-dim tasks** — expected, test on HumanoidRun instead
- **Run simple baseline before fancy version** — vanilla SAC 771 in 8 min vs FastSAC 582 in 108 min
- **Verify configs against source code** — 7+ critical mismatches found (tau 25x wrong, hidden dims, activation)
- **Read the whole recipe** — missing gamma=0.97 and AdamW β2=0.95 completely changed behavior
- **FastDSAC paper says "Gaussian NLL" but code uses Huber** — 2 days debugging the wrong loss
- **FlashSAC weight norm axis** — Flax kernel `(in, out)` vs PyTorch `(out, in)` → normalize `axis=0` not `axis=-1`. Silent correctness bug.
- **Target BN stats NOT copied from online** — target critics maintain own running stats via `train=True` forwards. EMA only updates learned params.
- **Asymmetric done signals** — reward normalizer resets on `terminated|truncated`, C51 bootstrap uses `terminated` only. Mixing them causes value underestimation.
- **BatchNorm running stats must follow training state** — Flax BN stats are separate pytrees. Eval with stale init stats → online 686, eval 26. Update before every eval call.

## [JAX Performance](lessons/jax_performance.md) — 8 lessons

- **Python collect loops kill throughput** — 87x overhead from Python-GPU sync, use lax.scan
- **lax.scan vs Python loops: 542x speedup** — for small MLPs, 95% of time is dispatch overhead
- **JIT closure recompilation in eval loops** — new closure = cache miss = 15s recompile each eval
- **lax.scan episode return tracking** — state spanning episode boundaries must persist outside scan
- **Python `if` vs `jax.lax.cond`** — TracerBoolConversionError inside scan/vmap, use lax.cond
- **JIT the hot path** — non-JIT'd JAX is 5x slower than numpy for small ops
- **lax.scan carry cost** — 4M-entry buffer in carry = 30% slower than Python loop
- **Faster component ≠ faster training** — 4.8x buffer speedup = 1.5% end-to-end improvement

## [TD-MPC2 (Model-Based RL)](lessons/tdmpc2.md) — 5 port bugs + reusable patterns

- **Replay sequence-sampling needs `stride=num_envs`** for round-robin multi-env storage (kingpin bug, 50× improvement)
- **action_repeat is part of HP context** — match source's value AND adjust discount via the heuristic
- **Truncated ≠ terminated in TD bootstrap** — same recurring class as `lessons/offpolicy.md §Truncation Handling`
- **Q dropout in policy/qscale paths** — PyTorch `.train()`/`.eval()` semantics don't auto-port; thread `deterministic`+`rngs` per call site
- **Eval RNG isolation** (`fold_in(seed+9000, eval_index)`) — applies to PPO/SAC/TD3 too, currently they all consume from training key
- **Multi-agent audit pattern** — 3 parallel opus + 1 validator finds bugs that any one misses
- **Paper Table 2 had a typo** — Humanoid action_dim=21 per Figure 15 caption (not 24); always cross-check Tables vs Figures

## [Skill Discovery (DIAYN on CheetahRun)](lessons/skill_discovery_diayn_cheetah.md) — 6 lessons

- **DIAYN+CheetahRun is a pipeline smoke test, not a behavioral demo** — DiscA hits 0.95+ at 1M, but 5-7 of 8 skills collapse to ~0 task return. Discriminator separates via tiny obs deltas, not gross behavior. Expected DIAYN limit; motivates METRA/D3.
- **Numerical gates suffice for SD-B sign-off; visual gate belongs on Ant** — CheetahRun planar 2D body all looks similar in renders. Ant xy-trajectory plot (DIAYN App. D.3) is the canonical legible diversity figure. Defer visual acceptance to Ant.
- **Per-skill collapse pattern is canonical, not a bug** — DIAYN paper Fig. 12 shows the same: max-skill mean varies wildly seed-to-seed (7-50 in our 3-seed run), most skills cluster near 0. Don't bisect on this; it's the method ceiling.
- **3 parallel runs at XLA_CLIENT_MEM_FRACTION=0.55 won't fit on 16GB** — 3 × 8.8GB > 16GB. Either serialize (1.5h for 3×1M) or drop fraction to ~0.25 (untested, eval-scan OOM risk). Serial is the safe default.
- **DIAYN wall-clock per seed (RTX 5080 + Warp + 8 skills + 1M)** — ~27.5 min, 2-3 SAC grad updates per env step at 128 envs / 8 grad_updates_per_step.
- **When SD-D/E (METRA, D3, DUSDi) reduces this collapse, that's the win** — contrast against this entry for any future method comparison. >5 of 8 skills behaviorally active = real improvement over DIAYN baseline.

## [Determinism (JAX/XLA + GPU Physics)](lessons/determinism.md) — bit-ID limits

- **JAX/XLA algo bit-ID** with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (verified via `scripts/check_tdmpc2_determinism.py`)
- **mujoco_warp env NOT bit-ID** across processes — `wp.atomic_add` in narrowphase, fix in flight (Warp 1.14, ~Jun 2026)
- **Cross-process training trajectories cannot be byte-identical on GPU** today; report seed-averaged results, not single-run

## [Infrastructure](lessons/infrastructure.md) — 18 lessons

- **Complete your migrations** — don't "archive" the old path. Archive ≠ delete. Validated new path? Same-day deletion, same PR. Otherwise you end up with 2 entry points for 1 feature.
- **Orbax checkpointing** — must call `wait_until_finished()`, save meta.json alongside
- **Orbax restore needs exact pytree match** — separate inference artifacts (numpy) from training (orbax)
- **Checkpoint should be self-describing** — store full `dataclasses.asdict(cfg)` in meta.json
- **Video recording: two-phase** — GPU rollout via lax.scan, then CPU rendering
- **Save trajectory data alongside videos** — two policies with same eval=11.6 had completely different behaviors
- **Eval/recording must match training preprocessing** — missing obs norm → instant death on video
- **Scientific notation for metrics** — `{:8.2f}` printed `0.00` for VLoss=0.003
- **CycloneDDS requires Python <3.13** — separate deploy venv (3.12) from training venv (3.13)
- **Integer division truncation** — `200000 // 128 * 128 = 199936`, final eval never fired
- **Verify training budget before debugging** — eval ~17 at 50M steps was on-curve, not broken
- **`--eval-every` is episodes, not steps** — `--eval-every 5000000` = 5M episodes, never triggers. Use ~50000 for Go2.
- **Env wrappers must be applied in all consumers** — FrameStackWrapper in training but not record_video = checkpoint incompatible at inference
- **Brax auto-reset does NOT reset state.info** — only pipeline_state and obs are reset. Any FIFO/history in state.info must use `jp.where(done, ...)` to self-reset
- **Inference artifacts must include ALL model state** — FlashSAC `actor_params.npy` missing BN batch_stats → eval 26 vs training 282. Orbax had it, inference artifact didn't.
- **mkdocstrings requires `Attributes:` for nn.Module** — `Args:` doesn't work for Flax dataclass fields; untyped params fail `--strict`
- **Wrapper composition is untested until combined** — JaxReplayBuffer dropped critic_obs in frame-stack JIT path; DomainRandWrapper bypassed FrameStackWrapper via `_swap_model`. Both worked alone, broke when combined.
- **Extract shared loops as functions, not classes** — 4 scripts shared 85% code. A Trainer ABC or re-unification both add noise. A shared helper function with 4 variation-point parameters keeps each script readable.
- **Ghost refs in docs propagate silently** — AGENT_HANDOFF updated in anticipation of a code change that got reverted. Every downstream doc update propagated stale info from that ghost. Always cross-reference docs against code before trusting internal handoff docs as authoritative.
- **4-persona docs review: undergrad catches factual drift** — high schooler finds jargon, PhD finds algo bugs, frontend finds CSS issues, but only the undergrad systematically cross-references every claim against code. That's where ghost refs get caught.
- **Treat every specific number in docs as a citation requirement** — the 18k sps claim was a real CheetahRun number cited on the Go2 page. Same failure mode as the obs dims ghost-ref. When copying a benchmark figure, copy the env name with it.
- **Validation blocks catch cross-agent drift** — `validation.links.unrecognized_links: warn` in mkdocs.yml caught a broken link between two parallel subagents before it shipped. Cost: zero. Benefit: catches cross-cutting breakage.
- **Drift test suite catches the "cleanup-N creates ghost-N+1" pattern** — `tests/test_docs_code_blocks.py` + `tests/test_docs_drift.py` mechanize the checks 4-persona reviewers keep making: Python fences compile-check, constructor kwargs match code, reverted-symbol greps stay zero, arxiv IDs resolve. Caught 16 issues on first run. See `lessons/infrastructure.md` §"Doc-Drift Test Suite".
- **XLA mem fraction has to drop for bigger-network algos** — `XLA_CLIENT_MEM_FRACTION=0.7` works for FastSAC/FastTD3 but FlashSAC OOMs Warp graph creation. Drop to 0.55 for FlashSAC. Bigger models = more JAX heap = less for Warp.
- **CheckpointManager "best" tracking excludes final eval** — `final_eval_and_checkpoint` doesn't update `ckpt_mgr.best_eval`, so grep "New best!" undercounts peak. Always check `max(best_in_loop, final_eval)` when reporting benchmarks.
- **`record_video.py` memory fix is two parts: `PREALLOCATE=false` *and* Python-loop instead of `lax.scan`** — env var handles preallocator-vs-Warp contention (~1-2 GiB transient at graph creation); the scan→loop refactor drops steady-state peak ~500 MB–1 GB by not preallocating the full mjx State × 1000 upfront. `lax.scan` is a training pattern; one-shot inference should use jitted-step in a Python loop. Now renders coexist with concurrent training. Both fixes landed 2026-04-13.
- **Anti-hallucination protocol cuts reviewer false positives** — drift-tests-first + tool-backed claims + date awareness + self-audit pass. Round 5 dropped hallucinations from ~2/round to 0, with verified rates 79-97% across reviewers. See `.context/references/docs_review_pattern.md` for paste-ready prompt template.
- **Docs reviews hit diminishing returns at round 5** — each round catches ~50% fewer issues than the prior. Stop rule: next round nets <3 actionable items. Past that, drift tests + quarterly reviews are enough; continuous reviewing is bikeshedding. **BUT** substantial code refactors reset the curve: round 6 (2026-04-13, after off-policy loop extraction + truncation fix + record_video refactor) netted 13 findings including 3 real 🟠s. Trigger reviews by code-change magnitude, not round count.
- **PhD reviewer catches the algorithmic bugs no one else can** — round 2 found truncation bug, round 6 found target-Q gating (`tau/policy_delay` effective decay) + FlashSAC standalone-justification gap. Every time: no other persona catches these; they need cross-referencing algo source against docs. Never skip PhD to save budget.
- **Drift tests protect structure; reviews protect meaning** — drift tests catch phantom kwargs, stale symbols, broken links. They can't catch "`tau=0.125` is 25x faster than SAC" being semantically wrong when the effective rate is `tau/policy_delay`. The claim is in a table matching the config default. Semantic correctness needs human review; structural consistency doesn't.
- **Architectural reasoning in code comments is invisible to doc readers** — `offpolicy_loop.py:4` had the explanation for why FlashSAC doesn't use the helper. Zero docs pages surfaced it. Rule: architectural decisions go in `docs/reference/architecture.md`, not module docstrings. Module docstrings document code; architecture docs document *why*.
- **Schema-from-checkpoint beats dim-only contracts** — deploy code that hardcodes obs layout silently mis-wires when sim env content drifts but total dim stays constant. Twice on Go2 (linvel→accel swap, ~14 days bad ckpts). Fix: serialize obs term list to meta.json at save, deploy reads it via sensor-fetcher registry keyed by name. Unknown name = clear error, not silent miswiring. Generalizes to any composable-obs env (bongo, pusht). See `lessons/infrastructure.md` §"Schema-from-Checkpoint for Deploy Obs".
- **`md_in_html` doesn't propagate `markdown` to child HTML elements** — the `markdown` attribute on an outer `<div>` only processes markdown in its direct text children; nested `<div>`s and `<p>`s need their own `markdown` / `markdown="1"` attributes. This combines with markdown-generating hooks (glossary auto-linking, etc.) to silently break rendered output — hook injects `[term](url){.cls}`, nested HTML keeps it as literal text. Survived 6 rounds of review because each component is valid in isolation; only the combination fails. Post-build check: `grep -rE '\[[A-Za-z][^\]]*\]\([^)]+\)\{\.' site/` should return zero matches.
- **Backend-agnostic env Protocol via registry beats per-env training scripts** — `train_pusht.py` was a 476-line fork of the SAC stack (numpy buffer, custom wrappers, custom loop, custom eval) because `make_envs()` hardcoded `pg_registry.load()`. Refactor: `EnvBundle` Protocol with `backend_kind` discriminator + `env_backends/` registry of `(name → builder)`. Adding a new env backend (gym, IsaacLab) is one file that calls `register_backend(name, builder)` at import. Adding a new env within a backend is 5 lines. `train_sac --env HalfCheetah` ran end-to-end on `gymnasium[mujoco]` (eval=5697 @ 200k, above published baselines) without touching any train script — proves the abstraction. Rule: when you find yourself forking a stack to support a new instance of the same task class, the abstraction is the env runtime, not the training infra. See `.superpowers/plans/2026-04-25-env-backend-refactor.md` and `journals/2026-04-26.md`.
- **Off-policy training loop already env-agnostic — fix at the construction layer, not the loop** — audit before refactoring: `run_offpolicy_loop` was already a Python loop (not lax.scan) calling `env_step(state, action)` directly, and `JaxReplayBuffer.add_batch` already auto-converted numpy → jax via `jnp.asarray`. The MJX coupling lived entirely in `make_envs()` (hardcoded `pg_registry.load()`) and `_make_nan_safe_step` (JIT-wraps env.step assuming JAX in/out). Phase 1 isolated those into `mjx_backend.py`; the loop didn't need to change at all. Pre-refactor audit saved ~3 hours of needlessly rewriting the loop that was already pluggable. Generalizes: when adding a new abstraction, find what's *already* abstract and don't rewrite it.

## [MuJoCo Engine](lessons/mujoco.md) — 4 lessons

- **Friction uses max-combine** — randomize foot geoms not just floor. PhysX DR ranges don't port to MuJoCo.
- **Sim2sim between different MJCFs is nearly as hard as sim2real** — train on the target model directly when possible.
- **Three Python APIs** — CPU (`mujoco`), MJX (`mjx` with `impl="jax"/"warp"`), standalone Warp (`mujoco_warp`). For RL: always MJX. Standalone Warp is a different interface to the same physics.
- **Use `<pair>` for per-contact friction control** — bypasses max-combine, gives independent friction per geom pair. `mjx.Model.pair_friction` is batchable for vmapped DR.

## [MJX Physics](lessons/mjx.md) — 5 lessons

- **MJX physics NaN at scale** — stochastic contact solver failure, guard with NaN+Inf checks on env boundary
- **GPU OOM is usually not a leak** — RTX 5080 starts at 95% capacity, XLA command buffers accumulate
- **MJX eval recompilation** — upstream issue, ~2 recompiles per eval call, mitigate with MEM_FRACTION=0.7
- **MJX→CPU transfer: every obs dimension must match** — zeroed linvel in CPU env killed transfer. Diff obs side-by-side.
- **MJX can't load all MJCFs** — cylinder-box collisions not implemented. Use MuJoCo Warp instead.

## [Vision RL](lessons/vision.md) — 4 lessons

- **Frame stacking: locomotion ≠ DMC** — DMC/manipulation stacks raw frames (DrQ-v2); locomotion uses CNN + GRU (ANYmal, DeFM). DreamWaQ/WTW are NOT pixel methods.
- **Asymmetric critic simplifies vision** — privileged critic skips images entirely. No shared encoder stop-grad, no frame stacking on critic. Actor CNN trains from policy gradients only.
- **Pixel replay buffer: uint8 is non-negotiable** — 100K entries at 84×84×9: 6.3GB (uint8) vs 25GB (float32). Assemble stacks at sample time.
- **MJWarp renderer: Warp-only, fixed nworld** — `mjx.render()` requires `impl="warp"`. nworld frozen at `create_render_context()` time.

## [AutoReset & Domain Randomization](lessons/autoreset_and_dr.md) — 10 lessons

- **Current DR is weak** — 256 frozen physics configs, never re-randomized. Same as Brax.
- **full_reset=True is wildly inconsistent** — +24% Go2, -99% CheetahRun. Env-specific.
- **JAX can't do selective reset** — vmap lowers cond→select. Both paths always execute. Isaac Lab avoids via PyTorch.
- **Zeroed qpos on Go2 = 10-19x solver blowup** — always use keyframe/default pose.
- **Dead envs don't slow physics** — constant throughput regardless of waste fraction.
- **Cumulative SPS is misleading** — JIT warmup dominates early. Always compare converged SPS.
- **Per-step GPU→CPU sync kills async execution** — one np.asarray() per step = 8x slowdown.
- **Syncd mode: 2x raw throughput, impractical waste** — 60-95% waste, tracker/buffer integration nightmare.
- **Per_step DomainRandWrapper (formerly DRv2): 4% slower, better eval** — Go2 FastSAC 5M: eval 280 (per_step) vs 270 (legacy).
- **DomainRandWrapper per_step is the path forward for Go2** — fresh ICs, clean state.info, per-episode DR foundation.

## [MuJoCo Warp](lessons/warp.md) — 6 lessons

- **CCD overflow — size naccdmax for complex geometry** — 8.6M overflow warnings at 1024 envs, 30% sps loss. Set `naccdmax=4000`, `ccd_iterations=100`, `njmax=100`.
- **OOMs in Python loops — must JIT physics steps** — Warp allocates collision buffers per `mjx.step()` call. Python loop = OOM. `lax.scan` = instant. Always JIT.
- **forcerange=[0,0] = unlimited** — unitree XML sets ctrlrange but not forcerange. PD torques unclamped → joints contorted → eval 2.2. Set forcerange = ctrlrange.
- **Inherits XML solver settings** — unitree's iterations=100, elliptic cone, eulerdamp=on vs MJX's 1/pyramidal/off. Audit `<option>` block when porting envs.
- **PD gains must match solver stiffness** — Kp=35/Kd=0.1 (MJX, 1-iter) collapsed on Warp (100-iter). Use Kp=20/Kd=0.5 (unitree_rl_gym). PD gains are coupled to solver config.
- **Joint order ≠ actuator order — THE root cause** — unitree qpos is FL-first, ctrl is FR-first. PD applied FL torque to FR actuator. Robot fought itself. Hours of debugging PD/solver/entropy were all red herrings. ALWAYS verify ordering when using third-party MJCFs.
- **"Stable" PD gains ≠ "trainable" PD gains** — Kp=10/Kd=1.0 holds the robot fine but trains 7x slower than Kp=20/Kd=0.5. Sluggish joint dynamics suppress the leg swings RL needs to find walking. Validate new PD gains with a training run, not a static hold test.

## [Manipulation (Push-T)](lessons/manipulation.md) — 26 lessons

- **Cylinder-box collisions need Warp** — MJX JAX backend raises `NotImplementedError`. Any manipulation env with cylinder pusher + box target is Warp-only.
- **Tighten solref for manipulation contacts** — default `solref=0.02` (20ms) gives visible penetration; use `0.004 1` + `solimp="0.98 0.995 ..."` + `iterations=50 ls_iterations=10`.
- **Shape-agnostic obs for cross-shape generalization** — `[center_xy, sin/cos(yaw), target_xy, sin/cos(goal_yaw), vels, last_action]` = 16d. Same obs for T/L/circle/plus. Policy infers contact dynamics, doesn't memorize T-geometry.
- **Three action modes for RL vs demo parity** — position-PD (our default, jittery), velocity/delta (smooth, RL-friendly), teleport (gym-pusht parity). `config.action_mode` flag, same obs, different ctrl mapping.
- **Data-driven shape registry** — `SHAPES = {"T": [geom_dicts], "circle": [...], ...}`. XML built programmatically. Adding new shape = one dict entry, no per-shape XML files.
- **Slide-joint body pos is an offset, not a starting position** — body `pos="-0.15 0 0.015"` + qpos=-0.19 = world -0.34, outside the wall. Anchor slide bodies at origin so qpos directly = world XY. Silent visual-only bug caught by top-down render.
- **Zero-action attractor in velocity/teleport modes** — pos-PD accidentally forces motion at action=0 (kp × err > 0); vel/tele do not (action=0 → ctrl=current → hover). Policies converge cautious in vel/tele regardless of shaping strength. Pos-PD is surprisingly the best default for RL-from-scratch pushing.
- **Reward shaping strength is a dial, not a monotonic knob** — bumping `r_block_vel` 2→10 on push-T regressed pos mode from eval +143 → -1.7. When a shaping term dominates ground-truth task terms, you optimize the wrong thing. Keep shaping ≤ 0.5× max task reward.
- **Always log per-component rewards + motion metrics when shaping** — `r_pos, r_angle, r_approach, r_block_vel, r_pusher_vel`, `pusher_vel_mag`, `block_vel_mag`, `pusher_to_block`. Essentially free, turns "why is A worse than B" from a multi-hour A/B into a 10-line diff rollout.
- **Vendor old static benchmarks; don't pip-depend** — gym-pusht broke on pymunk 7 (upstream API removed). Copied 700 LOC + LICENSE into repo, added `reward_mode` kwarg, packed 206 expert demos as 0.3 MB npz. Parity-tested byte-exact vs pip. Old 2023-paper benchmarks should be owned.
- **TimeLimit is NOT applied by direct env construction** — `gym.make("gym_pusht/PushT-v0")` adds TimeLimit(300); `PushTEnv(...)` does not. Episodes never truncate, Q bootstraps infinite future, critic explodes. Fix: `env = gym.wrappers.TimeLimit(env, max_episode_steps=300)`. This bug alone limited peak coverage 14% → 88% (6x) on push-T SAC.
- **Verify infrastructure before tuning HPs** — 8 runs and 15+ HP combos chased a ceiling caused by a 1-line env wrapper bug. Signal I missed: `ep_r_avg = -3100` is physically impossible (per-step reward ∈ [-0.3, 1.5]). Rule: when a metric goes outside its possible range, halt tuning, find the infra bug.
- **95% success threshold on push-T is above human-expert teleop** — LeRobot's 206 demos max coverage = 0.9489, never crosses 0.95. Reporting "0% success" is misleading for pure RL; always compare coverage distribution against the bundled demos.
- **Don't eyeball metrics from rendered video** — claimed a policy's episode had "30° yaw off" from screenshot; actual angle error was 4.66°. 6x wrong. Policy metrics are scalars accessible from env internals; 2 LOC of diagnostic > staring at pixels.
- **FastSAC C51 wrong choice for bounded-reward manipulation** — C51 atoms over default `[v_min=-20, v_max=20]` don't cover contact_gated Q-range (up to ~150 discounted). Critic blind past `v_max`. Use vanilla SAC (scalar Q) for short-horizon shaped-reward tasks; FastSAC for unbounded locomotion.
- **Combined recipe that works on push-T** — keypoint obs (18d) + frame_stack(3) (→ 54d) + obs normalization (pixel → [-1, 1]) + action_repeat(2) + contact_gated shaping + SAC(target_entropy=2, batch=1024, UTD=2, lr=1e-4, gamma=0.995) + TimeLimit. 84% mean / 89% peak sto coverage at 2M steps. Each component ablated independently; stack all.
- **Pymunk CoG offset breaks `block.position = goal_xy`** — T's center_of_gravity is body-frame (0, 45). Setting body.position+angle leaves world COG offset by ~30 px from intent. True identity state for `reset_to_state` is `[..., 224.2, 242.8, π/4]`, not `[..., 256, 256, π/4]`. Distance shaping using `block.position` carries this offset; not a hard bug but a footgun for diagnostics.
- **Coverage metric is geometrically sensitive — 2 px = 5% drop** — T has thin bars (~15 px wide); small translation loses lots of overlap area. Calibration table: 95% = 1.9 px / 2.5°, 90% = 3.8 px / 5.1°, 85% = 5.7 px / 7.8°. Visually-close poses can differ 10-20% coverage. Don't conclude "policy is bad" from coverage alone — check pose error directly.
- **Tunable `success_threshold` for sparse-reward tractability** — DP default 0.95 is unreachable by any policy from scratch (humans max 0.9489), so `reward_mode="sparse"` gives zero signal. Added kwarg to PushTEnv. Use `0.85` for sparse RL training. Default 0.95 preserves literature-comparison parity. Push-T papers report max-coverage-per-episode, NOT binary success rate.
- **Log-barrier coverage reward beats linear +8pp** — `r = -log(1 - cov/thresh + ε)`, ε=0.01 gives ceiling 4.6. Marginal reward ∝ 1/(1-cov+ε) — 9× steeper at cov=0.9 vs 0. Pushes the policy through the flat-marginal plateau that linear coverage creates around 85%. New `coverage_shape` kwarg on PushTEnv; 0.933 sto (vs linear 0.852), det std 5× tighter.
- **Action repeat is the dominant single knob on push-T RL** — stripping AR=2→1 collapses from 0.93 → 0.52 sto (−41pp). Larger effect than obs, reward shape, or frame stack combined. Contact-rich manipulation needs sustained directional force; AR=1 lets policy oscillate and kills push impulse. Test K∈{2,4,8} before tuning anything else. Action chunking (Q-chunking NeurIPS 2025) generalizes this.
- **Minimal shape-agnostic config beats full stack under log_bar** — `state(5d) + FS=1 + AR=2 + log_bar + contact_gated` hits 0.94 sto (first success event observed), vs full-stack 18d+FS=3 at 0.93 sto. Richer obs adds input noise without useful velocity signal when reward is strong. Use minimal config as cross-shape baseline — 5d obs is shape-agnostic (agent_xy + block_xy + block_yaw).
- **Bigger success bonus doesn't raise ceiling on unreachable thresholds** — 50→200 only tightened det policy (−5× std) because policy never crossed 0.95 threshold during training → never sampled the larger bonus. Verify terminals actually fire before tuning bonus magnitude. Otherwise the tuning is literally unused.
- **Pymunk shapes: decompose concave letters into annular sectors** — pymunk requires convex `Poly`. Letter S built from 2× 270° fat rings (rot-180 symmetric, overlapping mid-strip, 18 convex wedge quads); letter U from 180° half-ring + 2 rectangles. Bezier-centerline ribbons produce "fins" at tight curvature; ring decomposition gives uniform curvature. Size shapes so `inner_r > pusher_r + margin` for reachable hook interiors.
- **Shapely MultiPolygon fails on overlapping convex pieces** — `sg.MultiPolygon([s1, s2])` is not "polygon with holes"; overlapping members produce self-intersecting geometry → `GEOSException: TopologyException: side location conflict`. Fix: `unary_union([...]).buffer(0)` heals seams; or catch/stub for vibes-only rollouts. Don't assume pymunk→shapely round-trip yields valid geometry just because pymunk shapes are valid.
- **Zero-shot cross-shape transfer is a floor, not a working baseline** — T-trained policy on 5d pose-only obs hits 0.12 cov on ellipse/U, 0.025 on triangle, ~0 on S (vs 0.87 on T). 5d obs `(agent_xy, block_xy, yaw)` has no shape info so policy memorizes T-specific approach angles. Cross-shape needs DR training over shape set OR shape-aware obs (keypoints, contact history). Zero-shot only verifies infra correctness.
- **Variable-N keypoint obs with zero-padding leaks shape ID** — pusht letter matrix: padding per-shape KPs to MAX=11 with zeros gives each shape a unique number of trailing 0-slots. After NormalizeObsWrapper (low=0, high=512), pad slots become exactly −1.0 — a constant shape one-hot. Policy memorizes "which slots are pad" instead of using positions. Result: off-diagonal cross-shape transfer collapses to ~0%, DR row-mean 32.9% sto. Fix: dense fixed-N=10 KPs sampled by arc-length per shape (no padding) — DR jumps to 73.9% sto, beats 5d-state baseline by 11pp. Also resolved L/K/DR late-training critic collapses (downstream of the leak — multimodal target = "one policy per pad pattern"). Generalization: any "shape-agnostic" obs with per-shape variability (slot counts, dim padding, value-range scaling) creates an implicit shape ID. See `.context/studies/2026-04-22_pusht_letter_matrix.md`.

## [Bongo Board Handstand](lessons/bongo.md) — 10 lessons

- **Always verify policy behavior visually** — eval 397 looked great on paper, but the robot was balancing on the floor, not the board. Reward hacking is silent without video.
- **CMA-ES hard rejects poison the population** — returning 1e6 for invalid poses gives no gradient. Use soft penalties so CMA-ES can learn which direction is better.
- **Constrained DOFs should be computed, not optimized** — base_z is determined by pitch + joint angles + feet-on-ground constraint. Optimizing it wastes a dimension and causes spawn bugs.
- **MuJoCo euler uses degrees by default** — `euler="1.5708 0 0"` is 1.5° not 90°. Use `quat` for rotation-safe values across includes with different `<compiler angle>` settings.
- **Use contact sensors, not position heuristics** — `geom_xpos[i][2] < 0.03` misses edge cases. MuJoCo `<contact>` sensors are exact and threshold-free.
- **Reward hacking closes every loophole** — ground balance, board slam, head tripod — three exploits found across 4 runs. Enumerate ALL cheats and terminate for each.
- **Checkpoint resume doesn't save replay buffer** — expect transient dip on resume as buffer refills. Not a bug.
- **Frame stacking is critical for balance tasks** — single-frame obs can't infer acceleration. Frame-stack 3 jumped bongo eval from 24 → 47 (94% of max). Locomotion didn't benefit — balance is fundamentally about reacting to acceleration.
- **Regularization penalties can suppress necessary corrections** — torque/velocity penalties regressed bongo eval from 24 → 11. Only safe to add after the policy can solve the base task (or with frame stacking for efficient corrections).
- **PPO entropy collapse = dead exploration** — entropy -0.89, all 512 envs identical returns. First surviving strategy gets locked in. Monitor entropy + return variance together.

## [Terrain Curriculum](lessons/terrain_curriculum.md) — design + gotchas

- **Unified goal-directed design beats dual-class A/B** — all 4 types spawn on rim, cmd points to tile center; rotating body frame = free omnidirectional linvel DR. Single promote (reach & ~fall), single demote (fall | no_progress).
- **Contact-based termination, not `base_z < 0.18`** — world-frame z check false-positives at pit bottom of pyramid_down L5. `base_contact` sensor fires only on real torso-ground contact.
- **`reset_mode="per_step"` required** — preset default was silently wrong; TC wrapper effectively not applied at episode boundaries.
- **Flat as 5th col (2026-04-23)** — curriculum's goal-directed rotating cmd distribution ≠ `Go2WarpJoystickFlat`'s Bernoulli cmd. Without flat col in training, curriculum policy collapses in <30 steps on flat env. Adding flat as non-goal-directed 5th col (`_IS_GOAL_DIRECTED[-1]=False`, stays L0, Bernoulli cmd) fixes the distribution mismatch at cost of 1/5 hard-terrain training. v16 20M: 3/4 seeds survive full flat episode vs v14's uniform collapse.
- **Reward-weight ablation beat formula rewrite** — reward_orientation was 190× penalty spike on tilted terrain. User called out unverified claims; cloned legged_gym and confirmed our `sum(torso_zaxis[:2]²)` IS their `projected_gravity[:2]²`. Not a formula bug — weight dropped -5 → -1 fixed it.
- **pyramid_up is persistently hardest** — stuck ~L0–L1 after 20M, 4-col and 5-col. Probably needs face-stair spawn tuning or start-of-episode cmd shaping.
- **Warp non-determinism across runs with same seed** — single-seed lifetime tests misleading; flat env survival varies 607/871/1000/1000 across consecutive seed=0-3 runs on same checkpoint.

## [Go2 Locomotion](lessons/go2.md) — 7 lessons

- **Reward rebalancing when porting robots** — 10x tracking weights needed (Go1→Go2), pose dominated at 1x
- **Verify MJCF actuator limits against hardware** — Menagerie calf=24Nm, real Go2=45.43Nm (53% of real)
- **Env parity is not just reward math** — 5 physics issues found, all necessary but only reward rebalancing was sufficient
- **Entropy collapse was a symptom** — higher entropy_coef made training worse, fix was reward balance
- **Compare full env implementation, not just config** — diff collision geometry, termination, reward balance
- **Menagerie vs Playground contact physics** — solimp 0.015 (soft) vs 0.9 (firm), body collisions encourage crouching
- **Custom locomotion env integration** — subclass MjxEnv, scene XML adds missing sensors, Go2 vs Go1 naming
- **Training MJCF != deployment MJCF** — Menagerie go2_mjx.xml vs unitree_mujoco go2.xml are different physics; policy doesn't transfer
- **Actuator type mismatch is #1 sim2sim failure** — general (PD inside actuator) vs motor (raw torque); also 5x damping diff, condim 3 vs 6
- **Matching PD gains is not enough** — `general` (affine) vs `motor` (external PD) are NOT equivalent even at same rate/gains; retrain with `motor` actuators for transfer
- **DR covers parameter ranges, not model structure** — but "structural difference" was actually just damping=2 vs 0.1. Read the XML first.
- **Read the XML before numerical tests** — 20x damping diff found in 5 lines of XML, after hours of sim2sim experiments
- **MJX and CPU MuJoCo diverge over time** — not f32/f64 (tested), not settings. Bursty contact solver divergence at foot contact boundaries. DR + kicks for robustness.
