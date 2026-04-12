# Lessons Learned — Index

Navigable index of all project lessons. Each entry is a one-liner — click through for full context.
JAX/Flax fundamentals in `lessons/learner.md`.

---

## [PPO](lessons/ppo.md) — 17 lessons

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

## [Infrastructure](lessons/infrastructure.md) — 16 lessons

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
