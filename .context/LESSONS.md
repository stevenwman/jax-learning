# Lessons Learned — Index

Navigable index of all project lessons. Each entry is a one-liner — click through for full context.
JAX/Flax fundamentals in `LEARNER_LESSONS.md`.

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

## [Off-Policy (SAC / TD3)](lessons/offpolicy.md) — 8 lessons

- **Obs normalization: NEVER before buffer storage** — normalize at sample time with `--obs-norm` (Go2: 139 vs 97)
- **SAC validation results** — WalkerWalk 975, HumanoidRun 426 (vanilla) / 892 (FastSAC)
- **TD3 needs gradient clipping on high-dim tasks** — SAC's entropy provides implicit stability TD3 lacks
- **Replay ratio must scale with num_envs** — 1:1 for 128 envs = 1 grad step per 128 samples, use 4-8x
- **TD3 exploration fails at >10 action dims** — SAC's structured exploration wins on humanoid-class tasks
- **Target entropy = 0 for SAC at scale** — classic `-dim(A)` causes alpha collapse at 1024 envs / 100M steps
- **AdamW requires `params` in optimizer.update()** — weight decay needs the params themselves
- **Optimizer decoupling** — algorithms define what to optimize, not how (DI pattern)

## [Distributional RL (C51 / FastTD3 / FastSAC / FastDSAC)](lessons/distributional.md) — 8 lessons

- **FastTD3 scale matters** — 285 eval at 128 envs/5M steps, **880** at 1024 envs/86M steps
- **C51 V_min/V_max is critical** — distributional Q is hard-bounded, get it wrong and critic is blind
- **C51 cross-entropy NaN** — `log_softmax` → `-inf`, then `-inf * 0 = NaN`. Clamp to -30.
- **SAC variants can't match FastTD3 on low-dim tasks** — expected, test on HumanoidRun instead
- **Run simple baseline before fancy version** — vanilla SAC 771 in 8 min vs FastSAC 582 in 108 min
- **Verify configs against source code** — 7+ critical mismatches found (tau 25x wrong, hidden dims, activation)
- **Read the whole recipe** — missing gamma=0.97 and AdamW β2=0.95 completely changed behavior
- **FastDSAC paper says "Gaussian NLL" but code uses Huber** — 2 days debugging the wrong loss

## [JAX Performance](lessons/jax_performance.md) — 8 lessons

- **Python collect loops kill throughput** — 87x overhead from Python-GPU sync, use lax.scan
- **lax.scan vs Python loops: 542x speedup** — for small MLPs, 95% of time is dispatch overhead
- **JIT closure recompilation in eval loops** — new closure = cache miss = 15s recompile each eval
- **lax.scan episode return tracking** — state spanning episode boundaries must persist outside scan
- **Python `if` vs `jax.lax.cond`** — TracerBoolConversionError inside scan/vmap, use lax.cond
- **JIT the hot path** — non-JIT'd JAX is 5x slower than numpy for small ops
- **lax.scan carry cost** — 4M-entry buffer in carry = 30% slower than Python loop
- **Faster component ≠ faster training** — 4.8x buffer speedup = 1.5% end-to-end improvement

## [Infrastructure](lessons/infrastructure.md) — 9 lessons

- **Orbax checkpointing** — must call `wait_until_finished()`, save meta.json alongside
- **Orbax restore needs exact pytree match** — separate inference artifacts (numpy) from training (orbax)
- **Checkpoint should be self-describing** — store full `dataclasses.asdict(cfg)` in meta.json
- **Video recording: two-phase** — GPU rollout via lax.scan, then CPU rendering
- **Save trajectory data alongside videos** — two policies with same eval=11.6 had completely different behaviors
- **Eval/recording must match training preprocessing** — missing obs norm → instant death on video
- **Scientific notation for metrics** — `{:8.2f}` printed `0.00` for VLoss=0.003
- **Integer division truncation** — `200000 // 128 * 128 = 199936`, final eval never fired
- **Verify training budget before debugging** — eval ~17 at 50M steps was on-curve, not broken

## [MJX Physics](lessons/mjx.md) — 3 lessons

- **MJX physics NaN at scale** — stochastic contact solver failure, guard with NaN+Inf checks on env boundary
- **GPU OOM is usually not a leak** — RTX 5080 starts at 95% capacity, XLA command buffers accumulate
- **MJX eval recompilation** — upstream issue, ~2 recompiles per eval call, mitigate with MEM_FRACTION=0.7

## [Go2 Locomotion](lessons/go2.md) — 7 lessons

- **Reward rebalancing when porting robots** — 10x tracking weights needed (Go1→Go2), pose dominated at 1x
- **Verify MJCF actuator limits against hardware** — Menagerie calf=24Nm, real Go2=45.43Nm (53% of real)
- **Env parity is not just reward math** — 5 physics issues found, all necessary but only reward rebalancing was sufficient
- **Entropy collapse was a symptom** — higher entropy_coef made training worse, fix was reward balance
- **Compare full env implementation, not just config** — diff collision geometry, termination, reward balance
- **Menagerie vs Playground contact physics** — solimp 0.015 (soft) vs 0.9 (firm), body collisions encourage crouching
- **Custom locomotion env integration** — subclass MjxEnv, scene XML adds missing sensors, Go2 vs Go1 naming
