# Lessons Learned

Practical lessons from building and tuning RL algorithms in JAX. These are non-obvious findings that cost hours or days to discover.

!!! note "Note on benchmark numbers"
    All eval scores in these docs and in `.context/AGENT_HANDOFF.md` are from single training runs (one seed each). This is sufficient for "did it work?" but not for publication-quality comparisons. Treat benchmark numbers as approximate — seed variance on these algorithms is typically 10-20% at the step counts reported.

---

## PPO

### Entropy coefficient is environment-dependent

A positive entropy bonus pushes `log_std` toward its upper bound, making the policy too noisy for precise control. CheetahRun with `entropy_coef=0.001` plateaued at 248 because entropy grew unboundedly. Setting `entropy_coef=0.0` jumped returns to 503.

**Rule of thumb:** Simple tasks (CartpoleBalance) benefit from `entropy_coef=0.01`. Complex continuous control often needs `entropy_coef=0.0`. Watch for monotonically increasing entropy as the diagnostic signal.

### Use state-independent log_std for stability

State-dependent std (a Dense layer mapping features to `log_std`) causes NaN on high-dimensional tasks like HumanoidRun (21 action dims). Outlier observations spike `log_std`, entropy explodes, and gradients diverge. State-independent std (a learned parameter vector) is more stable and matches the Brax PPO reference.

Save state-dependent std for SAC, where maximum entropy is the explicit objective.

### Normalize advantages over the full batch, not per-minibatch

Normalizing over 2k samples (per-minibatch) gives noisy mean/std estimates compared to normalizing over the full batch (30k-983k samples). This was one of two root causes of a 2x sample efficiency gap with Brax PPO.

### Value loss needs 0.25x scaling

Brax PPO uses `0.5 * 0.5 = 0.25x` value loss scaling. Without it, critic gradients are 4x too large, destabilizing training. This was the other root cause of the Brax performance gap. When a reference implementation outperforms yours, diff the code before tuning hyperparameters.

---

## Off-Policy (SAC / TD3)

### NEVER normalize observations before buffer storage

With off-policy replay, normalization statistics drift over time. Transitions stored with old statistics become corrupted when replayed under new statistics. This manifests as Q-values diverging to -15,000 and alpha spiking to 19+.

**The correct approach:** Store raw observations in the replay buffer, normalize at sample time using current statistics. The `--obs-norm` flag implements this pattern.

PPO is immune because it consumes observations in the same iteration they were collected.

### Replay ratio must scale with num_envs

TD3 with `grad_updates_per_step=1` and 128 parallel envs collects 128 samples per step but trains on only 256 (one batch). The model cannot keep up with the data rate.

**Rule:** Use 4-8 gradient steps per env step for 128+ envs. A 1:1 ratio only works for single-env training.

### target_entropy=0 for SAC at 1024+ envs

With the standard `target_entropy = -dim(A)` heuristic at large scale, the policy easily achieves that entropy level, alpha decays to zero, and exploration pressure is lost. Setting `target_entropy=0` keeps alpha active throughout training. FastSAC with `target_entropy=-3`: alpha collapsed to 0.012, eval peaked at 401. With `target_entropy=0`: alpha stayed at 0.65, eval 509.

### Truncation handling: mask the loss, zero the bootstrap

`EpisodeWrapper` from Brax/Playground sets `done = terminated OR truncated` and `truncation = truncated AND NOT terminated`. `done` already includes timeouts — `truncation` is the extra bit you need to distinguish real termination from pure timeout.

**Correct convention (SAC/TD3 here, matches Brax):**
```python
target = r + gamma * (1 - done) * V_next       # zero bootstrap on both
mask = 1.0 - truncation                        # drop pure-timeout rows from loss
loss = jnp.mean(per_sample_loss * mask)
```

Pure terminations contribute their `r`-only target (correct). Pure timeouts are dropped (next_obs is corrupted by AutoReset, bootstrap can't be trusted, and the `r`-only target would teach `Q = r` at timeout, which is wrong).

**The bug we had until 2026-04-12:** FastSAC/FastTD3/FlashSAC omitted the loss mask. On long-horizon tasks (Go2, Humanoid) this caused systematic Q underestimation proportional to `(timeout_rate × true_tail_value)`. Fixed in commit `82c9fe5`. Pre-fix benchmark numbers on long-horizon tasks may not be reproducible.

---

## Distributional RL (C51, FastTD3, FastSAC)

### C51 V_min/V_max is a hard boundary, not a soft one

C51 represents Q-values as a categorical distribution over `[V_min, V_max]`. True Q-values beyond these bounds cause all probability mass to pile at the boundary, making the critic blind. Default `V_min=-10, V_max=10` capped CheetahRun returns at 154.

**Sizing rule:** `Q_max ~ avg_reward_per_step / (1 - gamma)`. Use 2-3x this value for V_max. The paper default `[-20, 20]` works for most locomotion tasks at `gamma=0.97`.

### "Gaussian NLL" in papers may actually be Huber loss in code

The FastDSAC paper describes a "Gaussian distributional critic" but the source code uses a Huber-based loss with clamped ratio weighting. No `1/variance` division anywhere. Implementing Gaussian NLL as described caused NaN. Always read the actual source repository, not just the paper.

### FastTD3 needs scale to work

FastTD3 underperformed vanilla TD3 at small scale (128 envs, 5M steps): 285 vs 749. At paper scale (1024 envs, 86M steps): 880 vs 749. The algorithm's C51 critic and large batch sizes need sufficient gradient steps to amortize their overhead.

**Rule:** Match the paper's operating regime (1024+ envs, 50M+ steps) or use the simpler algorithm.

---

## JAX Performance

### Python collect loops kill throughput -- use lax.scan

A Python-level rollout loop forces GPU sync at every step, adding 87x overhead. Moving the entire collect loop (env.step + action selection + normalization) into `jax.lax.scan` gives up to 542x speedup on small MLPs where dispatch overhead dominates.

```
Python loops, no JIT:          2.712s per update
JIT + jax.lax.scan for epochs: 0.005s per update (542x faster)
```

### JIT closure recompilation in eval loops

Creating a new `@jax.jit`-decorated function inside a loop (e.g., capturing a changing `norm_state` in a closure) causes JIT cache misses. Each eval recompiled from scratch (~15s), projecting training to 2.5 hours instead of 15 minutes.

**Fix:** Define the eval function once, pass changing values as arguments. Same function object = cache hit.

### lax.scan carry cost scales with array size

Putting large arrays (4M-entry replay buffers) in `lax.scan` carry makes it 30% slower than a Python loop. The carry overhead dominates at that size. Keep large state outside the scan; accept Python loops when the carry would be large.

---

## MuJoCo / Warp

### Friction uses max-combine -- randomize ALL geoms, not just floor

MuJoCo combines friction between colliding geoms using element-wise max (not multiply like PhysX). If foot friction is 0.6 and floor friction is 0.05, effective friction is `max(0.6, 0.05) = 0.6`. Floor-only domain randomization has no effect when foot friction caps it.

**Fix:** Randomize friction on all geoms (feet + floor + body). Also note that DR ranges from Isaac Gym/PhysX papers are not directly portable to MuJoCo due to the different combining rule.

### Joint order does not equal actuator order in unitree MJCF

In unitree's `go2.xml`, `qpos[7:]` follows body-tree order (FL, FR, RL, RR) but `ctrl` follows actuator order (FR, FL, RR, RL). A PD controller reading `qpos` and writing `ctrl` without remapping sends each leg's correction to the wrong actuator. The robot fights itself.

This was the single most impactful bug of the Warp migration. The robot partially learned (eval 211) by finding a static strategy, making it look like a tuning problem rather than a wiring bug.

**Fix:** Build a mapping from `model.actuator_trnid` and remap joint-space torques to actuator-space before writing to `ctrl`.

### PD gains are coupled to solver stiffness

Kp/Kd tuned on MJX's 1-iteration pyramidal solver (Kp=35, Kd=0.1) caused collapse on Warp's 100-iteration elliptic solver. The stiff contacts transmit full robot weight through joints, requiring different gain ratios.

**Reference gains:** Kp=35/Kd=0.1 for MJX (Menagerie), Kp=20/Kd=0.5 for Warp (unitree). Always use gains from the same solver configuration.

### Warp CCD overflow needs explicit sizing

Unitree's full collision geometry produces many more geom pairs than Menagerie's sphere-only model. The default CCD buffer is too small, causing ~30% throughput loss from overflow retries and silently dropped contacts.

**Fix:** Set `naccdmax=4000` (or 2x the max overflow value observed). The warnings are not just noise -- they indicate dropped contacts that affect physics fidelity.

---

## Sim-to-Real

### Actuator type mismatch is the #1 transfer failure

Training with `general` actuators (Menagerie default) and deploying to real `motor` actuators creates a systematic mismatch. The trained policy assumes dynamics that don't exist on the real robot.

**Fix:** Train on the target MJCF directly. For Go2, this means using the unitree MJCF via Warp rather than the simplified Menagerie MJCF via MJX.

### Domain randomization covers parameter ranges, not model structure

Sim-to-sim between different MJCFs of the "same" robot is nearly as hard as sim-to-real. Different solver defaults, collision geometry types, and geom counts create irreducible dynamics differences that domain randomization cannot bridge.

A policy trained on Menagerie's Go2 (walks 10s+) failed within 2s on unitree's Go2 despite matching all overridable parameters. Training directly on the unitree MJCF via Warp: eval 276.5, walks 20s+.
