# MuJoCo Engine Lessons

General MuJoCo properties that apply to ALL backends (CPU, MJX, Warp).

---

## Friction Uses Max-Combine — Randomize Foot Geoms, Not Just Floor (2026-03-28)

**What happened:** Domain randomization randomized floor friction U(0.2, 2.0) but the policy still failed on different surfaces. With aggressive range U(0.05, 4.5), training collapsed to eval 0.

**Root cause:** MuJoCo combines friction between colliding geoms using **element-wise max** (not multiply like PhysX). If foot friction is 0.6 and floor is 0.05, effective friction = max(0.6, 0.05) = 0.6. Floor-only randomization has no effect when foot friction caps it.

**Fix:** Randomize ALL geom friction (feet + floor + body) uniformly. Range [0.3, 1.5] (moderate). [0.05, 4.5] from WTW was designed for PhysX multiply-combine — too extreme for MuJoCo max-combine.

**Lesson:** DR ranges from Isaac Gym/Isaac Lab/PhysX papers are NOT directly portable to MuJoCo. The friction combining rule changes effective ranges dramatically. Always check the simulator's contact model before copying DR configs.

---

## Sim2sim Between Different MJCFs Is Harder Than Same-MJCF Transfer (2026-03-28)

**What happened:** Policy trained on Menagerie Go2 MJCF works perfectly on our CPU env (go2_cpu.py, 10s+ walking). Same policy fails within 2s on unitree_mujoco's Go2 MJCF, despite matching all overridable parameters (damping, friction, force limits, timestep, contacts).

**Root cause (investigated exhaustively):** The two MJCFs describe the same robot but with:
- Different solver defaults (pyramidal/1-iter vs elliptic/100-iter)
- Different collision geometry types (capsule vs cylinder on calf bodies)
- Different geom counts (57 vs 65)

Zero-torque test showed 2-3x joint velocity divergence after a single physics step. These are irreducible MJCF authoring differences — same robot, different model files, different dynamics.

**What worked:** Training on the target MJCF directly (via MuJoCo Warp, which supports cylinders unlike MJX). FastSAC on Warp: eval 276.5, walks 20s+ on CPU with the same MJCF.

**Lesson:** Sim2sim between your own envs (same MJCF, different backends) is easy. Sim2sim between different MJCFs of the "same" robot is nearly as hard as sim2real. Train on the target model directly when possible.

---

## Three Python APIs — Know Which One You're Using

MuJoCo has three distinct Python interfaces. They share the same physics engine but have different APIs:

| API | Import | Model type | Parallelism | Use case |
|---|---|---|---|---|
| **CPU MuJoCo** | `import mujoco` | `MjModel` / `MjData` (mutable) | Multiprocessing | Deploy, sim2sim, viewers |
| **MJX** (`impl="jax"` or `"warp"`) | `from mujoco import mjx` | `mjx.Model` / `mjx.Data` (immutable JAX pytrees) | `jax.vmap` / `jax.jit` | RL training (what we use) |
| **Standalone Warp** | `import mujoco_warp as mjw` | Warp-native model | `wp.launch` kernels | Pure simulation, no JAX |

**Our setup:** Playground envs → `mjx.put_model(m, impl="warp")` → Warp physics through the JAX API. Policy networks, vmap, jit, autodiff all stay in JAX. The `impl="warp"` flag swaps only the physics backend.

**The `mujoco_warp` tutorial notebook** (`mjw.put_model`) uses the standalone Warp API — same physics, different interface. You'd need manual data bridging (Warp→JAX) for RL training. Not what we want.

**Rule:** For RL training, always use `mjx` with `impl="warp"`. For standalone simulation/benchmarking without JAX, the `mjw` API is fine.
