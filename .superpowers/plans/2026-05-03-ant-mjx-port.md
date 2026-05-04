# Ant MJX Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `AntMJX` environment to our MJX/Warp pipeline that is a faithful behavioral port of `gymnasium.envs.mujoco.ant_v5.AntEnv`. Same observation, reward, termination, init distribution as Gym Ant-v5, but GPU-parallelizable through our existing MJX bundle. Will serve as the visual-gate env for SD-B (xy-trajectory diversity figure) and as a future canonical baseline.

**⚠️ Env-name discipline:** the registered string is **`AntMJX`** (NOT `"Ant"`). Reason: `gym_backend.py:200` already registers `"Ant"` for the CPU Gym lane, and `detect_backend("Ant")` returns `"gym"` — using `"Ant"` would silently route every command to CPU Gym, never reaching this MJX path. Python class is `Ant` (clean; class name doesn't conflict). Wherever this plan says "env_name", it's the literal string `AntMJX`. **Re-grep** if confused.

**Why this and not "wrap Gym Ant on CPU":** the SD-B Wave D acceptance runs (2026-05-03, see `.context/journals/2026-05-03-skill-discovery-sd-b-progress.md`) closed Gate 1 numerically on CheetahRun but flagged Gate 2 (qualitative video diversity) as visually ambiguous on a planar 2D body. Ant's xy-trajectory plot is the canonical DIAYN visual-diversity figure (commonly cited as Fig. 4 / App. D.3 of Eysenbach et al. 2018, "Diversity is All You Need" — but **eyeball-verify against the paper PDF before the figure caption ships**, see Risk register entry "Citation drift"; same trap as the prior D.4→D.3 incident). CPU-Gym route was punted earlier; user pushed back: physics is the same, port the env directly to MJX/Warp so it reuses our existing 600 sps pipeline and `train_skill_discovery.py` plumbing.

**Spec source (canonical):** `gymnasium/envs/mujoco/ant_v5.py` is the contract. We do NOT invent semantics — we mirror that file. Where Gym uses `numpy`, we use `jax.numpy`; where Gym uses `mujoco.MjData`, we use `mjx.Data`. Behavioral equivalence in distribution (same obs/reward/termination logic, same scalars), NOT bit-identity (GPU non-determinism known: see `lessons/determinism.md`, `feedback_gpu_nondeterminism.md`).

**Architecture:** Free-standing env — does NOT subclass `Go2WarpBase` or any locomotion class. New file `jax_rl/envs/locomotion/ant.py` subclasses `mjx_env.MjxEnv` directly (same pattern as `mujoco_playground/_src/dm_control_suite/cheetah.py:47`). XML lives in `jax_rl/envs/locomotion/xmls/ant.xml`. Backend dispatch: there are **THREE** call sites that resolve env_name → env instance, and ALL must be patched:
1. `jax_rl/training/env_backends/mjx_backend.py:190` (training env, in `make_envs`)
2. `jax_rl/training/env_backends/mjx_backend.py:215` (eval env, same function)
3. `scripts/record_video.py:334` (one-shot video render path; uses `pg_registry.load` directly)

The cleanest fix is a single helper `maybe_load_custom_env(env_name) -> Optional[MjxEnv]` (public — record_video.py imports it cross-module) imported by all three sites; falls through to `pg_registry.load()` when it returns None. **Verify line numbers by `grep "pg_registry.load" jax_rl/training/env_backends/mjx_backend.py scripts/record_video.py` — do not blindly seek; line numbers drift.**

**Tech stack:** JAX/Flax, MuJoCo Warp via `mjx.put_model(..., impl="warp")`. Tests use pytest. CPU-Gym Ant required for parity reference (already in `jax_rl/training/env_backends/gym_backend.py:200`).

---

## Hard rules (do not violate)

- `uv run python <cmd>` always.
- No Co-Authored-By lines in commits (`MEMORY.md`).
- Behavioral equivalence in distribution, NOT bit-identity. Use `assert_allclose(rtol=1e-3, atol=1e-3)` style on aggregate statistics (mean, std), NOT per-step matching, when comparing CPU Gym vs MJX.
- Default config matches Gym Ant-v5 defaults exactly; deviation requires a comment + journal entry.
- ALL gym scalars locked from `ant_v5.py` source, not from docstrings or papers. Re-grep on edits.
- No Brax import. We use Brax's `ant.xml` as a *reference text file*, not as a library.
- Match `frame_skip=5` semantics: each `step()` advances 5 physics substeps via `mjx_env.step(model, data, action, n_substeps=5)`.
- Truncation handled by `wrap_for_training(episode_length=1000)` (existing pipeline), not inside the env.

---

## Locked spec (re-grep before deviating)

From `gymnasium/envs/mujoco/ant_v5.py` (current install):

```python
forward_reward_weight = 1
ctrl_cost_weight      = 0.5
contact_cost_weight   = 5e-4
healthy_reward        = 1.0
healthy_z_range       = (0.2, 1.0)
contact_force_range   = (-1.0, 1.0)
reset_noise_scale     = 0.1
exclude_current_positions_from_observation = True   # default True → 105d obs (skip xy)
include_cfrc_ext_in_observation             = True
frame_skip            = 5
main_body             = 1   # torso
terminate_when_unhealthy = True
```

**Reward (per step):**
```
forward_reward = ((xpos_after - xpos_before)[0] / dt) * forward_reward_weight
healthy_reward = healthy_reward_w if is_healthy else 0.0
ctrl_cost      = ctrl_cost_weight * sum(action**2)
contact_cost   = contact_cost_weight * sum(clip(cfrc_ext, -1, 1)**2)
reward         = forward_reward + healthy_reward - ctrl_cost - contact_cost
```

**`is_healthy`:** `isfinite(state).all() and 0.2 <= state[2] <= 1.0` where `state = concat(qpos, qvel)`.

**Reset (asymmetric form, shared scale):**
```
qpos = init_qpos + uniform(-reset_noise_scale, reset_noise_scale, size=nq)   # uniform-bounded
qvel = init_qvel + reset_noise_scale * standard_normal(size=nv)              # unbounded gaussian
```
Both terms share `reset_noise_scale=0.1`; the asymmetry is in the *shape* of the noise (uniform vs gaussian), NOT in the magnitude. Verbatim from `ant_v5.py:411-417`.

**`init_qpos` source — important subtlety:** `ant_v5.py` calls `MujocoEnv.__init__`, which sets `self.init_qpos = self.data.qpos.ravel().copy()` after `mj_resetData`. That value is the free-joint default = the torso body's `pos="0 0 0.75"` attribute. **Result: `init_qpos[2] = 0.75`** (NOT 0.55). The `<custom name="init_qpos">` numeric in Brax's XML *says* 0.55, but stock MuJoCo (and our MJX path) ignores that `<custom>` element entirely — it's Brax-only metadata. Our env reads `mj_model.qpos0` which is also derived from body pos, giving 0.75, matching Gym. **Do NOT rely on the Brax `<custom>` numeric.**

**Obs (default 105d, exclude xy + include cfrc_ext):**
```
qpos[2:]          # 13 dims
qvel              # 14 dims
cfrc_ext[1:].flatten()    # (nbody-1)*6 = 13*6 = 78 dims
```
Total = 105.

**Forward velocity uses CoM xpos, NOT qpos:**
```
xy_before = data.xpos[main_body, :2]   # before mjx step
substeps for frame_skip=5
xy_after  = data.xpos[main_body, :2]
x_velocity = (xy_after[0] - xy_before[0]) / dt   # dt = 0.05
```
This matters: `qpos[:2]` is the joint state of the free joint; `xpos[1, :2]` is the body's world-frame CoM position. They differ for free joints.

---

## Stage 0: Pre-flight

### Task 0.1: Verify default test lane is green before changes

**Files:** none

- [ ] **Step 1:** `JAX_PLATFORMS=cpu uv run python -m pytest -q`. Record pass count. Your changes must not lower it.
- [ ] **Step 2:** Confirm Gym Ant CPU works as reference, AND verify nbody/obs dim before locking spec:
  ```bash
  uv run python -c "
  import gymnasium as gym
  env = gym.make('Ant-v5')
  print('obs.shape:', env.observation_space.shape)
  print('nbody:', env.unwrapped.model.nbody)
  print('nq, nv, nu:', env.unwrapped.model.nq, env.unwrapped.model.nv, env.unwrapped.model.nu)
  "
  ```
  Expected: `obs.shape: (105,)`, `nbody: 14`, `nq, nv, nu: 15 14 8`. **If observed obs is NOT 105 or nbody is NOT 14, STOP and reconcile** — the locked spec assumes 105 and (nbody-1)*6=78 cfrc terms; any deviation propagates through Stage 2 and Stage 3.
- [ ] **Step 3:** Verify `cfrc_ext` shape on our XML (Brax variant) before relying on it in obs:
  ```bash
  uv run python -c "
  import mujoco
  from mujoco import mjx
  m = mujoco.MjModel.from_xml_path('jax_rl/envs/locomotion/xmls/ant.xml')
  mx = mjx.put_model(m, impl='warp')
  d = mjx.make_data(mx)
  print('cfrc_ext.shape:', d.cfrc_ext.shape)   # expect (14, 6)
  "
  ```
  *Run this AFTER Stage 1.1 lands the XML.* If shape is anything other than `(14, 6)`, the obs-dim plan and parity test must be revised.
- [ ] **Step 4:** Confirm `mujoco_playground.dm_control_suite.cheetah.Run` is the reference port pattern: `uv run python -c "from mujoco_playground._src.dm_control_suite.cheetah import Run; print(Run.__init__.__doc__)"`.
- [ ] **Step 5:** Confirm `"AntMJX"` is NOT one of the existing gym names — it'll route to MJX by default:
  ```bash
  uv run python -c "
  from jax_rl.training.env_backends import detect_backend
  print('AntMJX detected as:', detect_backend('AntMJX'))   # expect 'mjx' (default fallthrough)
  print('Ant detected as:',    detect_backend('Ant'))      # expect 'gym' (registered)
  print('CheetahRun detected as:', detect_backend('CheetahRun'))   # expect 'mjx' (default)
  "
  ```
  Expected: `AntMJX` → `mjx` (the default-fallthrough in `env_backends/__init__.py` routes any non-`Gym/`-prefix non-IsaacLab name to MJX unless it's in `_GYM_ENV_NAMES`), `Ant` → `gym` (because `gym_backend.py:200` registered it), `CheetahRun` → `mjx`. **No registration step is needed for `AntMJX`** — the default routing covers it. Stage 4.1 Step 1 below is therefore a verification-only step, not a code-edit.

---

## Stage 1: XML

### Task 1.1: Land `ant.xml` in repo

**Files:**
- Create: `jax_rl/envs/locomotion/xmls/ant.xml`

- [ ] **Step 1:** Start from `.venv/lib/python3.13/site-packages/brax/envs/assets/ant.xml` (it has the foot-contact spheres MJX needs; gym's plain `ant.xml` lacks them and feet ghost through floor in Warp).
- [ ] **Step 2:** Strip ALL Brax-only `<custom>` tags. Verbatim list to remove (all 11 `<numeric>` elements in Brax's `<custom>` block):
  - `constraint_limit_stiffness`, `constraint_stiffness`, `constraint_ang_damping`, `constraint_vel_damping`
  - `joint_scale_pos`, `joint_scale_ang`
  - `ang_damping`, `spring_mass_scale`, `spring_inertia_scale`
  - `solver_maxls`
  - `init_qpos` ← **also strip** (vestigial; stock MuJoCo ignores `<custom>` numerics, and our `_post_init` reads `mj_model.qpos0` — keeping the Brax tag is misleading because it claims z=0.55 while real init is z=0.75 from torso body pos).
  After strip, the entire `<custom>` block is empty — drop the surrounding `<custom>...</custom>` element too.
- [ ] **Step 3:** Verify the `<option>` line. Brax's XML has `<option timestep="0.01" iterations="4"/>` (no integrator specified → defaults to Euler). Gym's XML has `<option integrator="RK4" timestep="0.01"/>` (RK4, no iterations). **Plan default is to MATCH Gym (RK4) first.** Try RK4 in MJX/Warp and only fall back to Euler if RK4 fails or produces unstable contact dynamics. To set RK4 explicitly:
  ```xml
  <option integrator="RK4" timestep="0.01" iterations="4"/>
  ```
  If RK4 fails Stage 1 step 5 (`mjx.put_model`) or causes contact_cost > 100× Gym at parity test, swap to `integrator="Euler"` and journal the deviation. **Do NOT preemptively deviate from Gym's RK4** — Stage 3 parity test is the gate.
- [ ] **Step 4:** Verify XML loads:
  ```python
  import mujoco
  m = mujoco.MjModel.from_xml_path("jax_rl/envs/locomotion/xmls/ant.xml")
  print(m.nq, m.nv, m.nu, m.nbody)   # Expect: 15 14 8 14
  ```
- [ ] **Step 5:** Verify MJX/Warp put_model works:
  ```python
  from mujoco import mjx
  mx = mjx.put_model(m, impl="warp")
  print("OK")
  ```
  If this fails on cylinder-box collisions or similar, see `lessons/mjx.md` "MJX can't load all MJCFs" — but ant is all capsules+spheres+plane so should not.

### Task 1.2: Test the XML loads in our pipeline

**Files:**
- Create: `tests/test_ant_xml.py`

- [ ] **Step 1: Write the failing test.** Per `.context/lessons/testing_new_envs.md`, **any test that calls `mjx.put_model(impl="warp")` MUST be marked `[gpu, warp]`** or the default hermetic CPU lane will load Warp on a machine without a GPU and crash. Split the cheap dim test from the put_model test:

  ```python
  """Hermetic test that ant.xml loads + has expected geometry."""

  import mujoco
  import pytest
  from mujoco import mjx
  from pathlib import Path


  ANT_XML = Path("jax_rl/envs/locomotion/xmls/ant.xml")


  def test_ant_xml_dims():
      """Cheap CPU-only structural check; no GPU required."""
      m = mujoco.MjModel.from_xml_path(str(ANT_XML))
      assert m.nq == 15  # 7 free joint + 8 hinge
      assert m.nv == 14  # 6 free joint vel + 8 hinge vel
      assert m.nu == 8   # 8 actuators
      assert m.nbody == 14  # worldbody + torso + 12 leg parts


  @pytest.mark.gpu
  @pytest.mark.warp
  def test_ant_xml_warp_putmodel():
      """Warp-backed put_model — requires GPU + Warp deps."""
      m = mujoco.MjModel.from_xml_path(str(ANT_XML))
      mx = mjx.put_model(m, impl="warp")
      assert mx.nq == 15
  ```
- [ ] **Step 2:** Run the cheap test: `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_ant_xml.py::test_ant_xml_dims -v`. Expect green.
- [ ] **Step 3:** Run the GPU test: `uv run python -m pytest tests/test_ant_xml.py::test_ant_xml_warp_putmodel -v -m "gpu and warp"`. Expect green on GPU host. **The `-m "gpu and warp"` marker selector is required to opt INTO the GPU lane.**

---

## Stage 2: AntEnv class

### Task 2.1: Skeleton + `__init__` + `_post_init`

**Files:**
- Create: `jax_rl/envs/locomotion/ant.py`
- Test: `tests/test_ant_env.py`

- [ ] **Step 1:** Write the env skeleton mirroring `cheetah.py`. Include all Gym-v5 config knobs as ml_collections fields:

  ```python
  """Ant environment — port of gymnasium AntEnv-v5 to MJX/Warp."""

  from typing import Any, Dict, Optional, Union
  from pathlib import Path

  import jax
  import jax.numpy as jp
  from ml_collections import config_dict
  import mujoco
  from mujoco import mjx
  from mujoco_playground._src import mjx_env

  _XML_PATH = Path(__file__).parent / "xmls" / "ant.xml"


  def default_config() -> config_dict.ConfigDict:
      return config_dict.create(
          ctrl_dt=0.05,        # frame_skip * sim_dt = 5 * 0.01
          sim_dt=0.01,
          episode_length=1000,
          action_repeat=1,     # frame_skip lives inside step() via n_substeps
          vision=False,
          impl="warp",
          # Warp contact-buffer sizing (mirrors mujoco_playground cheetah.py:42-43):
          naconmax=100_000,
          njmax=100,
          # Gym Ant-v5 defaults (locked from ant_v5.py):
          forward_reward_weight=1.0,
          ctrl_cost_weight=0.5,
          contact_cost_weight=5e-4,
          healthy_reward=1.0,
          healthy_z_min=0.2,
          healthy_z_max=1.0,
          contact_force_clip=1.0,    # symmetric clip [-1, 1]
          reset_noise_scale=0.1,
          exclude_current_positions_from_observation=True,
          include_cfrc_ext_in_observation=True,
          terminate_when_unhealthy=True,
          main_body_id=1,            # torso
      )


  class Ant(mjx_env.MjxEnv):
      """Gym Ant-v5 port (registered to env_name 'AntMJX')."""

      def __init__(self, config=None, config_overrides=None):
          if config is None:
              config = default_config()
          super().__init__(config, config_overrides)
          self._xml_path = _XML_PATH.as_posix()
          self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
          self._mj_model.opt.timestep = self.sim_dt
          self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
          self._post_init()

      def _post_init(self) -> None:
          # init_qpos comes from mj_model.qpos0 (= torso body's pos="0 0 0.75"),
          # NOT from any <custom name="init_qpos"> tag (Brax-only metadata that
          # stock MuJoCo ignores; keeping it in the XML would mislead readers
          # since it claims z=0.55 while the real init is z=0.75). Matches
          # gym ant_v5 behavior: MujocoEnv.__init__ does
          # self.init_qpos = self.data.qpos.ravel().copy() after mj_resetData.
          self._init_qpos = jp.asarray(self._mj_model.qpos0)
          # init_qvel: gym ant.xml has no <numeric init_qvel> and Gym defaults
          # to zeros via MujocoEnv.set_state convention.
          self._init_qvel = jp.zeros(self._mj_model.nv)

      @property
      def xml_path(self) -> str:
          return self._xml_path

      @property
      def action_size(self) -> int:
          return self.mjx_model.nu

      @property
      def mj_model(self) -> mujoco.MjModel:
          return self._mj_model

      @property
      def mjx_model(self) -> mjx.Model:
          return self._mjx_model
  ```

- [ ] **Step 2:** Failing test. **Module-level `pytestmark` is required** — every `Ant()` instantiation calls `mjx.put_model(impl="warp")` (GPU + Warp). The default hermetic CPU lane will crash without these markers.
  ```python
  """tests/test_ant_env.py — Ant env tests (GPU + Warp lane)."""

  import jax
  import jax.numpy as jp
  import pytest

  from jax_rl.envs.locomotion.ant import Ant, default_config


  pytestmark = [pytest.mark.gpu, pytest.mark.warp]


  def test_ant_init():
      env = Ant()
      assert env.action_size == 8
      assert env.mjx_model.nq == 15
      assert env.mjx_model.nv == 14


  def test_ant_default_config_matches_gym_v5():
      cfg = default_config()
      assert cfg.forward_reward_weight == 1.0
      assert cfg.ctrl_cost_weight == 0.5
      assert cfg.contact_cost_weight == 5e-4
      assert cfg.healthy_reward == 1.0
      assert cfg.healthy_z_min == 0.2
      assert cfg.healthy_z_max == 1.0
      assert cfg.reset_noise_scale == 0.1
      assert cfg.exclude_current_positions_from_observation is True
      assert cfg.include_cfrc_ext_in_observation is True
  ```
- [ ] **Step 3:** `uv run python -m pytest tests/test_ant_env.py::test_ant_init -v -m "gpu and warp"` then `::test_ant_default_config_matches_gym_v5 -v -m "gpu and warp"`. Both green.

### Task 2.2: `reset()` with asymmetric noise

**Files:** `jax_rl/envs/locomotion/ant.py` (extend), `tests/test_ant_env.py` (extend).

- [ ] **Step 1:** Implement reset:
  ```python
  def reset(self, rng: jax.Array) -> mjx_env.State:
      rng, rng_q, rng_v = jax.random.split(rng, 3)

      # Asymmetric noise (Gym v5):
      noise_q = jax.random.uniform(
          rng_q, (self.mjx_model.nq,),
          minval=-self._config.reset_noise_scale,
          maxval=self._config.reset_noise_scale,
      )
      noise_v = self._config.reset_noise_scale * jax.random.normal(
          rng_v, (self.mjx_model.nv,)
      )
      qpos = self._init_qpos + noise_q
      qvel = self._init_qvel + noise_v

      # Forward naconmax/njmax to size Warp contact buffers (cheetah.py:91-92).
      # NOT forwarding makes default_config()'s naconmax/njmax dead config.
      data = mjx_env.make_data(
          self.mj_model,
          qpos=qpos,
          qvel=qvel,
          impl=self.mjx_model.impl.value,
          naconmax=self._config.naconmax,
          njmax=self._config.njmax,
      )
      data = mjx.forward(self.mjx_model, data)

      # info is intentionally minimal — step() recomputes xy_before from
      # state.data.xpos directly each time, so no need to stash it here.
      info = {"rng": rng}
      metrics = {
          "reward_forward": jp.zeros(()),
          "reward_ctrl": jp.zeros(()),
          "reward_contact": jp.zeros(()),
          "reward_survive": jp.zeros(()),
      }
      reward, done = jp.zeros(2)
      obs = self._get_obs(data)
      return mjx_env.State(data, obs, reward, done, metrics, info)
  ```
- [ ] **Step 2:** Failing test — verify reset is finite, qpos[2] (z) ≈ 0.75, noise distribution covers expected range:
  ```python
  def test_ant_reset_finite():
      env = Ant()
      state = env.reset(jax.random.PRNGKey(0))
      assert jp.all(jp.isfinite(state.obs))
      assert state.reward == 0.0
      assert state.done == 0.0


  def test_ant_reset_z_near_initial():
      env = Ant()
      # 100 different seeds → z should cluster around 0.75 ± 0.1 (uniform noise).
      # Bounds are 0.649/0.851 (not 0.65/0.85) to avoid spurious failures from
      # float-precision when the uniform sampler hits the endpoints exactly.
      keys = jax.random.split(jax.random.PRNGKey(0), 100)
      states = jax.vmap(env.reset)(keys)
      zs = states.data.qpos[:, 2]
      assert jp.all(zs >= 0.649) and jp.all(zs <= 0.851), \
          f"z range: {zs.min():.3f}, {zs.max():.3f}"
  ```
- [ ] **Step 3:** Run tests, green.

### Task 2.3: `step()` with frame_skip + reward + termination

**Files:** `ant.py` (extend), `tests/test_ant_env.py` (extend).

- [ ] **Step 1:** Implement step. **Compute `unhealthy` once and pass through** (avoid double-eval of the same predicate). Use `jnp.logical_not` rather than Python `~` on JAX bool arrays (safer dtype semantics under JIT). **Wire `terminate_when_unhealthy` config flag** into the `done` computation so the flag is not dead.
  ```python
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
      # Capture pre-step CoM xy:
      xy_before = state.data.xpos[self._config.main_body_id, :2]

      # Advance frame_skip=5 substeps:
      data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

      xy_after = data.xpos[self._config.main_body_id, :2]
      dt = self.dt   # ctrl_dt = 0.05
      x_velocity = (xy_after[0] - xy_before[0]) / dt
      y_velocity = (xy_after[1] - xy_before[1]) / dt

      # Single pass through health check; reuse for survive_r and done:
      unhealthy = self._is_unhealthy(data)
      reward, reward_info = self._get_reward(data, action, x_velocity, unhealthy)
      obs = self._get_obs(data)

      if self._config.terminate_when_unhealthy:
          done = unhealthy.astype(jp.float32)
      else:
          done = jp.zeros((), dtype=jp.float32)

      # Match Gym Ant-v5 info dict (ant_v5.py:360-367):
      info = {**state.info,
              "x_position": data.qpos[0],
              "y_position": data.qpos[1],
              "distance_from_origin": jp.linalg.norm(data.qpos[0:2]),
              "x_velocity": x_velocity,
              "y_velocity": y_velocity}
      metrics = {**reward_info}

      return mjx_env.State(data, obs, reward, done, metrics, info)

  def _is_unhealthy(self, data) -> jax.Array:
      state = jp.concatenate([data.qpos, data.qvel])
      finite = jp.all(jp.isfinite(state))
      z = data.qpos[2]
      in_z = (z >= self._config.healthy_z_min) & (z <= self._config.healthy_z_max)
      healthy = finite & in_z
      return jp.logical_not(healthy)

  def _get_reward(self, data, action, x_velocity, unhealthy):
      forward_r = x_velocity * self._config.forward_reward_weight
      healthy = jp.logical_not(unhealthy)
      survive_r = jp.where(healthy, self._config.healthy_reward, 0.0)
      ctrl_c = self._config.ctrl_cost_weight * jp.sum(action**2)
      cfrc = jp.clip(data.cfrc_ext, -self._config.contact_force_clip,
                     self._config.contact_force_clip)
      contact_c = self._config.contact_cost_weight * jp.sum(cfrc**2)
      reward = forward_r + survive_r - ctrl_c - contact_c
      info = {
          "reward_forward": forward_r,
          "reward_survive": survive_r,
          "reward_ctrl": -ctrl_c,      # negated to match Gym v5 ant_v5.py:387
          "reward_contact": -contact_c,
      }
      return reward, info
  ```
- [ ] **Step 2:** Failing test — step is finite, reward components in info, terminates when ant flips:
  ```python
  def test_ant_step_finite():
      env = Ant()
      state = env.reset(jax.random.PRNGKey(0))
      action = jp.zeros(env.action_size)
      next_state = env.step(state, action)
      assert jp.all(jp.isfinite(next_state.obs))
      assert jp.isfinite(next_state.reward)
      for key in ("reward_forward", "reward_survive", "reward_ctrl", "reward_contact"):
          assert key in next_state.metrics


  def test_ant_terminates_below_z_floor():
      env = Ant()
      # Hand-craft a state with z=0.1 (below healthy_z_min=0.2) → done=1
      state = env.reset(jax.random.PRNGKey(0))
      bad_qpos = state.data.qpos.at[2].set(0.1)
      bad_data = state.data.replace(qpos=bad_qpos)
      bad_state = state.replace(data=bad_data)
      next_state = env.step(bad_state, jp.zeros(env.action_size))
      assert next_state.done == 1.0
  ```
- [ ] **Step 3:** Run, green.

### Task 2.4: `_get_obs()` with cfrc_ext flatten + xy exclusion

**Files:** `ant.py` (extend), `tests/test_ant_env.py` (extend).

- [ ] **Step 1:** Implement obs:
  ```python
  def _get_obs(self, data) -> jax.Array:
      qpos = data.qpos
      qvel = data.qvel
      if self._config.exclude_current_positions_from_observation:
          qpos = qpos[2:]   # skip xy
      parts = [qpos, qvel]
      if self._config.include_cfrc_ext_in_observation:
          # Skip worldbody (index 0); flatten (nbody-1, 6) → 78d
          cfrc = data.cfrc_ext[1:].flatten()
          parts.append(cfrc)
      return jp.concatenate(parts)
  ```
- [ ] **Step 2:** Failing test — obs is 105d default:
  ```python
  def test_ant_obs_default_dim():
      env = Ant()
      state = env.reset(jax.random.PRNGKey(0))
      assert state.obs.shape == (105,)


  def test_ant_obs_exclude_xy_off():
      cfg = default_config()
      cfg.exclude_current_positions_from_observation = False
      env = Ant(config=cfg)
      state = env.reset(jax.random.PRNGKey(0))
      assert state.obs.shape == (107,)


  def test_ant_obs_no_cfrc():
      cfg = default_config()
      cfg.include_cfrc_ext_in_observation = False
      env = Ant(config=cfg)
      state = env.reset(jax.random.PRNGKey(0))
      assert state.obs.shape == (27,)   # 13 qpos[2:] + 14 qvel
  ```
- [ ] **Step 3:** Run, green.

---

## Stage 3: Behavioral parity

The point of this stage is to catch scalar mismatches and cfrc_ext sign errors before training. CPU Gym Ant is the ground truth for *distributional* equivalence.

### Task 3.1: CPU-Gym vs MJX behavioral parity test

**Files:** `tests/test_ant_parity.py`

**Critical test-design rules** (informed by audit findings):

1. **Use small random actions, NOT action=0.** At action=0 ctrl_cost is exactly 0, forward velocity is dominated by tiny init xy-velocity (mean ≈ 0), and the relative-tolerance check `< 10% * (|gym.mean()| + 1e-6)` becomes `< 1e-7` — trivially false-positive. Action range `[-0.3, 0.3]` exercises ctrl_cost and produces non-zero forward velocity.
2. **Match action sequences across backends.** Use the same numpy RNG seed to draw actions, then feed identical action arrays to both Gym and MJX. Otherwise we're comparing distributions of different action distributions, not env-physics distributions.
3. **Multi-step rollouts (5-10 steps), not 1-step.** Ant-at-action=0 falls fast; reward components depend on contact accumulation over a few steps. Single-step is dominated by init-state noise.
4. **Add explicit termination-rate parity check.** If MJX terminates 80% of episodes by step 10 but Gym terminates 30%, survive_reward distributions silently diverge by a factor — survive-mean mean check could pass while behavior is wrong.

- [ ] **Step 1:** Write the parity test:

  ```python
  """tests/test_ant_parity.py — distributional parity vs Gym Ant-v5.

  Strategy: matched action sequences, multi-step rollouts, four assertions:
  reset z-distribution, reward-component means/stds, termination-fraction.
  GPU/CPU divergence + RK4-vs-MJX integrator differences mean we cannot
  bit-match; we check distributions over n=200 trajectories.
  """

  import gymnasium as gym
  import jax
  import jax.numpy as jp
  import numpy as np
  import pytest

  from jax_rl.envs.locomotion.ant import Ant


  pytestmark = [pytest.mark.gpu, pytest.mark.warp]


  N_TRAJ = 200
  N_STEPS = 5
  ACTION_SCALE = 0.3


  def _make_action_sequence(n_traj, n_steps, seed):
      """Same actions for both backends — keeps parity comparison sound."""
      rng = np.random.default_rng(seed)
      return rng.uniform(-ACTION_SCALE, ACTION_SCALE, size=(n_traj, n_steps, 8)).astype(np.float32)


  def test_reset_obs_dim_matches_gym():
      gym_env = gym.make("Ant-v5")
      gym_obs, _ = gym_env.reset(seed=0)
      mjx_env = Ant()
      mjx_state = mjx_env.reset(jax.random.PRNGKey(0))
      assert gym_obs.shape == mjx_state.obs.shape, \
          f"Gym {gym_obs.shape} vs MJX {mjx_state.obs.shape}"


  def test_reset_qpos_z_distribution_matches_gym():
      """qpos[2] (torso z) on reset should be init_qpos[2] (=0.75) + U[-0.1, 0.1]."""
      gym_env = gym.make("Ant-v5")
      gym_zs = []
      for s in range(N_TRAJ):
          gym_env.reset(seed=s)
          gym_zs.append(gym_env.unwrapped.data.qpos[2])
      gym_zs = np.array(gym_zs)

      mjx_env = Ant()
      keys = jax.random.split(jax.random.PRNGKey(0), N_TRAJ)
      mjx_states = jax.vmap(mjx_env.reset)(keys)
      mjx_zs = np.array(mjx_states.data.qpos[:, 2])

      # Both are init_qpos[2] + U(-0.1, 0.1) → mean ~0.75, std ~0.058
      assert abs(gym_zs.mean() - mjx_zs.mean()) < 0.02, \
          f"Mean mismatch: gym {gym_zs.mean():.4f} vs mjx {mjx_zs.mean():.4f}"
      assert abs(gym_zs.std() - mjx_zs.std()) < 0.02, \
          f"Std mismatch: gym {gym_zs.std():.4f} vs mjx {mjx_zs.std():.4f}"


  def _rollout_gym(actions: np.ndarray):
      """Returns dict of per-step reward components + per-traj termination flag."""
      env = gym.make("Ant-v5")
      n_traj, n_steps, _ = actions.shape
      out = {k: [] for k in ("forward", "ctrl", "contact", "survive", "total")}
      term = []
      for t in range(n_traj):
          env.reset(seed=t)
          terminated_at = -1
          for s in range(n_steps):
              if terminated_at >= 0:
                  break
              _, r, term_flag, _, info = env.step(actions[t, s])
              out["forward"].append(info["reward_forward"])
              out["ctrl"].append(info["reward_ctrl"])
              out["contact"].append(info["reward_contact"])
              out["survive"].append(info["reward_survive"])
              out["total"].append(r)
              if term_flag:
                  terminated_at = s
          term.append(terminated_at >= 0)
      return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, np.asarray(term)


  def _rollout_mjx(actions: np.ndarray):
      env = Ant()
      n_traj, n_steps, _ = actions.shape
      out = {k: [] for k in ("forward", "ctrl", "contact", "survive", "total")}
      term = []
      keys = jax.random.split(jax.random.PRNGKey(0), n_traj)
      for t in range(n_traj):
          state = env.reset(keys[t])
          terminated_at = -1
          for s in range(n_steps):
              if terminated_at >= 0:
                  break
              state = env.step(state, jp.asarray(actions[t, s]))
              out["forward"].append(float(state.metrics["reward_forward"]))
              out["ctrl"].append(float(state.metrics["reward_ctrl"]))
              out["contact"].append(float(state.metrics["reward_contact"]))
              out["survive"].append(float(state.metrics["reward_survive"]))
              out["total"].append(float(state.reward))
              if float(state.done) > 0.5:
                  terminated_at = s
          term.append(terminated_at >= 0)
      return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, np.asarray(term)


  def test_reward_component_distributions_match():
      actions = _make_action_sequence(N_TRAJ, N_STEPS, seed=42)
      gym_rew, _ = _rollout_gym(actions)
      mjx_rew, _ = _rollout_mjx(actions)

      # Use absolute tolerance for components whose mean is near zero, relative
      # for components with non-zero scale. ctrl_cost is dominated by ‖action‖²
      # which is identical across backends (we share the action sequence), so
      # reward_ctrl mean is essentially deterministic and a tight check is OK.
      tolerances = {
          "forward":  ("rel", 0.30),  # forward vel is small + noisy; loosen
          "ctrl":     ("rel", 0.05),  # action shared → near-bitID
          "contact":  ("rel", 0.50),  # contact forces are integrator-sensitive
          "survive":  ("abs", 0.10),  # values 0/1, small abs tol
          "total":    ("rel", 0.30),
      }
      for k, (mode, tol) in tolerances.items():
          g, m = gym_rew[k], mjx_rew[k]
          if mode == "rel":
              denom = max(abs(g.mean()), 1e-3)
              rel_err = abs(g.mean() - m.mean()) / denom
              assert rel_err < tol, \
                  f"{k} mean rel-err {rel_err:.3f} > {tol}: gym={g.mean():.4f} mjx={m.mean():.4f}"
          else:
              abs_err = abs(g.mean() - m.mean())
              assert abs_err < tol, \
                  f"{k} mean abs-err {abs_err:.4f} > {tol}: gym={g.mean():.4f} mjx={m.mean():.4f}"


  def test_termination_fraction_matches_gym():
      """Within N_STEPS, fraction of trajectories that terminated should match."""
      actions = _make_action_sequence(N_TRAJ, N_STEPS, seed=42)
      _, gym_term = _rollout_gym(actions)
      _, mjx_term = _rollout_mjx(actions)
      gym_frac = gym_term.mean()
      mjx_frac = mjx_term.mean()
      # Tolerance: 10 percentage points. Different integrators (RK4 vs MJX)
      # produce different per-step contact responses; a few extra terminations
      # is normal; an order-of-magnitude difference is a bug.
      assert abs(gym_frac - mjx_frac) < 0.10, \
          f"Termination fraction mismatch: gym={gym_frac:.3f} mjx={mjx_frac:.3f}"
  ```

- [ ] **Step 2:** Run on GPU lane: `uv run python -m pytest tests/test_ant_parity.py -v -m "gpu and warp"`. Record any failures verbatim.
- [ ] **Step 3:** **Diagnostic decision tree if any test fails:**
  - `test_reset_obs_dim_matches_gym` fails → cfrc_ext shape wrong or `nbody` mismatch; revisit Stage 0 step 2.
  - `test_reset_qpos_z_distribution_matches_gym` fails on mean → init_qpos source bug (likely reading the Brax `<custom>` numeric or zero); confirm `_post_init` reads `mj_model.qpos0`.
  - `test_reset_qpos_z_distribution_matches_gym` fails on std → noise scale wrong; check `reset_noise_scale=0.1` and uniform-vs-normal swap.
  - `reward_ctrl` rel-err > 5% → action plumbing broken (ant might be receiving scaled or clamped actions when shared sequence should be deterministic). Check `step()`'s control assignment.
  - `reward_forward` rel-err > 30% → CoM xpos vs qpos bug. Re-grep `ant_v5.py:350-356`.
  - `reward_contact` rel-err > 50% → MJX feet ghosting through floor. Re-check Brax XML's foot spheres `contype="1"` survived the strip step. Possibly try `integrator="Euler"` per Stage 1 fallback.
  - `test_termination_fraction_matches_gym` fails → integrator divergence over multi-step; consider RK4 → Euler swap. Or healthy-check threshold drift.

---

## Stage 4: Wire into pipeline

### Task 4.1: Register `AntMJX` into both backend dispatch + detect_backend

**Files:**
- Modify: `jax_rl/training/env_backends/mjx_backend.py`
- Modify: `jax_rl/training/env_backends/__init__.py`
- Modify: `scripts/record_video.py`

**Three call sites resolve env_name → env instance — ALL must be patched.** Find current line numbers with:
```bash
grep -n "pg_registry.load" jax_rl/training/env_backends/mjx_backend.py scripts/record_video.py
```
Audit r1 saw lines 190 and 215 in `mjx_backend.py` and 334 in `record_video.py` — verify before editing.

- [ ] **Step 1 (verification only — no code change):** Confirm Stage 0 step 5 passed: `detect_backend("AntMJX")` already returns `"mjx"` via the default fallthrough in `jax_rl/training/env_backends/__init__.py`. **No registration code needed.** If Stage 0 step 5 is missing, run it now and confirm before continuing.

- [ ] **Step 2:** Add the dispatch helper (public, importable by all 3 sites — leading underscore would imply private but record_video.py imports it cross-module):
  ```python
  # In jax_rl/training/env_backends/mjx_backend.py (module level, near the top):

  def maybe_load_custom_env(env_name: str):
      """Return a locally-defined MJX env instance if env_name is one of
      our hand-rolled envs (NOT registered in mujoco_playground's pg_registry);
      else return None so the caller can fall through to pg_registry.load.
      """
      if env_name == "AntMJX":
          from jax_rl.envs.locomotion.ant import Ant
          return Ant()
      return None
  ```

- [ ] **Step 3:** Replace BOTH `pg_registry.load(cfg.env_name)` call sites in `mjx_backend.py` (the training env + the eval env, around lines 190 and 215 per r1 audit) with an **explicit None-check** (NOT `or`-truthiness — env structs may evaluate falsy under JAX dataclasses):
  ```python
  env = maybe_load_custom_env(cfg.env_name)
  if env is None:
      env = pg_registry.load(cfg.env_name)
  ```

- [ ] **Step 4:** Replace the `pg_registry.load(env_name)` call at `scripts/record_video.py` (around line 334) with the same explicit None-check pattern. Import the helper at the top of `record_video.py`:
  ```python
  from jax_rl.training.env_backends.mjx_backend import maybe_load_custom_env

  # Then at the call site (was: env = pg_registry.load(env_name)):
  env = maybe_load_custom_env(env_name)
  if env is None:
      env = pg_registry.load(env_name)
  ```
  *Without this, `record_video.py --env AntMJX` will KeyError on `pg_registry.load("AntMJX")` and Stage 5.3 figure rendering breaks.*

- [ ] **Step 5:** Smoke the dispatch helper:
  ```bash
  XLA_CLIENT_MEM_FRACTION=0.55 uv run python -c "
  from jax_rl.training.env_backends.mjx_backend import maybe_load_custom_env
  env = maybe_load_custom_env('AntMJX')
  print('env:', type(env).__name__, 'action_size:', env.action_size)
  "
  ```
  Expected: `env: Ant action_size: 8`.

- [ ] **Step 6:** Smoke `detect_backend`:
  ```bash
  uv run python -c "
  from jax_rl.training.env_backends import detect_backend
  print('AntMJX:', detect_backend('AntMJX'))   # expect 'mjx'
  print('Ant:',    detect_backend('Ant'))      # expect 'gym' (unchanged)
  print('CheetahRun:', detect_backend('CheetahRun'))   # expect 'mjx'
  "
  ```
  Expected: AntMJX→mjx, Ant→gym, CheetahRun→mjx. **Critical**: if AntMJX→gym or unknown, Step 1 didn't take; revisit before continuing.

- [ ] **Step 7:** Smoke `make_env_bundle("AntMJX")` — isolates the bundle factory bug from any SAC issue:
  ```bash
  XLA_CLIENT_MEM_FRACTION=0.55 uv run python -c "
  from jax_rl.training import make_env_bundle
  from dataclasses import dataclass
  # Mock minimal cfg matching make_envs expectations:
  @dataclass
  class Cfg:
      env_name: str = 'AntMJX'
      num_envs: int = 4
      episode_length: int = 1000
      action_repeat: int = 1
      reset_mode: str = 'legacy'
      action_delay_range_ms: object = None
  bundle = make_env_bundle(Cfg(), seed=0)
  print('obs_dim:', bundle.obs_dim, 'action_dim:', bundle.action_dim)
  assert bundle.obs_dim == 105, f'Expected 105, got {bundle.obs_dim}'
  assert bundle.action_dim == 8, f'Expected 8, got {bundle.action_dim}'
  "
  ```
  Expected: `obs_dim: 105 action_dim: 8`. **If this fails, fix here BEFORE running SAC** (SAC failure mode is "all NaN" which hides obs-dim/wrapper bugs).

- [ ] **Step 8:** Bigger smoke: SAC 5K-step training on AntMJX (no DIAYN yet) — verifies the existing SAC pipeline runs end-to-end:
  ```bash
  XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_sac.py \
      --env AntMJX --total-timesteps 5000 --seed 0 2>&1 | tail -20
  ```
  Expected: finite losses, finite returns, no NaN. If `reward_forward` is consistently negative early, that's expected (random policy). If everything is `nan`, check `_is_unhealthy` gating + reset noise.

---

## Stage 5: DIAYN acceptance + visual figure

### Task 5.1: DIAYN smoke (10K seed 0)

**Files:** none (training only).

- [ ] **Step 1:**
  ```bash
  XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_skill_discovery.py \
      --env AntMJX --num-skills 8 --total-timesteps 10000 --seed 0 \
      2>&1 | tee .temp/logs/ant_diayn_smoke.log
  ```
- [ ] **Step 2:** Acceptance: finite aux losses, no NaN, buffer fills past `min_buffer=8192`. Per-skill eval (z0..z7) returns finite. Same gates as SD-B Phase 5.1 on CheetahRun.
- [ ] **Step 3:** Journal entry: `.context/journals/2026-05-04-ant-port.md` with smoke result.

### Task 5.2: DIAYN 1M seed 0 acceptance

- [ ] **Step 1:**
  ```bash
  XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_skill_discovery.py \
      --env AntMJX --num-skills 8 --total-timesteps 1000000 --seed 0 \
      2>&1 | tee .temp/logs/ant_diayn_1m_seed0.log
  ```
- [ ] **Step 2:** Acceptance gate (with fallback for slower convergence on the richer 105d obs):
  - **Primary**: DiscA > `1/num_skills + 0.1 = 0.225` by 100K, per-skill spread > 50% of max-skill mean at 1M.
  - **Fallback**: DiscA > `1/num_skills + 0.05 = 0.175` by 100K AND > 0.225 by 200K. Ant's 105d obs is ~5× CheetahRun's 17d, so the discriminator may climb slower in absolute steps even though the learning dynamics are healthy. If primary fails but fallback passes, log it and continue — don't bisect.
  - Wall-clock projection: similar to CheetahRun (~28 min/seed). Run in background.
- [ ] **Step 3:** If skill collapse on Ant looks similar to CheetahRun (1-3 active of 8 by task return), that confirms DIAYN-on-quadruped ceiling — SAME canonical pattern. Document; this is expected per `lessons/skill_discovery_diayn_cheetah.md`. **Crucially**: even if return-spread collapses, the xy-trajectory figure (Stage 5.3) may still show legible diversity since DIAYN can produce skills that go in different *directions* with similar magnitudes — the plot reveals diversity that scalar return doesn't.

### Task 5.3: xy-trajectory diversity figure

**Files:** `scripts/plot_skill_xy.py`

**Critical methodology:**
- **Fixed-seed init across skills.** Each skill rolls out from the SAME initial state (deterministic reset to `init_qpos` with zero noise), otherwise reset noise dominates the early trajectory and skill-specific signal is buried.
- **Multiple rollouts per skill (≥3).** DIAYN paper App. D.3 uses ~5 rollouts/skill; single-rollout figures are seed-noise-dominated. Plot all rollouts thinly + mean trajectory thickly per skill.
- **Reuse `record_video.py` helpers, do NOT re-implement.** `record_video.py` exposes `_resolve_skill_vector` (line ~66), `_SkillWrappedAlgo` (~118), `_build_select_action` (~148), and `load_actor_for_inference` import (~45). Stage 5.3 budget assumes these are imported; if any helper is buried inside `record()` body and not at module scope, **a Step 0 refactor is required first** to expose them.

- [ ] **Step 0 (verification only — expected no-op):** Inspect `scripts/record_video.py` and confirm the helpers above are at module scope. As of HEAD (commit `47d41cc`+), `_resolve_skill_vector` (line 66), `_SkillWrappedAlgo` (line 118), and `_build_select_action` (line 148) are all module-level — no refactor needed. Run `grep -nE "^(def |class )(_resolve_skill_vector|_SkillWrappedAlgo|_build_select_action)" scripts/record_video.py`; expect 3 matches. If any are missing or nested (only possible if record_video.py was refactored after this plan was written), refactor those specific helpers to module scope as a separate prep PR before starting Step 1.

- [ ] **Step 1:** Write the figure script. **Locked record_video.py helper signatures (verified 2026-05-04):**
  - `_resolve_skill_vector(meta, skill_index, skill_vector_path) -> jnp.ndarray | None` — needs `meta` dict (NOT `num_skills` / `mode`); requires `meta["skill_discovery"]["total_skill_dim"]`.
  - `_build_select_action(meta, obs_dim, action_dim) -> (algo_obj, kind: "ppo"|"offpolicy")` — `obs_dim` here is the **augmented** dim (raw_obs + skill_z), NOT raw. Returns the algo *object*, NOT a select_action callable.
  - `load_actor_for_inference(ckpt_dir) -> (meta, actor_params, norm_state, actor_batch_stats)` — 4-tuple, NOT 3.
  - Real action call: `algo.select_action(actor_params, obs, key, deterministic=True)`.
  - `_SkillWrappedAlgo(inner=algo, skill_z=z)` is the canonical way to inject skill_z; its `select_action` does `concat([obs, z])` and forwards.

  ```python
  """scripts/plot_skill_xy.py — DIAYN-paper-canonical skill diversity figure.

  Loads a skill-discovery checkpoint, rolls out N steps per skill from a
  FIXED initial state (no reset noise), captures the torso CoM xy per
  step, and plots one figure with all skills overlaid (color = skill idx).
  """

  import argparse
  from pathlib import Path

  import jax
  import jax.numpy as jp
  import matplotlib.pyplot as plt
  import numpy as np
  import mujoco
  from mujoco import mjx
  from mujoco_playground._src import mjx_env

  from jax_rl.envs.locomotion.ant import Ant
  from jax_rl.training.checkpointing import load_actor_for_inference
  from scripts.record_video import (
      _resolve_skill_vector,
      _SkillWrappedAlgo,
      _build_select_action,
  )


  def deterministic_reset(env: Ant) -> mjx_env.State:
      """Reset env to init_qpos/init_qvel WITHOUT random noise.

      Different from env.reset() which adds U(-0.1, 0.1) qpos noise +
      N(0, 0.1) qvel noise. We need shared init across skills/rollouts so
      the figure shows skill-driven variance, not reset noise.
      """
      data = mjx_env.make_data(
          env.mj_model,
          qpos=env._init_qpos,
          qvel=env._init_qvel,
          impl=env.mjx_model.impl.value,
          naconmax=env._config.naconmax,
          njmax=env._config.njmax,
      )
      data = mjx.forward(env.mjx_model, data)
      info = {"rng": jax.random.PRNGKey(0)}
      metrics = {k: jp.zeros(()) for k in
                 ("reward_forward", "reward_ctrl", "reward_contact", "reward_survive")}
      reward, done = jp.zeros(2)
      obs = env._get_obs(data)
      return mjx_env.State(data, obs, reward, done, metrics, info)


  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--checkpoint", type=Path, required=True)
      parser.add_argument("--rollouts-per-skill", type=int, default=3)
      parser.add_argument("--rollout-length", type=int, default=500)
      parser.add_argument("--output", type=Path, required=True)
      args = parser.parse_args()

      env = Ant()

      # Load checkpoint: returns 4-tuple. We don't use batch_stats for SAC.
      meta, actor_params, norm_state, _ = load_actor_for_inference(args.checkpoint)
      total_skill_dim = int(meta["skill_discovery"]["total_skill_dim"])

      raw_obs_dim = int(env.observation_size)   # 105 for Ant default
      augmented_obs_dim = raw_obs_dim + total_skill_dim
      action_dim = env.action_size

      # _build_select_action returns (algo, kind). We pass AUGMENTED obs_dim
      # because the saved actor was trained on (obs + skill_z).
      algo, kind = _build_select_action(meta, augmented_obs_dim, action_dim)
      assert kind == "offpolicy", \
          f"Expected SAC-family checkpoint, got kind={kind} for ant skill ckpt"

      cmap = plt.get_cmap("tab10")
      fig, ax = plt.subplots(figsize=(8, 8))

      # Per-skill aggregation for the numerical visual gate (Step 3):
      per_skill_endpoints: dict[int, list] = {}

      for skill_idx in range(total_skill_dim):
          # Build z via the canonical helper: pass meta + index, no skill_vector.
          z = _resolve_skill_vector(meta, skill_idx, None)
          assert z is not None, "ckpt missing skill_discovery block"

          # Wrap algo with the skill_z so .select_action concats internally.
          # _SkillWrappedAlgo expects un-augmented obs at call time.
          wrapped = _SkillWrappedAlgo(inner=algo, skill_z=z)

          per_skill_endpoints[skill_idx] = []

          for rollout_idx in range(args.rollouts_per_skill):
              key = jax.random.PRNGKey(rollout_idx + skill_idx * 1_000)
              state = deterministic_reset(env)
              xys = [np.array(state.data.xpos[env._config.main_body_id, :2])]

              for _ in range(args.rollout_length):
                  # Wrapped.select_action takes raw obs (it concats z internally).
                  obs_b = state.obs[None, :]    # add batch dim
                  key, sub = jax.random.split(key)
                  action = wrapped.select_action(
                      actor_params, obs_b, sub, deterministic=True
                  )
                  action = action[0]            # drop batch dim
                  state = env.step(state, action)
                  xys.append(np.array(state.data.xpos[env._config.main_body_id, :2]))
                  if float(state.done) > 0.5:
                      break

              xys = np.stack(xys)
              per_skill_endpoints[skill_idx].append(xys[-1])
              ax.plot(xys[:, 0], xys[:, 1], color=cmap(skill_idx),
                      alpha=0.5, linewidth=1.0,
                      label=f"z{skill_idx}" if rollout_idx == 0 else None)

      ax.set_xlabel("x (m)")
      ax.set_ylabel("y (m)")
      ax.set_aspect("equal")
      ax.set_title(f"Ant DIAYN skill diversity ({total_skill_dim} skills × "
                   f"{args.rollouts_per_skill} rollouts)")
      ax.legend(loc="best", fontsize=8)
      fig.tight_layout()
      fig.savefig(args.output, dpi=150)
      print(f"saved figure to {args.output}")

      # Numerical visual gate (Step 3):
      _evaluate_gate(per_skill_endpoints)


  def _evaluate_gate(per_skill_endpoints: dict) -> None:
      """Compute and print the numerical visual gate metrics."""
      # Mean endpoint per skill (average of rollout endpoints).
      skills = sorted(per_skill_endpoints.keys())
      mean_endpoints = np.stack([
          np.mean(per_skill_endpoints[k], axis=0) for k in skills
      ])    # (num_skills, 2)

      # Max pairwise endpoint distance:
      diffs = mean_endpoints[:, None, :] - mean_endpoints[None, :, :]
      dists = np.linalg.norm(diffs, axis=-1)
      max_pairwise = float(dists.max())

      # Circular spread of endpoint headings (angle from origin).
      headings = np.arctan2(mean_endpoints[:, 1], mean_endpoints[:, 0])
      # Resultant length R for circular variance: smaller R = wider spread.
      R = np.sqrt(np.mean(np.cos(headings))**2 + np.mean(np.sin(headings))**2)
      # Circular std (radians): sqrt(-2 ln R), guarded.
      circ_std_rad = float(np.sqrt(-2.0 * np.log(max(R, 1e-9))))
      heading_std_deg = float(np.degrees(circ_std_rad))

      # Pairwise heading separations (radians, taking the wrap-around shortest).
      hpd = np.abs(((headings[:, None] - headings[None, :] + np.pi)
                    % (2 * np.pi)) - np.pi)
      max_pairwise_heading_deg = float(np.degrees(hpd.max()))

      print(f"max pairwise xy-endpoint distance: {max_pairwise:.2f} m")
      print(f"circular std of skill endpoint headings: {heading_std_deg:.1f}°")
      print(f"max pairwise heading separation: {max_pairwise_heading_deg:.1f}°")

      gate_pass = (max_pairwise > 3.0) or (heading_std_deg > 30.0)
      print(f"VISUAL GATE: {'PASS' if gate_pass else 'FAIL'}")


  if __name__ == "__main__":
      main()
  ```
  *Two private-attribute reads (`env._init_qpos`, `env._init_qvel`, `env._config.*`, `env._get_obs`) are intentional — `Ant` is a sibling module, not external API. Add public properties on `Ant` if this script grows beyond a one-shot figure.*

- [ ] **Step 2:** Generate figure for seed 0:
  ```bash
  uv run python scripts/plot_skill_xy.py \
      --checkpoint checkpoints/<latest_antmjx_seed0> \
      --num-skills 8 \
      --rollouts-per-skill 3 \
      --rollout-length 500 \
      --output .context/figures/ant_diayn_seed0.png
  ```

- [ ] **Step 3:** **Numerical visual gate** (already implemented as `_evaluate_gate(...)` in Step 1's script). The gate prints three metrics:
  - `max_pairwise` — Euclidean distance between the most-distant pair of skill mean-endpoints (m).
  - `heading_std_deg` — **circular** std of skill endpoint headings (degrees), computed from the resultant length R: `sqrt(-2 ln R)`. Linear `np.std(arctan2(...))` is wrong because it treats `+179°` and `-179°` as 358° apart instead of 2°.
  - `max_pairwise_heading_deg` — max wrap-around-shortest pairwise heading separation (degrees) between any two skills. Bounded `[0, 180]`.

  **Gate**: at least ONE of:
  - `max_pairwise > 3.0 m` (skills physically reach distinct xy locations after `rollout_length` steps)
  - `heading_std_deg > 30°` (skills go in distinct directions, in the *circular-std* sense)

  Note: `np.arctan2(y, x)` measures the angle of the **endpoint vector from origin**, not the instantaneous heading of motion at the endpoint. That's intentional — DIAYN App. D.3-style figures plot trajectory *endpoints* and ask whether they fan out, not whether the ant's heading is varied at any specific moment.

  If both fail → DIAYN collapsed even on Ant (consistent with the canonical limit). Log the values, treat as expected outcome (NOT a failure of the port), and journal alongside `lessons/skill_discovery_diayn_cheetah.md` for cross-env contrast.

### Task 5.4: 3-seed acceptance (REQUIRED — not optional)

**Rationale:** Cheetah's Wave D needed 3 seeds to call Gate 1 (seed-2 had max-skill=48.2 vs seed-0 max=7.3). Single-seed is high-variance; the headline figure is much weaker without multi-seed reproduction. The 1.5h serial cost is already in the wall-clock estimate.

- [ ] **Step 1:** Same pattern as SD-B Phase 5.3 — serial 3 seeds (XLA_CLIENT_MEM_FRACTION=0.55 doesn't fit 3 parallel on 16GB GPU; ~1.5h serial total). Save task ID + log paths to memory before launching.
  ```bash
  for s in 0 1 2; do
    XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_skill_discovery.py \
        --env AntMJX --num-skills 8 --total-timesteps 1000000 --seed $s \
        2>&1 | tee .temp/logs/ant_diayn_1m_seed${s}.log
  done
  ```
- [ ] **Step 2:** Generate 3-panel figure (one xy plot per seed) using `plot_skill_xy.py` × 3, then `matplotlib.subplot`-stitch into a single PNG. Commit alongside.
- [ ] **Step 3:** Apply the numerical visual gate from Task 5.3 Step 3 to each seed. At least 2 of 3 seeds should pass to call SD-B Gate 2 closed.

---

## Stage 6: Docs sync + commit

### Task 6.1: Lesson + journal + TODO

- [ ] **Step 1:** Append to `.context/lessons/skill_discovery_diayn_cheetah.md` (or create new `skill_discovery_diayn_ant.md` if findings diverge significantly). Compare Ant skill diversity vs Cheetah's collapse pattern. Include the numerical visual-gate values (max_pairwise_dist, heading_std).
- [ ] **Step 2:** New journal: `.context/journals/2026-05-04-ant-port.md` with smoke + 1M results + figure path + numerical visual-gate readout per seed.
- [ ] **Step 3:** Update `.context/TODO.md` SD-B Wave D Phase 5.3 Gate 2 → checked, with Ant figure path.
- [ ] **Step 4:** Update `.context/AGENT_HANDOFF.md` if `--env AntMJX` is now a routine entry (e.g. visible in Quick Reference). Skip if it's a one-shot figure that won't be re-run regularly.

### Task 6.2: Commit

- [ ] **Step 1:** Stage code files (run `git status` first to confirm what was actually modified — `env_backends/__init__.py` is a no-edit verification per Stage 4.1 Step 1, so drop it from this list if `git status` shows it clean):
  ```bash
  git add jax_rl/envs/locomotion/ant.py \
          jax_rl/envs/locomotion/xmls/ant.xml \
          jax_rl/training/env_backends/mjx_backend.py \
          scripts/record_video.py \
          scripts/plot_skill_xy.py \
          tests/test_ant_xml.py tests/test_ant_env.py tests/test_ant_parity.py
  ```
  Stage docs separately.
- [ ] **Step 2:** Commit with subject `feat(env): port Gym Ant-v5 to MJX/Warp as AntMJX`. Body covers: behavioral contract source, scalars locked, three-site dispatch hook, env-name choice (`AntMJX` to avoid collision with existing gym `Ant`), xy-trajectory figure as SD-B Gate 2.
- [ ] **Step 3:** Second commit with docs sync.

---

## Acceptance gates summary

| Gate | Where | Pass criterion |
|---|---|---|
| Pre-flight nbody | Task 0.1 step 2 | gym Ant-v5 reports nbody=14, obs=(105,) |
| cfrc shape | Task 0.1 step 3 | `mjx.make_data(...).cfrc_ext.shape == (14, 6)` |
| AntMJX unregistered | Task 0.1 step 5 | `detect_backend("AntMJX")` not in {gym, ...} before our changes |
| XML loads | Task 1.2 | nq=15, nv=14, nu=8, mjx.put_model(impl="warp") works |
| Env init | Task 2.1 | action_size=8, default config matches Gym v5 scalars |
| Reset finite | Task 2.2 | obs finite, qpos[2] in [0.649, 0.851] across 100 seeds |
| Step finite | Task 2.3 | step output finite; all 4 reward components in metrics; info has x_velocity, y_velocity, distance_from_origin |
| Termination | Task 2.3 | done=1 when z<0.2 |
| Obs dim | Task 2.4 | default 105d; toggles produce 27/107 as expected |
| Parity (reset z) | Task 3.1 | gym vs MJX z-mean within 0.02, std within 0.02 |
| Parity (rewards) | Task 3.1 | per-component tolerances per Stage 3 table |
| Parity (termination rate) | Task 3.1 | gym vs MJX termination fraction within 10 pp |
| detect_backend | Task 4.1 step 6 | `detect_backend("AntMJX") == "mjx"`; unchanged for `Ant` and `CheetahRun` |
| Bundle factory | Task 4.1 step 7 | `make_env_bundle("AntMJX")` produces bundle with obs_dim=105, action_dim=8 |
| SAC pipeline | Task 4.1 step 8 | 5K SAC train run finishes, no NaN |
| DIAYN smoke | Task 5.1 | 10K DIAYN, no NaN, finite eval |
| DIAYN acceptance | Task 5.2 | Primary: DiscA > 0.225 by 100K + spread > 50% of max at 1M. Fallback: DiscA > 0.175 by 100K, > 0.225 by 200K |
| Visual gate (numerical) | Task 5.3 step 3 | max_pairwise_endpoint > 3.0 m OR heading_std > 30° |
| Multi-seed visual | Task 5.4 step 3 | ≥ 2 of 3 seeds pass numerical visual gate |

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Env-name collision: `Ant` already gym-registered** | 🚨 critical | Use `AntMJX` (NOT `Ant`) as env_name string; verify `detect_backend("AntMJX") == "mjx"` in Task 4.1 step 6 BEFORE proceeding. |
| **Three dispatch sites, not two** | 🚨 critical | `record_video.py:~334` also calls `pg_registry.load`. All three patched in Task 4.1 step 4. |
| **Parity test design** (action=0 + 1-step) | 🚨 critical | Stage 3 redesigned: random matched actions, multi-step rollouts, termination-rate check, per-component tolerances. |
| **`init_qpos` source confusion** (Brax `<custom>` says z=0.55 but stock MuJoCo ignores it) | 🟠 high | Strip Brax `<custom name="init_qpos">` (Task 1.1 step 2); read `mj_model.qpos0` (= 0.75 from torso body pos); add explicit comment in `_post_init`. |
| **DIAYN paper Fig. 4 / App. D.3 citation** | 🟠 high | Eyeball-verify against the paper PDF before figure caption ships. Same trap as the prior D.4→D.3 incident. |
| **Warp contact buffer overflow** | 🟠 high | Add `naconmax=100_000`, `njmax=100` to default_config and forward to `make_data` (mirrors cheetah.py). |
| **MJX/Warp + ant.xml: feet ghost through floor** | 🟠 high | Use Brax XML variant with foot spheres `contype="1"` (Task 1.1 step 1). Caught by Stage 3 contact_cost parity check. |
| **Integrator deviation (Gym RK4 vs MJX/Warp)** | 🟡 medium | Try RK4 first (Task 1.1 step 3); fall back to Euler ONLY if RK4 fails Stage 1 step 5 or contact_cost rel-err > 100% in parity. Journal the deviation. |
| **`cfrc_ext` shape under MJX/Warp** | 🟡 medium | Verify in Stage 0 step 3 BEFORE locking obs=105d. Should be `(nbody, 6)`. |
| **`terminate_when_unhealthy` config flag dead** | 🟡 medium | Wire flag into `done` computation (Task 2.3 step 1, branch on flag). |
| **pytest markers missing → CPU-lane crash on `mjx.put_model`** | 🟡 medium | Add module-level `pytestmark = [pytest.mark.gpu, pytest.mark.warp]` in all 3 new test files. |
| **DIAYN skill collapse on Ant** | 🟡 medium | Expected per `lessons/skill_discovery_diayn_cheetah.md`; xy-trajectory figure may still show legible diversity even when scalar return spread collapses. Numerical visual gate (Task 5.3 step 3) handles this case. |
| **Skill_offpolicy_loop expects env attributes** | 🟢 low | Validated by Task 4.1 step 7 (`make_env_bundle` smoke). |
| **3 parallel seeds OOM at fraction=0.55** | 🟢 low | Serialize per SD-B 5.3 lesson (~1.5h). |
| **`record_video.py` helper functions nested** | 🟢 low | Stage 5.3 step 0 refactor exposes them at module scope. Separate prep PR. |
| **Stage 1 ↔ Stage 3 round-trip on contact issues** | 🟢 low | Budget 2-3 iterations on parity_test failure → XML fix → re-run. Total wall-clock 7-9h (revised from 6-8h). |

---

## Out of scope

- Multi-agent / multi-ant.
- Domain randomization on Ant — DR is not part of Gym Ant-v5; would diverge from contract.
- METRA / D3 on Ant — that's SD-D / SD-E.
- Ant on the gym CPU backend via `gym_backend.py:200` — that registration stays as-is; this plan is the MJX/Warp companion path under a different env_name.
- Any non-Ant env in this plan (Hopper, Walker2d, etc.). Each is its own port, even though the pattern is the same.

---

## Success = all of:

1. `tests/test_ant_*.py` green (xml + env + parity, all under `[gpu, warp]` markers).
2. `detect_backend("AntMJX") == "mjx"` and `make_env_bundle("AntMJX")` produces obs_dim=105, action_dim=8.
3. `scripts/train_skill_discovery.py --env AntMJX --total-timesteps 1000000 --seed 0` runs to completion. DiscA hits primary gate (> 0.225 by 100K + spread > 50% at 1M) OR fallback gate (> 0.175 by 100K, > 0.225 by 200K).
4. 3 seeds × 1M completed serial; xy-trajectory figure exists at `.context/figures/ant_diayn_seed{0,1,2}.png` with at least 2 of 3 seeds passing the numerical visual gate (max_pairwise_dist > 3.0 m OR heading_std > 30°).
5. `lessons/skill_discovery_diayn_cheetah.md` updated with Ant cross-env comparison (or new `_ant.md` lesson if findings diverge enough).
6. SD-B Wave D Phase 5.3 Gate 2 checkbox checked in `TODO.md` with figure path.

---

## Estimated wall-clock (revised)

- Stage 0: 15 min (added 2 verification steps)
- Stage 1 (XML): 30 min
- Stage 2 (env class): 1.5h
- Stage 3 (parity): 1.5h (was 1h; budget +30 min for diagnostic decision-tree iterations)
- Stage 4 (dispatch): 45 min (was 30 min; 8 steps now incl. detect_backend + bundle smoke)
- Stage 5.1 (smoke): 20 min
- Stage 5.2 (1M seed 0): 30 min wall-clock + 15 min review
- Stage 5.3 (figure, incl. helper-extract refactor + numerical gate): 2h (was 1.5h)
- Stage 5.4 (3 seeds, REQUIRED): 1.5h
- Stage 6 (docs + commit): 30 min

**Total: 7-9h human-attended (some training is background).** Single agent, two-and-a-half audit rounds before dispatch (r1: 11 BLOCKERs caught; r2: 4 more BLOCKERs caught — naconmax-not-forwarded, truthiness-OR pattern, plot script signatures, redundant detect_backend registration; r3: only fixes, no new audit).
