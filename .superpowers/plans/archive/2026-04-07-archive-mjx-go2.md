# Archive MJX Go2 Environment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the MJX Go2 environment from the active codebase, keeping Warp as the sole Go2 backend. Archive (not delete) MJX files, clean all references.

**Architecture:** File moves to `archive/` directories + find-and-replace across registry, presets, docs. No logic changes to any active env.

**Tech Stack:** git mv, pytest, mkdocs, grep

**Spec:** `docs/superpowers/specs/2026-04-06-archive-mjx-go2.md`

---

## File Structure

| Action | File | Notes |
|--------|------|-------|
| Archive | `jax_rl/envs/locomotion/go2_base.py` | → `jax_rl/envs/locomotion/archive/` |
| Archive | `jax_rl/envs/locomotion/go2_joystick.py` | → `jax_rl/envs/locomotion/archive/` |
| Archive | `jax_rl/envs/locomotion/go2_cpu.py` | → `jax_rl/envs/locomotion/archive/` |
| Archive | `record_video_cpu.py` | → `archive/` (repo root) |
| Archive | `tests/test_go2_env.py` | → `tests/archive/` |
| Modify | `jax_rl/training/env_setup.py` | Remove Go2JoystickFlat registration |
| Modify | `jax_rl/configs/env_presets.py` | Remove Go2JoystickFlat PPO preset |
| Modify | `record_video.py` | Remove Go2JoystickFlat from ENV_DEFAULTS |
| Modify | `jax_rl/envs/locomotion/go2_constants.py` | Add ARCHIVED comments |
| Modify | `README.md` | Remove MJX examples/bullets |
| Modify | `deploy/README.md` | Update to Warp env |
| Modify | `deploy/obs_builder.py` | Update docstring reference |
| Modify | `deploy/sim2sim_direct.py` | Update comment |
| Modify | `docs/api/envs.md` | Remove MJX autodoc |
| Modify | `docs/reference/cli-flags.md` | Update example env |
| Modify | `docs/reference/env-presets.md` | Re-run generator |
| Modify | `train_ppo.py`, `train_ppo_fast.py`, `train_offpolicy.py` | Update help text |
| Modify | `.context/AGENT_HANDOFF.md` | Update env listing |
| Modify | `.context/NEW_AGENT_PROMPT.md` | Single env description |

---

### Task 1: Archive MJX env files

**Files:**
- Move: `jax_rl/envs/locomotion/go2_base.py` → `jax_rl/envs/locomotion/archive/go2_base.py`
- Move: `jax_rl/envs/locomotion/go2_joystick.py` → `jax_rl/envs/locomotion/archive/go2_joystick.py`
- Move: `jax_rl/envs/locomotion/go2_cpu.py` → `jax_rl/envs/locomotion/archive/go2_cpu.py`
- Move: `record_video_cpu.py` → `archive/record_video_cpu.py`
- Move: `tests/test_go2_env.py` → `tests/archive/test_go2_env.py`

- [ ] **Step 1: Create archive directories**

```bash
mkdir -p jax_rl/envs/locomotion/archive
mkdir -p tests/archive
mkdir -p archive
```

- [ ] **Step 2: Move files**

```bash
git mv jax_rl/envs/locomotion/go2_base.py jax_rl/envs/locomotion/archive/go2_base.py
git mv jax_rl/envs/locomotion/go2_joystick.py jax_rl/envs/locomotion/archive/go2_joystick.py
git mv jax_rl/envs/locomotion/go2_cpu.py jax_rl/envs/locomotion/archive/go2_cpu.py
git mv record_video_cpu.py archive/record_video_cpu.py
git mv tests/test_go2_env.py tests/archive/test_go2_env.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: archive MJX Go2 env files (go2_base, go2_joystick, go2_cpu)"
```

---

### Task 2: Remove MJX from registry and presets

**Files:**
- Modify: `jax_rl/training/env_setup.py:18-24`
- Modify: `jax_rl/configs/env_presets.py:50-69`
- Modify: `record_video.py:45`

- [ ] **Step 1: Remove Go2JoystickFlat registration from env_setup.py**

Remove lines 18-24 (the import + registration block for `Go2JoystickFlat`):

```python
# DELETE these lines:
    from jax_rl.envs.locomotion.go2_joystick import Joystick, default_config
    if "Go2JoystickFlat" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2JoystickFlat",
            functools.partial(Joystick, task="flat_terrain"),
            default_config,
        )
```

- [ ] **Step 2: Remove Go2JoystickFlat PPO preset from env_presets.py**

Remove lines 50-69 (the `"Go2JoystickFlat"` entry and its comment):

```python
# DELETE these lines:
    # Go2 locomotion — matches Playground Go1 Joystick PPO recipe.
    "Go2JoystickFlat": TrainConfig(
        env_name="Go2JoystickFlat",
        ...
    ),
```

- [ ] **Step 3: Remove Go2JoystickFlat from record_video.py ENV_DEFAULTS**

Remove line 45:

```python
# DELETE this line:
    "Go2JoystickFlat":  ((480, 480), "track"),
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All pass. `test_go2_env.py` is already archived so it won't run.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/training/env_setup.py jax_rl/configs/env_presets.py record_video.py
git commit -m "refactor: remove Go2JoystickFlat from registry, presets, and record_video"
```

---

### Task 3: Update go2_constants.py

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_constants.py`

- [ ] **Step 1: Add ARCHIVED comments**

Add `# ARCHIVED` comments above the MJX-only constants:

```python
# ARCHIVED — MJX env used Menagerie scene XML (go2_base.py, go2_joystick.py).
# Kept for reference. Active env uses WARP_SCENE_FLAT_XML.
SCENE_FLAT_XML = ROOT_PATH / "go2_scene_flat.xml"
```

```python
# ARCHIVED — Menagerie root body name. Warp env uses WARP_ROOT_BODY.
ROOT_BODY = "base"
```

- [ ] **Step 2: Commit**

```bash
git add jax_rl/envs/locomotion/go2_constants.py
git commit -m "docs: mark MJX-only constants as archived in go2_constants.py"
```

---

### Task 4: Update training script help text

**Files:**
- Modify: `train_ppo.py:362`
- Modify: `train_ppo_fast.py:444`
- Modify: `train_offpolicy.py:4,382`

- [ ] **Step 1: Update all help text references**

In each file, change `Go2JoystickFlat` to `Go2WarpJoystickFlat` in `--env` help strings:

`train_ppo.py:362`:
```python
help="Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat)"
```

`train_ppo_fast.py:444`:
```python
help="Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat)"
```

`train_offpolicy.py:382`:
```python
help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)"
```

`train_offpolicy.py:4` (docstring):
```python
    uv run python train_offpolicy.py --algo sac --env Go2WarpJoystickFlat
```

- [ ] **Step 2: Commit**

```bash
git add train_ppo.py train_ppo_fast.py train_offpolicy.py
git commit -m "docs: update training script help text to reference Warp env"
```

---

### Task 5: Update deploy docs and comments

**Files:**
- Modify: `deploy/README.md:121,153,174`
- Modify: `deploy/obs_builder.py:35`
- Modify: `deploy/sim2sim_direct.py:124`

- [ ] **Step 1: Update deploy/README.md**

Line 121 — change training command:
```
uv run python train_ppo_fast.py --env Go2WarpJoystickFlat --num-envs 1024 \
```

Line 153 — update training env description:
Replace any mention of "Training env (MJX)" and "Menagerie go2_mjx.xml" with Warp env info (Kp=20, Kd=0.5, unitree MJCF). Read the surrounding context before editing to preserve the section structure.

Line 174 — update joint order note:
Replace "Our training env (MJX/Menagerie)" with "Our training env (Warp/unitree)". Note: unitree actuator order is FR,FL,RR,RL (different from Menagerie's FL,FR,RL,RR). Verify this matches the `act_to_joint` remapping in `go2_warp_base.py`.

- [ ] **Step 2: Update deploy/obs_builder.py docstring**

Line 35 — change:
```python
# Before:
    Obs layout (matching go2_joystick.py _get_obs):
# After:
    Obs layout (matching go2_warp_joystick.py _get_obs):
```

- [ ] **Step 3: Update deploy/sim2sim_direct.py comment**

Line 124 — change:
```python
# Before:
    # Match training env physics (go2_base.py overrides).
# After:
    # Match training env physics (go2_warp_base.py overrides).
```

- [ ] **Step 4: Commit**

```bash
git add deploy/README.md deploy/obs_builder.py deploy/sim2sim_direct.py
git commit -m "docs: update deploy docs to reference Warp env"
```

---

### Task 6: Update docs site

**Files:**
- Modify: `docs/api/envs.md:9`
- Modify: `docs/reference/cli-flags.md:44`
- Modify: `docs/reference/env-presets.md` (auto-generated)
- Modify: `docs/tutorials/sim2real.md:20-25`
- Modify: `docs/scripts/gen_cli_reference.py:67`

- [ ] **Step 1: Remove MJX autodoc from docs/api/envs.md**

Remove line 9:
```markdown
:::: jax_rl.envs.locomotion.go2_joystick.Joystick
```

- [ ] **Step 2: Update docs/reference/cli-flags.md**

Line 44 — change example:
```markdown
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
```

- [ ] **Step 3: Update docs/scripts/gen_cli_reference.py**

Line 67 — change example env:
```python
help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
```

- [ ] **Step 4: Re-run env presets generator**

Run: `uv run python docs/scripts/gen_env_presets.py`
Expected: `docs/reference/env-presets.md` regenerated without `Go2JoystickFlat` row.

- [ ] **Step 5: Update docs/tutorials/sim2real.md**

Lines 20-25 — read the full section, then rewrite the MJX vs Warp comparison to note MJX is archived. Keep the lesson (why MJX was insufficient) but clarify Warp is the only active path.

- [ ] **Step 6: Build docs to verify**

Run: `uv run mkdocs build`
Expected: No errors, no warnings about missing references.

- [ ] **Step 7: Commit**

```bash
git add docs/
git commit -m "docs: update site references — MJX Go2 archived, Warp is sole path"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md:57-58,76,100`

- [ ] **Step 1: Remove MJX training example**

Lines 57-58 — remove or replace with Warp example:
```markdown
# Before:
# PPO on Go2 joystick walking (MJX backend, Menagerie MJCF)
uv run python train_ppo_fast.py --env Go2JoystickFlat --total-timesteps 50000000

# After (replace with Warp FastSAC, our proven pipeline):
# FastSAC on Go2 joystick walking (Warp backend, unitree MJCF)
uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --total-timesteps 50000000
```

- [ ] **Step 2: Remove Go2JoystickFlat bullet**

Line 76 — delete the bullet:
```markdown
# DELETE:
- `Go2JoystickFlat` — MJX (JAX) backend, Menagerie go2_mjx.xml (simplified collision geometry). Fast, proven.
```

- [ ] **Step 3: Update CLI example**

Line 100 — change:
```markdown
--env NAME              # Environment name (e.g., CheetahRun, Go2WarpJoystickFlat)
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README — remove MJX Go2 references"
```

---

### Task 8: Update project context docs

**Files:**
- Modify: `.context/AGENT_HANDOFF.md`
- Modify: `.context/NEW_AGENT_PROMPT.md`

- [ ] **Step 1: Update AGENT_HANDOFF.md**

Read the file first. Then:
- Remove or update the `Go2JoystickFlat` env listing (around line 215)
- Change the "MJX env frozen" policy statement (around line 240) to "MJX env archived"
- Update any example training commands that use `Go2JoystickFlat` (around lines 249, 252)
- Update file listings that reference `go2_base.py`, `go2_joystick.py` (around line 167)

- [ ] **Step 2: Update NEW_AGENT_PROMPT.md**

Read the file first. Change the "Two Go2 envs" description (around line 35) to single env:
```markdown
# Before:
Two Go2 envs: Go2JoystickFlat (MJX, Menagerie MJCF, Kp=35/Kd=0.1) and Go2WarpJoystickFlat (Warp, unitree MJCF, Kp=20/Kd=0.5). Warp is preferred...

# After:
Go2WarpJoystickFlat (Warp backend, unitree MJCF, Kp=20/Kd=0.5). MJX Go2 env archived — see jax_rl/envs/locomotion/archive/.
```

- [ ] **Step 3: Commit**

```bash
git add .context/AGENT_HANDOFF.md .context/NEW_AGENT_PROMPT.md
git commit -m "docs: update project context — MJX Go2 archived"
```

---

### Task 9: Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All pass.

- [ ] **Step 2: Verify Warp env loads**

Run: `uv run python -c "from jax_rl.training.env_setup import _register_custom_envs; print('OK')"`
Expected: `OK` — no import errors.

- [ ] **Step 3: Grep for stale references**

Run: `grep -rn "Go2JoystickFlat\|from.*go2_base\|from.*go2_cpu\|from.*go2_joystick[^_w]" --include="*.py" --exclude-dir=archive --exclude-dir=tools .`
Expected: Zero hits (tools/ excluded since they're debug scripts we're leaving as-is).

Run: `grep -rn "Go2JoystickFlat" --include="*.md" --exclude-dir=archive . | grep -v "journals\|lessons\|LESSONS\|specs/2026-04-06\|plans/2026-04-07"`
Expected: Zero hits (journals/lessons are historical, spec/plan reference themselves).

- [ ] **Step 4: Build docs**

Run: `uv run mkdocs build`
Expected: Clean build, no warnings.
