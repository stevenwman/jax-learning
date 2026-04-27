# Algo Port / Refactor Protocol

> Contract for any agent porting a new RL algorithm into this repo or
> refactoring an existing one. Captures patterns that emerged during
> the env-backend refactor (2026-04-26), the codex audit (2026-04-27),
> and the TDMPC2 refactor + post-audit cleanup (2026-04-27).
>
> This is not a checklist for "polish to taste." Every item below
> corresponds to a real bug, silent failure, or documentation lie that
> shipped before the contract existed. New ports that skip these will
> reintroduce the same class of bug.

---

## §1. Env construction goes through the bundle

```python
from jax_rl.training import make_env_bundle

bundle = make_env_bundle(cfg, seed)
```

NOT `make_envs(cfg, seed)` — that's the legacy path kept around as a
shim. Bundle dispatch is what lets the same training script run on MJX
or gym backends without `if`-ladders.

**Bind `num_envs` from the bundle, not from cfg:**
```python
num_envs = bundle.num_envs  # may be capped by backend (gym → cpu_count)
```
Use `num_envs` for buffer shapes, action shapes, episode tracker dims,
and outer-loop counters. `cfg.num_envs` is what the user *requested*;
`bundle.num_envs` is what the backend *actually* runs. Mismatch → silent
shape-broadcast corruption on gym (codex audit, 2026-04-27).

**See:** `jax_rl/training/offpolicy_loop.py:73` (canonical pattern).

---

## §2. Backend gating — fail loudly, not silently

If your script can only run on MJX (e.g., uses `lax.scan` rollout, world-
model + sequence replay, BN running stats), gate explicitly at script
entry:

```python
bundle = make_env_bundle(cfg, seed)
if bundle.backend_kind != "mjx":
    raise ValueError(
        f"<your_script> requires an MJX env bundle, but env "
        f"{cfg.env_name!r} routes to backend_kind={bundle.backend_kind!r}.\n"
        f"<one-line reason why>\n"
        f"For gym envs, use <fallback script> instead."
    )
```

Reference: `scripts/train_ppo_fast.py:70`, `scripts/train_flashsac.py`,
`scripts/train_ppo_contraction.py`, `scripts/train_tdmpc2.py`.

If your script is universal (Python collect loop + jit'd inner ops, no
`lax.scan` over env steps), no gate. Eval must dispatch:
```python
from jax_rl.utils.eval import evaluate, evaluate_gym
_eval_fn = evaluate_gym if bundle.backend_kind == "gym" else evaluate
```

Do NOT call `evaluate(...)` directly — that's the MJX `lax.scan` path
and will silently fail on a gym env.

---

## §3. Artifact contract

Every checkpoint declares what shape it is. See
`jax_rl/training/artifact_contract.py` for the registry.

**Writers stamp `meta["artifact_kind"]` + `meta["artifact_version"]`:**
```python
from jax_rl.training.artifact_contract import stamp_meta, KIND_SHARED_ACTOR
meta = stamp_meta(meta, KIND_SHARED_ACTOR)
```

If your algo writes the shared shape (`actor_params.npy` + `orbax/`),
use `KIND_SHARED_ACTOR` and let `save_checkpoint()` handle it (already
stamps on every shared-loop save). If your algo writes a *different*
shape (TDMPC2's `actor_params.npz` + `world_model_params.npz` is the
precedent), add a new constant:

1. In `artifact_contract.py`: `KIND_DRQV2 = "drqv2_v1"` (or whatever).
2. Document the file shape in the kind catalog table.
3. Add `validate_<kind>_files(ckpt_dir)` checking required files exist.
4. In your save path, call `stamp_meta(meta, KIND_DRQV2)` before
   `json.dump`.

**Consumers gate on artifact_kind:**
```python
from jax_rl.training.artifact_contract import (
    assert_artifact_kind, KIND_SHARED_ACTOR, KIND_LEGACY_SHARED_ACTOR,
)
assert_artifact_kind(
    meta,
    allowed=[KIND_SHARED_ACTOR, KIND_LEGACY_SHARED_ACTOR],
    tool_name="<your_tool>.py",
    ckpt_path=ckpt_dir,
    redirect="For DrQv2 ckpts, use <whatever>",
)
```

Reference: `jax_rl/training/checkpointing.py:load_actor_for_inference`,
`deploy/policy_runner.py:PolicyRunner.__init__`.

**For Go2 Warp envs only:** also write `meta["control"]` via
`get_control_metadata()` (Phase D — Kp/Kd/action_scale/dts/contact_mode/
joint_order). `deploy/sim2sim_direct.py` reads this. See
`jax_rl/envs/locomotion/go2_warp_base.py:get_control_metadata`.

---

## §4. CLI plumbing

**Expose `build_parser()`** in every public-facing script. NOT just
`main()`:
```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(...)
    return parser

def main():
    args = build_parser().parse_args()
    ...
```

`docs/scripts/gen_cli_reference.py` reflects on `build_parser()` to
generate the CLI ref. Without it, your script won't appear.

**Register in `gen_cli_reference.py` SCRIPTS list:**
```python
SCRIPTS = [
    ...,
    ("train_drqv2.py", "scripts.train_drqv2"),
]
```

**Help strings on every flag.** They render into the docs verbatim.

**Usage strings in docstrings use `scripts/` prefix:**
```python
"""...
Usage:
    uv run python scripts/train_drqv2.py --env CheetahRun
"""
```

NOT `uv run python train_drqv2.py` — that path doesn't work post the
2026-04-24 scripts-relocation.

---

## §5. Algo file layout

**Single-file algos** for simple cases (SAC, TD3, FastSAC, FastTD3,
FlashSAC, PPOContraction). Stays as `jax_rl/algos/<name>.py`.

**Package layout** for big multi-component algos (TDMPC2 has encoder,
dynamics, reward, Q ensemble, policy prior, MPPI planner, runtime
helpers — too much for one file at >1000 LOC):
```
jax_rl/algos/<name>/
    __init__.py        # re-exports public symbols
    networks.py        # Flax modules
    agent.py           # TrainState + update step
    losses.py          # loss fns
    <planner>.py       # algo-specific (mppi, augmentation, etc.)
    runtime.py         # eval / build helpers (was at jax_rl/algos/<name>_runtime.py historically — keep it inside the package now)
```

**Always export from `jax_rl/algos/__init__.py`:**
```python
from jax_rl.algos.drqv2 import DrQV2State, make_update_step  # whatever your public API is
__all__ = [..., "DrQV2State", "make_update_step"]
```
Symmetric public API across algos. SAC/FastSAC/FlashSAC/TDMPC2 are all
exported there; new algos should follow.

---

## §6. Off-policy: use the shared loop where shape fits

If your algo is off-policy with:
- Standard transition replay (no sequence replay, no BN, no adaptive
  reward norm),
- Standard `update(state, batch)` shape,
- Standard explore_fn signature (`actor_params, obs, key → action`),

then plug into `jax_rl/training/offpolicy_loop.py:run_offpolicy_loop`.
Mirror `scripts/train_sac.py`, `train_td3.py`, `train_fast_sac.py`,
`train_fast_td3.py` — they're all ~110 lines: build optimizer + algo +
explore closure, call the shared loop.

If the shape doesn't fit (FlashSAC's BN running stats + Zeta noise +
adaptive reward norm; TDMPC2's sequence replay + world model + MPPI),
write a standalone loop. That's a deliberate choice, not a bug — but
add a one-line comment in the loop saying *why* it's standalone, like:
> "FlashSAC stays standalone because BN running stats + Zeta noise +
> adaptive reward norm don't fit `run_offpolicy_loop`'s shape."

---

## §7. Resume warmup is opt-in random, default policy

Off-policy resume has a known bug class (codex audit / 2026-04-26
session): the warmup gate refires on resume and corrupts the buffer
with random uniform actions, dropping first eval. Fix is universal:
`--resume-warmup {policy,random}` flag, default `policy`. Action
selection branch in your loop:

```python
is_warmup = len(buffer) < algo_cfg.min_buffer_size
use_random = is_warmup and (start_step == 0 or resume_warmup == "random")
if use_random:
    action = jax.random.uniform(...)  # cold-start exploration
else:
    action = explore_fn(actor_params, obs, key)  # use loaded policy on resume
```

Reference: `jax_rl/training/offpolicy_loop.py` (the canonical pattern),
`scripts/train_flashsac.py` (mirrored for the standalone loop). Skip
this if your algo is on-policy or model-based without a replay buffer.

---

## §8. Domain randomization is per-episode, not per-step

If you write or modify an env wrapper that touches DR: state.info
**persists** sampled DR across steps. Sample fresh values for the
reset path, use persisted for the step path, where_done merges fresh
into persisted on envs that just reset. This was a real bug in the
DomainRand wrapper before the codex audit (2026-04-27, fixed in
`826c326`).

Reference: `jax_rl/envs/wrappers/domain_rand.py`. Read its docstring
section on "DR persistence semantics" before touching any wrapper that
writes to state.info.

---

## §9. Tests stay hermetic by default

**Do NOT** put `jax.random.PRNGKey(...)` at module level in test files
— that triggers CUDA allocation at collection time and OOMs the whole
suite on constrained GPUs.

**DO** put PRNGKey calls inside test functions or fixtures. Lazy-import
heavy modules (Warp, MJX-specific envs) inside fixtures so collection
doesn't pull them in.

For new algos: add a `tests/test_<algo>.py` that runs hermetically (CPU
only — `JAX_PLATFORMS=cpu`) for unit tests, and a separate file for
integration tests that need GPU. The marker taxonomy is in flux but
the rule is constant: collection on a constrained GPU must not OOM.

---

## §10. README maturity label

New algos default to **Experimental research** until benchmark numbers
land. Add a row to the README §Algorithms table + the §Maturity
section explicitly labeling it. See README.md after commit `804c074`.

Don't claim Stable / Deploy-critical without:
- Benchmark numbers vs published baselines, OR
- A real deployment relying on it (Go2 hardware, etc.).

Most ports will sit at Experimental for a while. That's fine.

---

## §11. Audit your own work before merging

After landing a port, dispatch parallel Explore agents to verify
contract compliance (the pattern that surfaced the post-TDMPC2-refactor
gaps in 2026-04-27). Useful query axes:

1. **Bundle compliance:** does the script call `make_env_bundle`?
   Does it bind `num_envs` from the bundle? Does it gate or dispatch
   on `backend_kind`?
2. **Artifact contract:** does the writer call `stamp_meta`? Does the
   load path call `assert_artifact_kind`?
3. **CLI plumbing:** does it expose `build_parser`? Is it in
   `gen_cli_reference.py`'s SCRIPTS list?
4. **Test hermeticity:** any module-level `jax.random.PRNGKey`? Any
   hardcoded paths to a worktree?
5. **Stale docs:** does the docstring use `uv run python <script>` or
   `uv run python scripts/<script>`?

The pattern is documented in `codex_audit.md` (top-level, untracked).
Even if codex isn't running, the same questions are the right ones for
a self-audit.

---

## §12. Pointers

- Bundle layer: `.context/lessons/env_backends.md`
- Artifact contract: `jax_rl/training/artifact_contract.py` (kind
  catalog + helpers)
- Resume warmup: `.context/lessons/offpolicy.md` §"Resume Warmup"
- DomainRand: `jax_rl/envs/wrappers/domain_rand.py` docstring +
  `.context/lessons/autoreset_and_dr.md` (if extant)
- Existing algo refs:
  - Single-file: `jax_rl/algos/sac.py`, `flash_sac.py`
  - Package: `jax_rl/algos/tdmpc2/{networks,agent,losses,mppi,runtime}.py`
  - Standalone train: `scripts/train_flashsac.py`, `train_tdmpc2.py`
  - Shared-loop train: `scripts/train_sac.py`, `train_fast_sac.py`
- Recent codex audit + fix commits (2026-04-27): `826c326` (DomainRand),
  `a846501` (bundle.num_envs), `1059572` (Phase A), `3bd5a4c` (Phase D),
  `9a1c326` (Phase B+C).
