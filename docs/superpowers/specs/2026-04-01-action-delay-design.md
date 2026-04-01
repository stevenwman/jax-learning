# Action Delay (Latency FIFO) — Design Spec

**Date:** 2026-04-01
**Status:** Approved (brainstorming complete)

## Motivation

Real robots have 40-120ms latency between policy inference and motor execution (communication, motor response, sensor sampling). Training in sim with zero latency produces policies that rely on instant feedback — they overshoot and fall on hardware. Every serious sim2real locomotion paper (Walk These Ways, Rapid Locomotion, DreamWaQ) includes action delay in training.

## Design

### Core Mechanism

`ActionDelayWrapper` — a Brax-compatible wrapper (same pattern as `FrameStackWrapper`) that maintains a FIFO buffer of actions in `state.info`.

**Data flow:**
```
Policy outputs action_t
  → Wrapper pushes action_t into FIFO back
  → Wrapper pops action_{t-K} from FIFO front
  → env.step() receives action_{t-K}
```

### Delay Specification

Delay is defined in **milliseconds**, converted to control steps at wrapper init:

```python
delay_steps = int(round(delay_ms / (ctrl_dt * 1000)))
```

This is robust to sim timestep changes — `--action-delay-ms 120` always means 120ms regardless of `ctrl_dt`.

### CLI Flags

- `--action-delay-ms 120` — fixed 120ms delay
- `--action-delay-range-ms 40 120` — randomized per-episode (uniform), overrides `--action-delay-ms`

### Buffer Design

**Shape:** `(max_delay_steps, action_dim)` — fixed for all envs (JIT-compatible).

**Fixed delay mode:** `min == max`, buffer is a simple FIFO. Push to back, pop from front.

**Randomized delay mode:** Buffer allocated at `max_delay_steps`. Per-env delay sampled on reset, stored in `state.info["action_delay_steps"]`. Read position is `max_delay_steps - sampled_delay` (earlier rows are stale padding).

Example with `max_delay_steps=6`, `sampled_delay=3`:
```
buffer = [stale, stale, stale, action_{t-3}, action_{t-2}, action_{t-1}]
                                ^-- read here (index 3)
```

**Step pseudocode (explicit FIFO logic):**
```python
def step(self, state, action):
    buffer = state.info["action_delay_buffer"]       # (max_delay, action_dim)
    delay = state.info["action_delay_steps"]          # int32

    # Pop: read the delayed action
    read_idx = self._max_delay_steps - delay
    delayed_action = buffer[read_idx]

    # Push: shift buffer left, write new action to end
    buffer = jp.roll(buffer, -1, axis=0)
    buffer = buffer.at[-1].set(action)

    # Reset handling: on done, zero buffer + re-sample delay
    rng = state.info["action_delay_rng"]
    rng, sample_key = jax.random.split(rng)
    new_delay = jax.random.randint(sample_key, (), self._min_delay, self._max_delay + 1)
    zero_buffer = jp.zeros_like(buffer)

    buffer = jp.where(state.done, zero_buffer, buffer)
    delay = jp.where(state.done, new_delay, delay)
    # Note: new_delay is sampled every step (JAX can't conditionally execute),
    # but jp.where only applies it when done=1

    # Step the inner env with the delayed action
    state = self.env.step(state, delayed_action)

    # Update state.info
    state.info["action_delay_buffer"] = buffer
    state.info["action_delay_steps"] = delay
    state.info["action_delay_rng"] = rng

    return state
```

**On reset:** Buffer filled with zeros (PD holds default pose for first K steps — realistic startup transient). Sampled delay drawn from `[min_delay, max_delay]`. Wrapper stores its own RNG key in `state.info["action_delay_rng"]` (split from initial `rng` in `reset()`), independent of the base env's RNG.

**state.info keys added:**
- `"action_delay_buffer"`: `(max_delay_steps, action_dim)` — the FIFO
- `"action_delay_steps"`: `int32` — per-env sampled delay (equals max in fixed mode)
- `"action_delay_rng"`: `PRNGKey` — wrapper-owned RNG for delay re-sampling

### Wrapper Ordering

```
Base env (Go2WarpJoystick)
  → ActionDelayWrapper       ← NEW (modifies action going in)
  → FrameStackWrapper        (modifies obs coming out)
  → VmapWrapper              (parallelize across envs)
  → EpisodeWrapper           (truncation)
  → AutoResetWrapper         (auto-reset on done)
```

Action delay before frame stack — it transforms actions, frame stack transforms observations. Both before Vmap so they get vmapped across envs.

### Delay Timestep Conversion

`ctrl_dt` is read from `env.dt` (the Brax env property for control timestep). This is the canonical source — matches what `step()` uses.

### Edge Cases

- **delay=0:** Skip wrapping entirely in `env_setup.py`. Do not instantiate the wrapper with `max_delay_steps=0` (empty buffer shape).
- **Auto-reset caveat:** Brax `AutoResetWrapper` does NOT reset `state.info`. The wrapper handles this explicitly with `jp.where(state.done, ...)` — same proven pattern as `FrameStackWrapper`. See step pseudocode above.

### Eval Policy

Eval env uses **fixed delay** at the expected real-robot latency (default 120ms), even when training uses randomized delay. Eval should test the target deployment condition, not a random one.

### Deploy Parity

`deploy/go2_constants.py` gets `ACTION_DELAY_MS = 120` constant. The deploy script can use it to apply matching delay, or the measured real latency.

### Integration Points

**`env_setup.py`:** After base env construction, **before FrameStackWrapper and before `wrap_for_training()`**:
```python
# Action delay first (modifies actions going in)
if action_delay_ms > 0 or action_delay_range_ms is not None:
    env = ActionDelayWrapper(env, ...)
# Frame stack second (modifies obs coming out)
if cfg.n_frame_stack > 1:
    env = FrameStackWrapper(env, ...)
# Then training wrappers
env = wrap_for_training(env, ...)
```

**Eval env** in `env_setup.py` also gets the wrapper with fixed delay (see "Eval Policy" section).

**`train_ppo_fast.py` / `train_offpolicy.py`:** Parse `--action-delay-ms` and `--action-delay-range-ms` flags, pass to env setup.

**`record_video.py`:** Apply same delay wrapper when loading env for video recording (must match training).

## Files Changed

| File | Change |
|------|--------|
| `jax_rl/envs/wrappers/action_delay.py` | NEW — ActionDelayWrapper |
| `jax_rl/envs/wrappers/__init__.py` | Export ActionDelayWrapper |
| `jax_rl/training/env_setup.py` | Wire wrapper based on CLI flags |
| `train_ppo_fast.py` | Add `--action-delay-ms` / `--action-delay-range-ms` flags |
| `train_offpolicy.py` | Same flags |
| `record_video.py` | Apply delay wrapper when recording |
| `deploy/go2_constants.py` | Add `ACTION_DELAY_MS = 120` |
| `tests/test_action_delay.py` | NEW — wrapper tests |

## Test Plan

1. **Fixed delay correctness** — push N distinct actions, verify output is delayed by exactly K steps
2. **Reset clears buffer** — after done=1, first K actions should be zeros (not stale from previous episode)
3. **Randomized delay** — verify different envs get different delays within range
4. **delay=0 passthrough** — action in = action out
5. **Shape preservation** — obs, reward, done unchanged by wrapper
6. **Composed with FrameStackWrapper** — both wrappers applied, both reset cleanly on done, no key collisions
7. **Integration** — full training loop with delay doesn't crash (smoke test, 10k steps)

## References

- Walk These Ways (WTW) — 120ms fixed delay for Unitree robots
- Rapid Locomotion — action delay as domain randomization
- FrameStackWrapper — `jax_rl/envs/wrappers/frame_stack.py` (pattern reference)
