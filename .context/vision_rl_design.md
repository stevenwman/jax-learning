# Vision RL Design — CNN Encoder + Pixel Observations

**Status:** In progress (brainstorming phase)
**Date:** 2026-03-20
**Updated 2026-03-30:** Madrona MJX replaced by MJWarp GPU renderer built into `mujoco>=3.6.0`

## Goal

Add vision-based RL to the framework: train agents from pixel observations instead of (or alongside) state vectors. Two phases:
- **Phase A:** Validate CNN encoder works on existing tasks (compare vision vs state scores)
- **Phase B:** ManiSkill integration for GPU-accelerated vision training at scale

## Research Priority

Encoder first, then integration. But need at least one pixel source to verify encoder works.

## Pixel Sources (ranked by maturity)

### 1. MuJoCo Playground via MJWarp (primary target) — VERIFIED WORKING
- MJWarp rendering is part of official MuJoCo 3.6 (no separate install needed, `warp-lang` is the only dependency)
- `mjx.create_render_context()` + `mjx.render()` — GPU-accelerated batch rendering via Warp ray-tracing
- **Verified on our hardware (RTX 5080, 2026-03-20):**
  - Full pipeline works: physics step → render → unpack → `(nworld, 84, 84, 3)` uint8 RGB
  - 4 worlds at 84×84 resolution, real rendered content (pixel range [4, 194])
  - Render context uses `model.cam_resolution` for resolution, cameras defined in model XML
  - Output is packed uint32 (RGBA), unpack to uint8 with `view(uint8).reshape(nw, H, W, 4)[:,:,:,:3]`
- **Integration:** Same envs we already use. Set `model.cam_resolution[cam_idx] = [84, 84]` before creating render context
- **Constraint:** `nworld` is fixed at render context creation. Must match training batch size.
- **Memory:** Needs GPU headroom — OOM'd when SAC training was running simultaneously. Works fine with GPU free.
- Playground vision colab exists (`learning/notebooks/vision.ipynb`) for CartpoleBalance, PandaPickCube

### 2. ManiSkill (secondary, for diverse tasks)
- 50+ manipulation/locomotion tasks with built-in RGBD at 30k+ FPS (SAPIEN renderer)
- Standard Gymnasium API: `gym.make("PickCube-v1", obs_mode="rgbd")`
- **PyTorch-based** — needs DLPack zero-copy bridge to JAX
- Recommended integration: collect torch tensors → batch-convert to JAX via DLPack at training time
- Standard benchmarks: PickCube, PushT, PegInsertionSide, StackCube
- **Status:** Beta (v3.0.0b22) but sim/render pipeline is solid

### 3. DMC pixel wrapper via Gymnasium (fallback)
- `dm_control` has built-in pixel observation wrapper
- CPU rendering — slow but well-documented
- Would require Gymnasium adapter (Phase 4 work)

## Architecture

### Encoder Layer

**Separation of concerns:**
1. **Image preprocessing** (resolution, grayscale, frame stacking) — env/wrapper side
2. **Encoder architecture** (CNN vs MLP vs future ViT) — network side, swappable
3. **Augmentation** (DrQ random shifts) — training side, applied to batches

**Config per encoder type** (not one big config with unused fields):

```python
@dataclass
class MlpEncoderConfig:
    hidden_dim: tuple = (256, 256)
    feature_dim: int = 256
    norm: str | None = None

@dataclass
class CnnEncoderConfig:
    channels: tuple = (32, 64, 64)       # Nature CNN default
    kernel_sizes: tuple = (8, 4, 3)
    strides: tuple = (4, 2, 1)
    feature_dim: int = 256               # output dim (shared across all encoders)
    mlp_hidden_dim: tuple = ()           # optional dense layers after conv
                                          # () = direct projection to feature_dim
                                          # (256,) = one hidden layer after conv

@dataclass
class PixelObsConfig:
    resolution: tuple = (84, 84)
    frame_stack: int = 3
    grayscale: bool = False

@dataclass
class AugmentationConfig:
    enabled: bool = False
    random_shift: int = 4                # DrQ pad size. 0 = disabled
```

**Algo config points to encoder:**
```python
encoder_type: str = "mlp"    # "mlp", "cnn", "vit" (future)
encoder_config: MlpEncoderConfig | CnnEncoderConfig = field(default_factory=MlpEncoderConfig)
```

**Builder factory:** `make_encoder(encoder_type, encoder_config) → nn.Module`

All encoders output `(batch, feature_dim)`. The algo never knows what encoder type is used.

### Frame Stacking Strategy (researched 2026-04-01)

Three approaches exist in the literature:

| Approach | How | Used by | Tradeoff |
|---|---|---|---|
| **Raw frame stacking** | Concat N frames along channel dim (84×84×9 for 3 RGB) → one CNN pass | DrQ-v2, DQN, CURL | Simple, cheap, standard for off-policy |
| **Encode-then-stack** | Shared CNN per frame → concat latent vectors | Flare (NeurIPS 2021) | Cleaner temporal separation, N× CNN cost |
| **CNN + GRU** | CNN per frame → concat with prop → GRU for temporal memory | ANYmal, DeFM, LocoMamba | Best for long-horizon terrain estimation |

**Decision: Start with raw frame stacking (DrQ-v2 style).** Reasons:
1. It's the established default for off-policy pixel control (SAC/TD3 territory)
2. Playground's frame stacking logic already handles it inside the env
3. Our asymmetric critic **doesn't see images** — it gets privileged 122d state. Frame stacking is actor-only, so the simpler approach costs less.

**Upgrade path:** If raw stacking plateaus on locomotion (likely when terrain context matters over >10 steps), upgrade actor to CNN + GRU on latents. This is the standard locomotion approach (ANYmal in the wild, DeFM).

**Key insight — asymmetric critic simplifies vision:**
With the privileged critic (Pinto et al. 2018 pattern), the critic never touches pixels. Only the actor needs temporal visual context. This means:
- No frame stacking complexity on the critic side
- No shared encoder stop-gradient gymnastics (DrQ-v2 shares CNN between actor/critic with detached grads — we skip this entirely)
- Critic trains on clean 122d state, actor learns vision independently

**DreamWaQ / Walk These Ways are NOT pixel methods** — they use proprioception + learned terrain estimators. No frame stacking reference from them.

**Memory:**
- Store frames as **uint8** in replay buffer (4× savings vs float32)
- 100K buffer at 84×84×9 (3 RGB frames) ≈ 6.3GB — tight but feasible on 16GB
- Assemble stacks at sample time, not storage time (DrQ-v2 pattern)
- **Already built:** `JaxReplayBuffer` has `FrameStackConfig` for sample-time reconstruction — stores raw single frames, reconstructs stacks at sample time with episode boundary handling. Extend to uint8 pixel obs for vision.

**Obs normalization with stacked frames:**
- Normalize per-frame, not per-stacked-obs. Each frame position has a different temporal distribution. Compute running stats on raw single-frame obs (same distribution regardless of frame position), apply to each slice of the stack independently. This prevents blending statistics across time offsets.

### CNN Encoder (Nature CNN + optional MLP)

```
Input: (batch, H, W, C*frame_stack)    # e.g., 84×84×9 for 3 RGB frames
  → Conv(32, 8x8, stride 4) → ReLU
  → Conv(64, 4x4, stride 2) → ReLU
  → Conv(64, 3x3, stride 1) → ReLU
  → Flatten
  → [optional MLP hidden layers]
  → Dense(feature_dim) → ReLU
Output: (batch, feature_dim)
```

New file: `jax_rl/networks/encoders/cnn.py`

### Image Preprocessing / Pixel Wrapper — Playground can handle it (NOT YET INTEGRATED)

**Discovery (2026-03-20, updated 2026-03-30):** MuJoCo Playground has vision support via the MJWarp GPU renderer built into `mujoco>=3.6.0`. **Madrona MJX is no longer used or required.** This is a research finding — **not yet integrated into our training scripts.** No CNN encoder, no `--vision` flag exists yet.

**Rendering API (MJWarp, `impl="warp"` required):**

```python
# Create render context — nworld fixed at creation, must match training batch size
rc = mjx.create_render_context(mjm=model, nworld=N, cam_res=(84, 84), render_rgb=[True])

# Update BVH for current geometry (call before render each step)
rc = mjx.refit_bvh(model, data, rc)

# GPU ray-trace render — returns (rgb, depth), all GPU-resident, no CPU round-trip
rgb, depth = mjx.render(model, data, rc)

# Extract per-camera RGB (packed uint32 RGBA → unpack to uint8)
pixels = mjx.get_rgb(rc, cam_idx=0, pixels=pixels_buf)
# Unpack: pixels.view(jnp.uint8).reshape(nworld, H, W, 4)[:, :, :, :3]
```

- **`mjx.render()` only works with `impl="warp"`** — pure-JAX MJX cannot render. Vision requires Warp backend.
- Only **CartpoleBalance** and **FrankaPickCubeCartesian** have vision implemented in Playground. Locomotion envs (Go1, Go2) don't have it yet — we need to add the render context to `Go2WarpJoystick`.
- `wrap_for_training()` does **NOT** have a `vision=True` parameter. Vision is handled inside the env class itself (see Playground's CartpoleBalance vision env). The wrapper just wraps the env normally.
- Obs come out as `state.obs` with pixel keys like `pixels/view_0`
- Frame stacking (grayscale, sequential frames) is handled by the env
- **No `madrona_mjx` dependency** — only `mujoco>=3.6.0` and `warp-lang>=1.11`
- Memory cost: MJWarp BVH + render buffers (smaller footprint than the old Madrona batch renderer)

**What we still need to build:**
- Render context added to `Go2WarpJoystick` env (follow Playground's CartpoleBalance vision env pattern)
- CNN encoder to process the pixel obs → feature vector
- Integration into our off-policy training scripts (SAC/TD3)
- Brax already has `networks_vision.make_ppo_networks_vision` for PPO — we can reference this

**What we do NOT need to build:**
- No pixel wrapper
- No frame stacking logic
- No separate rendering pipeline
- No camera configuration (handled by env config)

Source: `learning/notebooks/vision.ipynb` in Playground package (`learning/notebooks/training_vision_1.ipynb` was the old Madrona-based colab)

### DrQ Data Augmentation

Random image shifts applied to sampled batches during training:
```python
def random_shift(images, key, pad=4):
    padded = jnp.pad(images, ((0,0), (pad,pad), (pad,pad), (0,0)))
    crop_h, crop_w = jax.random.randint(key, (2,), 0, 2*pad)
    return jax.lax.dynamic_slice(padded, (0, crop_h, crop_w, 0), images.shape)
```

Applied to both `obs` and `next_obs` with different random crops. Toggled via `AugmentationConfig.enabled`.

Location: `jax_rl/utils/augmentation.py`

### Training Loop Changes

One script per algo with `--vision` flag, not separate vision scripts:
- `--vision` → loads env with `vision=True`, uses CNN encoder, optionally enables augmentation
- Without `--vision` → current behavior (state obs, MLP encoder)

When vision is enabled:
1. Env loaded with `config_overrides={"vision": True}` — Playground handles rendering + frame stacking
2. Obs are `(batch, H, W, C*frames)` instead of `(batch, obs_dim)`
3. Buffer stores pixel obs as uint8 (converted to float32 on sample, normalized to [0,1])
4. Optional DrQ augmentation applied after `buffer.sample()`, before `algo.update()`
5. Encoder is CNN instead of MLP — algo receives same `(batch, feature_dim)` output either way

### Memory Budget — Test Empirically

Pixel replay buffers are much larger than state buffers. Rough estimates suggest 1M entries at 84×84×9 won't fit on 16GB. Likely need smaller buffer (50k-100k entries) and/or lower resolution (64×64). Pin this for empirical testing once the pipeline works. Note: MJWarp BVH + render buffers are expected to be a smaller fixed overhead than the old Madrona batch renderer was.

### Builders Unification (prerequisite)

**DONE (2026-03-22):** All algos now use `Actor`/`DeterministicActor`/`VCritic` from `builders.py`. Encoder is swappable — adding CNN means changing builders, zero algo changes. See `.context/archive/builders_unification_plan.md` for the original plan.

**Order of execution:**
1. Add render context to `Go2WarpJoystick` env and verify `mjx.render()` pipeline on our hardware (RTX 5080). Use CartpoleBalance vision env as reference. Requires `impl="warp"`.
2. ~~Unify builders (`make_encoder` factory)~~ — DONE (2026-03-22)
3. Add CNN encoder (`jax_rl/networks/encoders/cnn.py`)
4. Add encoder configs (`CnnEncoderConfig`, `AugmentationConfig`)
5. Add `--vision` flag to `train_offpolicy.py` (consolidated script replaces train_sac.py/td3.py)
6. Verify on CartpoleBalance from pixels (Playground vision.ipynb as reference)
7. Add DrQ augmentation as optional toggle
8. Benchmark CNN vs CNN+DrQ vs state-based on same task
9. ManiSkill integration (Phase B — Gymnasium adapter + DLPack bridge)

## Open Questions

### Resolved
- [x] MJWarp batched render pipeline — **VERIFIED WORKING** (2026-03-20). 4 worlds × 84×84 RGB, real pixels.
- [x] Pixel wrapper needed? — **NO**. Playground handles rendering + frame stacking inside env class. Vision is integrated via `mjx.create_render_context` / `mjx.render`, not a separate wrapper.
- [x] Separate vision train scripts? — **NO**. One script per algo with `--vision` flag.
- [x] Madrona MJX dependency? — **GONE** (2026-03-30). MJWarp renderer is built into `mujoco>=3.6.0`. Only `warp-lang>=1.11` required.
- [x] `vision=True` on `wrap_for_training()`? — **NO such parameter**. Vision logic lives inside the env class. Wrapper is unaware of vision.

### Pinned for empirical testing
- [ ] GPU memory budget — pixel replay buffer sizing on 16GB. Test with different buffer sizes / resolutions.
- [ ] MJWarp rendering speed at 1024 envs — benchmark FPS on RTX 5080 once render context added to Go2WarpJoystick.
- [ ] Memory overhead of BVH + render buffers at various `nworld` sizes — expected smaller than old Madrona footprint.

### Deferred
- [ ] ViT encoder — config structure supports it, implement when needed
- [ ] ManiSkill integration — Phase B, after Playground vision is working
- [ ] Separate actor/critic encoders — DrQ-v2 uses shared encoder with stop-gradient on critic. Start with shared, add option later if needed.

## Future: ViT Encoder

```python
@dataclass
class ViTEncoderConfig:
    patch_size: int = 8
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    feature_dim: int = 256
    mlp_hidden_dim: tuple = ()
```

Same `make_encoder()` factory, same `(batch, feature_dim)` output. The config structure already supports this — just add the module when needed.

## References

- DrQ: Kostrikov et al., "Image Augmentation Is All You Need" (2020)
- Nature CNN: Mnih et al., "Human-level control through deep RL" (2015)
- MuJoCo Playground vision colab: `learning/notebooks/vision.ipynb`
- ManiSkill 3: `maniskill.readthedocs.io`
- MJWarp: Built into MuJoCo 3.6+ (`pip install mujoco>=3.6.0 warp-lang>=1.11`). No separate install.
