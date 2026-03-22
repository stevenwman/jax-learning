# Vision RL Design — CNN Encoder + Pixel Observations

**Status:** In progress (brainstorming phase)
**Date:** 2026-03-20

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

### CNN Encoder (Nature CNN + optional MLP)

```
Input: (batch, H, W, C*frame_stack)
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

**Discovery (2026-03-20):** MuJoCo Playground has vision support via Madrona MJX. This is a research finding — **not yet integrated into our training scripts.** No CNN encoder, no `--vision` flag, no `madrona_mjx` install exists yet. The API below is from Playground's docs, not our code.

```python
config_overrides = {
    "vision": True,
    "vision_config.render_batch_size": num_envs,
}
env = dm_control_suite.load(env_name, config_overrides=config_overrides)
env = wrapper.wrap_for_brax_training(env, vision=True, num_vision_envs=num_envs, ...)
```

- Obs come out as `state.obs` with pixel keys like `pixels/view_0`
- Frame stacking (grayscale, sequential frames) is handled by the env
- Rendering at 290k transitions/sec on RTX 4090 (1024 envs, 64x64)
- **Requires `madrona_mjx` package** installed locally
- Recommends 24GB+ VRAM (RTX 4090). Our 16GB RTX 5080 may need fewer envs (256-512)
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` recommended to leave room for Madrona

**What we still need to build:**
- CNN encoder to process the pixel obs → feature vector
- Integration into our off-policy training scripts (SAC/TD3)
- Brax already has `networks_vision.make_ppo_networks_vision` for PPO — we can reference this

**What we do NOT need to build:**
- No pixel wrapper
- No frame stacking logic
- No rendering pipeline
- No camera configuration (handled by env config)

Source: `learning/notebooks/training_vision_1.ipynb` in Playground package

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

Pixel replay buffers are much larger than state buffers. Rough estimates suggest 1M entries at 84×84×9 won't fit on 16GB. Likely need smaller buffer (50k-100k entries) and/or lower resolution (64×64). Pin this for empirical testing once the pipeline works.

### Builders Unification (prerequisite)

All algos need to use `make_encoder()` factory instead of hardcoding `MlpEncoder(...)`. This is the builders unification plan already documented in `.context/builders_unification_plan.md`.

**Order of execution:**
1. Install `madrona_mjx` and verify vision env loads on our hardware
2. Unify builders (`make_encoder` factory) — prerequisite for swappable encoders
3. Add CNN encoder (`jax_rl/networks/encoders/cnn.py`)
4. Add encoder configs (`CnnEncoderConfig`, `AugmentationConfig`)
5. Modify one train script (train_sac.py) to support vision obs + CNN encoder
6. Verify on CartpoleBalance from pixels (Playground colab baseline: 57s to solve on 4090)
7. Add DrQ augmentation as optional toggle
8. Benchmark CNN vs CNN+DrQ vs state-based on same task
9. ManiSkill integration (Phase B — Gymnasium adapter + DLPack bridge)

## Open Questions

### Resolved
- [x] MJWarp batched render pipeline — **VERIFIED WORKING** (2026-03-20). 4 worlds × 84×84 RGB, real pixels.
- [x] Pixel wrapper needed? — **NO**. Playground has built-in `vision=True` config with Madrona MJX rendering + frame stacking.
- [x] Separate vision train scripts? — **NO**. One script per algo with `--vision` flag.

### Pinned for empirical testing
- [ ] GPU memory budget — pixel replay buffer sizing on 16GB. Test with different buffer sizes / resolutions.
- [ ] MJWarp rendering speed at 1024 envs — benchmark FPS on RTX 5080.
- [ ] Warp physics vs MJX physics — Playground's `vision=True` mode uses Madrona MJX (separate from raw MJWarp `Impl.WARP`). Need to verify which physics backend Madrona uses and if it's compatible with our existing env setup.
- [ ] `madrona_mjx` installation — requires local build, CUDA compatibility. Need to verify on our setup.

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
- MJWarp: Part of MuJoCo 3.6 (`pip install mujoco-warp`)
