# GPU Management

## Nuclio (SAM ViT-H) — Recurring GPU Hog

A Nuclio serverless worker loads a SAM ViT-H vision model, eating 2.8-6.6GB of GPU memory. It respawns when killed because Docker has a restart policy.

**To temporarily stop:**
```bash
sudo docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h
sudo docker stop nuclio nuclio-local-storage-reader
```

**To permanently stop (survives reboot):**
```bash
sudo docker update --restart=no nuclio-nuclio-pth-facebookresearch-sam-vit-h
sudo docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

**Process tree:**
- PID varies — Python worker (`/opt/nuclio/_nuclio_wrapper.py`)
- Parent: Go `processor` binary
- Grandparent: `dashboard` (PID ~5952)
- All managed by Docker container `nuclio-nuclio-pth-facebookresearch-sam-vit-h`

**Why it exists:** Part of CVAT (annotation tool) setup. Uses SAM for interactive segmentation. Not related to our RL work.

## RTX 5080 Memory Budget (16GB)

| User | Typical usage |
|---|---|
| Nuclio SAM | 2.8-6.6GB (variable) |
| Training (1024 envs) | 8-12GB |
| Xorg/desktop | ~0.3GB |

With nuclio running, max ~1024 envs for training. Without nuclio, can push to 2048.

`XLA_CLIENT_MEM_FRACTION=0.7` is set in all training scripts to leave headroom.

## Buffer + Asymmetric Critic + Frame Stack = OOM (2026-04-10)

FastSAC default `buffer_size=4_194_304` is sized for symmetric critic with small obs. With:
- frame_stack=3 (raw 48d obs stored, but stacked 144d at sample time)
- asymmetric critic (extra `critic_obs` 122d + `critic_next_obs` 122d buffers)
- 1024 envs

The buffer alone consumes ~5 GB:
- raw obs: 4M × 48 × 4 = 768 MB
- critic_obs: 4M × 122 × 4 = ~1.95 GB
- critic_next_obs: 4M × 122 × 4 = ~1.95 GB
- + actions/rewards/dones (~250 MB)

Plus FastSAC C51 critic activations during the gradient step (8K × 768 × 101 atoms × 12 actions ≈ ~30 MB peak per Q × 4 Qs = ~120 MB), Warp simulation buffers (~3 GB), XLA scratch space.

**Symptom:** `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.95GiB` during JIT compilation, or `Warp CUDA error 2: out of memory in wp_cuda_graph_create_exec` during step.

**Mitigation:** Reduce `--buffer-size 2000000` (saves ~2.5 GB). Validates the env even if it caps the replay window. For long runs with full 4M buffer, would need a 24GB+ GPU.

**Lesson:** Buffer memory budget grows with `buffer_size × (raw_obs + critic_obs × 2)`, not just `obs_dim`. When enabling asymmetric critic OR raising obs_dim OR enabling frame stack, recheck the buffer size.
