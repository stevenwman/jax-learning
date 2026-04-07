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
