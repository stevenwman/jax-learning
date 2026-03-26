# Go2 SAC Phase B — Plan & Research Notes

**Goal:** Validate off-policy (SAC) on Go2 env. Research Q: can SAC match PPO eval=233? Required prerequisite for DIAYN/METRA.

**Status:** Not started. PPO Phase A complete (seed 2100, eval 233).

---

## Baseline Config

From env_presets.py / current SAC defaults:
- 128 envs, gamma=0.97, tau=0.005, hidden_dim=(256,256)
- `--obs-norm` at sample time (eval 139 vs 97 without)
- Actor AND critic both see `state` (48d) — no asymmetric AC for off-policy
- Q LayerNorm enabled
- Raw obs stored in replay buffer, normalized at sample time

---

## Research Notes from Araffin's Blog Posts

Source: [SAC on Massive Parallel Sim](https://araffin.github.io/post/sac-massive-sim/) and [Tuning SAC for Isaac Sim](https://araffin.github.io/post/tune-sac-isaac-sim/)

### 1. Replay Ratio Must Drop With More Envs (HIGH PRIORITY)

Classical SAC assumes UTD >= 1 (multiple gradient steps per env step). With 1024+ envs, optimal replay ratio drops to ~0.03 — roughly **1 gradient step per 32 env steps**.

Formula: `replay_ratio = gradient_steps / (num_envs * train_freq)`

Implication: if we scale Go2 SAC from 128 to 1024 envs without adjusting, we'll massively over-update relative to new data. Either reduce gradient_steps or increase train_freq.

Their tuned values (1024 envs): gradient_steps=32, train_freq=1 → RR=0.03.

### 2. Action Bounds Matter for SAC (LIKELY OK)

Their root cause for SAC failure: Unitree A1 had action bounds [-100, 100] but the agent only used ~5% of the range. SAC's tanh squashing + rescaling to that range produces terrible initial exploration.

**Our status:** Go2 uses action_scale=0.5 with proper joint limits. PPO's trained action distribution stays within reasonable bounds. Probably fine, but could extract percentile bounds from PPO seed 2100 trajectory as a sanity check.

### 3. Policy Delay = 8

They found updating actor once per 8 critic updates improves wall-clock speed. Our FastSAC uses policy_delay=4 (holosoma paper). Vanilla SAC uses 1. Worth trying 4 or 8 for Go2.

### 4. Network Architecture [512, 256, 128]

They use a tapered 3-layer network with ELU activation + layer norm on both actor and critic. We use (256, 256) with ReLU. Could try larger/tapered if (256,256) underperforms.

### 5. gamma = 0.983

Confirms our gamma=0.97 is in the right ballpark. They found lower gamma favors shorter-term rewards, improving convergence speed for locomotion.

### 6. gSDE for Hard Exploration (FUTURE)

Generalized State-Dependent Exploration — noise resampled every N env steps instead of every step. Default Gaussian SAC failed on rough terrain; gSDE solved it. Combined with train_freq=10 and proportionally scaled gradient_steps.

Not needed for flat Go2 joystick. Note for future rough terrain / domain rand work.

### 7. N-Step Returns (n=3)

Multi-step TD targets helped on harder tasks. Inspired by FastTD3. We don't currently use n-step returns in vanilla SAC. Could add if SAC struggles with credit assignment on Go2's 1000-step episodes.

### 8. Entropy Coefficient

They use `auto_0.00947` (auto-tuned with low initial value). High initial entropy hurts — same finding as our PPO entropy_coef experiments. Let SAC's alpha auto-tuning handle it, but watch for collapse.

### 9. Percentile-Based Action Bounds (NICE TRICK)

Train PPO first, extract 2.5th/97.5th percentile of action distribution, use as SAC's action bounds. We have PPO seed 2100 + record_video.py saves _traj.npz with actions. Could extract bounds from that trajectory data.

---

## Experiment Plan

### Phase 1: Vanilla SAC at 128 envs (baseline)
```bash
uv run python train_offpolicy.py --algo sac --env Go2JoystickFlat --obs-norm
```
- Expect: eval ~140 range (based on prior run)
- Watch: Q1 growth, entropy/alpha trajectory, reward breakdown

### Phase 2: Scale to 1024 envs with adjusted replay ratio
- Reduce gradient_steps or increase train_freq to keep RR ~0.03-0.1
- Compare wall-clock and final eval vs 128 envs

### Phase 3: Hyperparameter sweep (if Phase 1-2 underperform)
- Policy delay: 1 vs 4 vs 8
- Network: (256,256) vs (512,256,128)
- Activation: ReLU vs ELU
- N-step returns: 1 vs 3

### Stretch: Action bound extraction from PPO
- Load PPO seed 2100 traj.npz
- Compute per-joint 2.5th/97.5th percentiles
- Override SAC action bounds accordingly

---

## Key Differences: Our Setup vs Araffin's

| | Araffin (SBX/SB3) | Ours (jax_rl) |
|---|---|---|
| Framework | Stable-Baselines3/SBX (JAX) | Custom JAX |
| Simulator | Isaac Sim / Isaac Lab | MuJoCo Playground (MJX) |
| Robot | Unitree A1/Go1/Go2, Anymal | Go2 |
| Action bounds | [-100, 100] (broken) → extracted | action_scale=0.5 (reasonable) |
| Obs norm | Via VecNormalize wrapper | Sample-time normalization |
| Critic | Standard twin Q | Twin Q + LayerNorm |
| Buffer | CPU (numpy) | GPU-resident (JAX) for Fast, numpy for vanilla |
