# Skill Discovery — Source Code Ground Truth

**Date:** 2026-05-02
**Purpose:** Paper text omits load-bearing implementation details. This doc captures the actual values and code shapes from each project's official repository. Use as the single source of truth when implementing or porting.
**Sources:** Parallel agent source audits (2026-05-02). Repos cloned to `/tmp/<paper>_src` during audit. File:line cites verify every claim.
**Companion:** `references/skill_discovery_validation.md` (validation contract per phase).

---

## Major paper-vs-source deltas (read first)

| Paper text says | Source actually does | Impact |
|---|---|---|
| DIAYN: hidden 256-ish, num_skills 20 | **`[300, 300]`, `num_skills=50` (`mujoco_all_diayn.py:30, 38`)** | Update SD-A defaults |
| DIAYN: SAC squashed Gaussian | **GMM K=4 mixture policy (`mujoco_all_diayn.py:29, 212`; `gmm.py:18-55`)** | Skip — DIAYN's TF1 reference is legacy SAC era; our SD-B FastSAC uses standard squashed Gaussian. (D3 is PPO with diagonal Gaussian MLP actor, also no GMM.) |
| DIAYN: skill per episode | **Skill per epoch** (1000 steps); coincides with episode only because epoch_length=max_path_length (`diayn.py:394`) | Stick with per-episode (what we want for sim2real) |
| METRA: λ=30 init | Confirmed (`tests/main.py:131`); **stored as `log(lambda)`** (`tests/main.py:391`) | Spec already correct |
| METRA: ‖Δφ‖ ≤ ‖s'−s‖ Lipschitz | **`dual_dist='one'` default → constraint is `‖Δφ‖²≤1`** (`metra.py:255`); 'l2'/'s2_from_s' are non-default ablations | **Important** — our SD-E impl should default to constant-1 |
| METRA: slack ε ≈ 1e-5 | **`dual_slack=1e-3`** (`tests/main.py:132`); applied as **upper-clamp** on `cst_penalty` (`metra.py:273`) | Update v1 plan's 1e-5 default |
| METRA: target φ network | **None** — only Q-targets exist | Spec correct |
| D3: lr=1e-4 | **`learning_rate=1e-3`, adaptive schedule** (`rsl_rl_usd_cfg.py:48`) | Update SD-E |
| D3: λ Dirichlet/truncated-Gaussian sum=1 | **Half-normal raised to skew, L2-normalized**: `F.normalize(\|N(0,1)\|^skew)` (`factoized_…py:475-488`) | Major correction for SD-E |
| D3: METRA λ=30 | 30 is initial Lagrangian; **`lambda_exploration=100`** is a separate disagreement coefficient (with `ensemble_size=1`, no actual disagreement signal exists) (`rsl_rl_usd_cfg.py:75-86`, `metra.py:40`) | Note for SD-E |
| D3: Dirichlet α curriculum 0.05→1.0 | **Adaptive cosine-sim driven** (×1.01 if cos>0.7, ×0.99 if cos<0.6) clamped to [0.05, 1.0] (`diayn.py:566-576`) | Update SD-E |
| D3: heading 2D | **`heading_rate` is 1D scalar** (`observations.py:858-862`); skill dim is 2 but obs dim is 1 | Note for SD-C/E |
| D3: per-factor value heads | **6 fully separate critic MLPs** (extrinsic + 5 factors); UCB term wired but `beta_advantage_UCB=0.0` → pure weighted sum (`ppo.py:372-409`) | SD-E architecture choice |
| D3: skill resampling per episode | **Every 375 steps (7.5s @ 50Hz)** for all factors (`anymal_usd_env_cfg.py:232-238`) | Note for SD-E |
| DADS: L=500 alt-z samples | **L=100 in shipped configs** (`*_offpolicy.txt:65`); paper's 500 is an unused ablation | Note if we ever port DADS |
| DADS: logsumexp stability | **Hard `np.clip(±50)`**, no logsumexp (`dads_agent.py:143-144`) | Confirms "improper bound" caveat |
| DUSDi: λ=0.1 disentanglement penalty headline | **`anti: false` default** in shipped `dusdi_diayn.yaml:45`; even when on, ramps in step-function at 1M env steps (`dusdi_diayn.py:308-310`) | Major: paper claim is gated behind a flag that defaults off |
| SkiLD: hierarchical PPO/PPO | **Lower policy is Rainbow DQN by default** for discrete envs; only upper graph-selection is PPO (`config.yaml:240`) | Note for manipulation extension |
| SkiLD: graph reselection L | **`option.timeout=20` env steps** | Note for future |

---

## DIAYN — `ben-eysenbach/sac` (TF1)

### Network architectures
- **Discriminator:** plain MLP `[300, 300] → num_skills` (logits, no output nonlin). ReLU hidden, Xavier init weights, zero biases. **No layer norm, no dropout, no batch norm.** Input: current state `s` only (NOT augmented obs, NOT next state). (`value_function.py:87-110`, `mlp.py:83-149`, `diayn.py:175-180`)
- **Policy:** GMM K=4 mixtures, `[300, 300]` MLP per component, tanh squashing (`gmm.py:18-55`, `mujoco_all_diayn.py:29, 210-216`). For our port, skip the GMM and use standard squashed Gaussian — D3 does this too.
- **V-net + Q-net:** `[300, 300]`, ReLU. **Only V has a target network**, no target Q (`sac.py:214-216`). Modern SAC uses target Q — we should follow modern SAC, not DIAYN's reference.

### Hyperparameters (launcher overrides class defaults)
| Param | Value | Source |
|---|---|---|
| `lr` (all 4 nets share) | `3e-4` Adam, no clip, no weight decay | `mujoco_all_diayn.py:26` |
| `tau` (Polyak) | `0.01` | `mujoco_all_diayn.py:28` |
| `gamma` | `0.99` | `mujoco_all_diayn.py:27` |
| `batch_size` | `128` | `mujoco_all_diayn.py:31` |
| `replay_buffer_size` | `1e6` (HC: `1e7`) | `mujoco_all_diayn.py:32, 65` |
| `scale_entropy` (α) | `0.1` **fixed** (no auto-tune) | `mujoco_all_diayn.py:39` |
| `num_skills` | `50` | `mujoco_all_diayn.py:38` |
| `min_pool_size` | `1000` | `mujoco_all_diayn.py:188` |
| `n_train_repeat` | `1` per env step | `mujoco_all_diayn.py:33` |

### Reward + buffer
- **Intrinsic reward computed at sample time**, recomputed in TF graph each gradient step from `(s, z)` of sampled batch (`diayn.py:172-190`).
- Replay stores `(aug_obs, action, terminal, aug_next_obs)` where `aug_*` carries the z active at collect time. **No relabeling.** (`diayn.py:418-424`)
- Discriminator trained on every transition in every minibatch — same minibatch as critic/value/policy (`diayn.py:443-446`).
- **Reward formula verbatim** (`diayn.py:181-190`):
  ```python
  reward = -softmax_cross_entropy(z_one_hot, logits) - log p(z) + EPS
  ```
  with `EPS = 1e-6`. Note `add_p_z=True` default but Mountain Car overrides to False.

### z lifecycle (key correctness call)
**z is sampled per-epoch (1000 steps), not per-episode.** Coincides with episode only because `epoch_length == max_path_length`. If you change one without the other, behavior breaks (`diayn.py:394, 397, 426-435`). For sim2real / Go2 use case, per-episode is what we want — explicit choice in our impl.

### File:line citations
- Discriminator graph + reward: `diayn.py:172-278`
- z sampling per epoch: `diayn.py:394`
- Hyperparameters: `mujoco_all_diayn.py:24-43`
- GMM policy: `gmm.py:18-55`

---

## METRA — `seohongpark/METRA` (PyTorch)

### Network architectures
- **Phi:** `[1024, 1024]` MLP, ReLU hidden, **linear output (no unit-norm projection)**. Built via Gaussian module but only `.mean` head is used → effectively deterministic phi. **No target phi network.** (`metra.py:99-103, 199`, `gaussian_mlp_module_ex.py:134`)
- **Encoder (pixel envs):** CNN, depth=48, kernels=(4,4,4,4), stride=2, ELU (`with_encoder.py:27`). Input divided by 255.
- **SAC actor/critic:** `[1024, 1024]` (`tests/main.py:295`).

### Hyperparameters
| Param | Value | Source |
|---|---|---|
| `dual_lam` init | `30` | `tests/main.py:131` |
| `dual_lam` parameterization | `log(lambda)`, ParameterModule | `tests/main.py:391` |
| `dual_slack` ε | `1e-3` | `tests/main.py:132` |
| `dual_dist` | `'one'` (constant 1) — NOT L2 by default | `tests/main.py:133` |
| `common_lr` (phi, dual, SAC) | `1e-4` | `tests/main.py:98-100, 134` |
| `sac_tau` | `5e-3` | `tests/main.py:106` |
| `gamma` | `0.99` | `tests/main.py:109` |
| `alpha` init | `0.01`, **auto-tuned** | `tests/main.py:102, 465` |
| `target_entropy` | `−|A|/2` (half SAC standard) | `metra.py:69` |
| `batch_size` | `256` | `tests/main.py:82` |
| `replay_buffer_size` | state: `1e6`, pixel quad/hum: `300k`, kitchen: `100k` | `tests/main.py:113` |
| `unit_length` (skill) | `1` (continuous → unit sphere) | `tests/main.py:128` |
| Critic loss scale | `× 0.5` (Stable-Baselines convention) | `sac_utils.py:53-54` |

### Update equations (verbatim)

**Phi loss** (`metra.py:245-289`):
```python
phi_x = traj_encoder(obs).mean
phi_y = traj_encoder(next_obs).mean
rewards = ((phi_y - phi_x) * z).sum(dim=1)        # alignment

if dual_dist == 'l2':
    cst_dist = (next_obs - obs).square().mean(dim=1)
elif dual_dist == 'one':
    cst_dist = ones_like(obs[:, 0])

cst_penalty = cst_dist - (phi_y - phi_x).square().mean(dim=1)
cst_penalty = cst_penalty.clamp(max=dual_slack)

dual_lam = log_dual_lam.exp().detach()             # stop_gradient
te_obj = rewards + dual_lam * cst_penalty
loss_te = -te_obj.mean()
```

**Dual lambda loss** (`metra.py:292-300`):
```python
loss_dual_lam = log_dual_lam * cst_penalty.detach().mean()  # stop_gradient(phi)
```
Adam `_gradient_descent` minimizes this. Note: this is gradient ascent on lambda when `cst_penalty < 0` (constraint violated) and descent when saturated. Verify sign convention before porting.

### z lifecycle
- **Continuous skill:** `z ~ N(0, I) → projected to unit sphere` (`unit_length=1`).
- **Discrete skill:** zero-centered one-hot scaled by `K/(K-1)`.
- **One skill per episode** (no mid-episode resample by default).

### File:line citations
- Phi loss + dual: `metra.py:245-300`
- Reward formula: `metra.py:194-229`
- Hyperparameters: `tests/main.py:98-136`

---

## D3 — `leggedrobotics/d3-skill-discovery` (PyTorch + IsaacLab)

### Factor specs (ANYmal-D)

| Factor | obs key | obs dim | skill dim | algo | Notes |
|---|---|---|---|---|---|
| `position` | `origin_spawn(pos_2d_only=True)` | 2 | 2 | METRA | World-frame xy relative to spawn pose, not normalized |
| `heading` | `heading_rate` | 1 | 2 | DIAYN | Yaw angular velocity scalar (NOT 2D heading angle as paper text suggests) |
| `base_height` | `base_height` | 1 | 2 | DIAYN | Height above terrain via height_scanner; `not_symmetric=True` (skill not mirrored) |
| `roll_pitch` | `roll_pitch` | 2 | 4 | DIAYN | wrap_to_pi |
| `base_vel` | `base_lin_vel` | 3 | 4 | DIAYN | Body-frame |

Total skill width = 2+2+2+4+4 = 14 + 6 weight slots = **20**.

### λ (factor weights)
- **Distribution:** `F.normalize(|N(0,1)|^skew, dim=-1)` — half-normal raised to `skew=1.0`, L2-normalized. Not Dirichlet, not sum-to-1.
- **`randomize_factor_weights=True`** for ANYmal — weights resampled per-env on done.
- **Embedded in z**: yes. Concat order `[skill_factor_1...factor_5, weight_extrinsic, weight_factor_1...factor_5]`.
- **No curriculum on λ.**
- (`factoized_…py:475-488`, `rsl_rl_usd_cfg.py:70`)

### Style + safety + regularization
**Style rewards** (extrinsic stream, gated by `λ_extrinsic` slot):
| Term | Weight |
|---|---|
| `undesired_contacts_thigh` | -30.0 |
| `undesired_contacts_shank` | -30.0 |
| `base_height` (penalty) | -10.0 |
| `flat_orientation` | -10.0 |
| `joint_vel_limits` | -0.01 |
| `terminated (upside_down)` | -5000.0 |

**Regularization** (always-on stream, **broadcast to every factor channel**, NOT gated by λ):
| Term | Weight |
|---|---|
| `dof_torques_l2` | -0.001 |
| `dof_acc_l2` | -2.5e-7 |
| `action_rate_l2` | -0.05 |
| `torque_limits` | -15.0 |
| `torque_limits_ratio (0.75)` | -15.0 |
| `joint_torques l2` (dup) | -0.01 |
| `joint_vel_limits soft` | -10.0 |
| `joint_vel_l2` | -0.01 |
| `joint_pos_limits` | -10.0 |

Composition (`ppo.py:217-227`):
```python
self.transition.rewards = torch.cat([rewards.unsqueeze(1), usd_reward], dim=1)
self.transition.rewards += regularization_reward    # broadcast all factors
```

### Symmetry augmentation (K=4)
Permutations: identity, left-right, front-back, 180° rotation (`mirroring.py:307-369`).

**Joint mappings** (HAA=0..3, HFE=4..7, KFE=8..11, ordering [LF, LH, RF, RH]):
- LR: HAA pairs (0↔2, 1↔3) **negate**; HFE/KFE pairs (4↔6, 5↔7, 8↔10, 9↔11) **swap only**.
- FB: HAA (0↔1, 2↔3) **swap only**; HFE/KFE swap+negate.
- 180°: all negate (0↔3, 1↔2, 4↔7, 5↔6, 8↔11, 9↔10).

**Skill permutations** by factor (`metra.py:561-569`, `diayn.py:884-927`):
- METRA `skill_dim=2`: lr/fb permute axes [1,0]; rot identity.
- DIAYN `skill_dim%4==0`: permute the four chunks.
- DIAYN `skill_dim=2`: lr/fb swap axes; rot identity.
- `not_symmetric=True` factors (e.g. `base_height`): repeat skill 4x (no permutation).

**Always applied at PPO update step** when `symmetry_augmentation=True`. Quadruples minibatch. **`symmetry_loss_weight=0.0`** for ANYmal — i.e. mirror-MSE not actually back-propped.

### Per-factor value functions
- **6 separate critic MLPs** (extrinsic + 5 factors). Same actor obs+skill input to each. NOT shared encoder + multi-head.
- Each critic: same shape as actor `[512, 256, 256]` ELU.
- Advantage aggregation: `Σ (advantage_i × λ_i) + β_UCB × Σ (std_within_ensemble × λ_i)`.
- **`beta_advantage_UCB = 0.0`** for ANYmal — UCB term wired but disabled, pure weighted sum.
- `ensemble_size=1` for METRA → no actual disagreement signal despite `lambda_exploration=100`.
- (`ppo.py:372-409`, `rsl_rl_usd_cfg.py:58, 61`)

### Discriminator + Phi networks
- **DIAYN discriminator:** SimBa (residual MLP) `[256, 256]` ELU (`diayn.py:105-116`). NOT plain MLP.
- **METRA Phi (StateRepresentation):** `[256, 256]` ELU (`metra.py:98-103`). No target net.
- Adam, lr=1e-4 for both. Lagrangian `lr_tau=5e-4`. **METRA slack `1e-5`** (D3 class default at `metra.py (D3):41`; differs from METRA reference repo's `1e-3`). Note divergence: when porting, decide which value to follow — D3-fidelity or METRA-reference-fidelity. Our `AuxNetConfig.dual_slack` defaults to `1e-3` (METRA reference); SD-E should ablate against `1e-5` if D3 reproduction matters.

### Hyperparameters
| Param | Value |
|---|---|
| `learning_rate` (PPO) | `1e-3` adaptive schedule |
| `entropy_coef` | `0.005` |
| PPO clip / desired_kl | `0.2` / `0.01` |
| epochs / mini_batches | `5` / `4` |
| `gamma` / `lam` | `0.99` / `0.95` |
| `max_grad_norm` | `1.0` |
| `num_steps_per_env` | `25` |
| `episode_length_s` | `30s` (1500 steps @ 50Hz, decimation=4, sim 200Hz) |
| `num_envs` | `4096` default scene; paper says 2048 |
| Skill resampling interval | `375` steps (7.5s) per factor |
| METRA `lambda_exploration` | `100` (disagreement coef, but ensemble_size=1 means dead) |
| METRA `sigma` (z sampling) | `10` |
| METRA `sample radius` | hardcoded `1.5` (`metra.py:348`) |
| DIAYN `initial_dirichlet_param` | `0.05` |
| DIAYN α curriculum | adaptive: ×1.01 if cos>0.7, ×0.99 if cos<0.6, clamped [0.05, 1.0] |
| DIAYN `lambda_skill_disentanglement` | `0.1` (DUSDi-style negative-MI penalty) — **D3 includes this!** |
| DIAYN `skill_disentanglement` | `True` (active) |
| Reward normalization | EMA per factor, decay 0.9 (`factoized_…py:202-207`) |
| Domain randomization | mass ±5kg, friction 0.6-0.9/0.4-0.8, pushes every 2-10s ±0.75 m/s |

**Critical: D3 already incorporates DUSDi-style disentanglement.** `skill_disentanglement=True` + `lambda_skill_disentanglement=0.1` (`rsl_rl_usd_cfg.py:90, 97`). This wasn't called out in the paper text or our prior audit. Means DUSDi's contribution is partially baked into D3's reference impl already.

> **D3 file path note:** the file is `factoized_unsupervised_skill_discovery.py` in the upstream repo — the codebase ships with a typo (`factoized_` instead of `factorized_`). All cites in this doc preserve the original spelling.

### Deploy contract
- Policy exported as TorchScript `.pt`; ONNX attempted in try/except.
- Operator slider GUI: per-factor block + per-factor weight slider + extrinsic weight. All weights L2-renormalized on each callback (matches training).
- Total slider z = 20-dim (5 factors + 6 weights).
- **No `hardware_ready` flag.** **No deploy clipping/safety/fallback** — slider widget hard-bounds values.
- `resampling_skill` toggle for auto-resample every 5/dt steps in sim eval.
- (`scripts/d3_rsl_rl/play.py`, `skill_gui.py`)

### Eval scripts NOT shipped
- Tables 1, 2, 3 numbers in paper are not in the repo. Training-time diversity metric exists (`factoized_…py:670-705`: pairwise mean L2 between trajectory-mean state-features for max 512 envs, split by close/far skill quantiles) but this is NOT the 10K-skill-sample protocol described in the paper.
- Goal-tracking downstream tasks ship via separate `Isaac-Goal-Tracking-Anymal-D-v0` env + standard PPO runner; aggregation script not in repo.

### File:line citations
- Factor list: `anymal_usd_env_cfg.py:189-256`
- λ sampling: `factoized_…py:475-488`
- Style + safety: `anymal_usd_env_cfg.py:159-179, 340-378`
- Symmetry: `mirroring.py:307-369`, `metra.py:515-570`, `diayn.py:883-933`
- Reward composition: `ppo.py:217-227`
- Per-factor critics: `runners/on_policy_runner_usd.py:64-81`
- Hyperparameters: `rsl_rl_usd_cfg.py:25-108`
- Deploy: `scripts/d3_rsl_rl/play.py:111-167`, `skill_gui.py:63-108`

---

## DADS — `google-research/dads` (TF1)

### Skill dynamics model
- MoG K=4 components, predicts `Δs = s' − s` (not s'), fixed identity covariance, mean-only outputs.
- Per-env hidden: Ant/DKitty `[512, 512]`, Humanoid `[1024, 1024]`. ReLU.
- Input batchnorm + scale=False/center=False output batchnorm.
- `reduced_observation=2` for Ant/DKitty xy → only feeds (x,y) to skill dynamics; `restrict_observation` slices off agent's own xy from policy input (`observation_omission_size=2`).

### Reward (verbatim, `dads_agent.py:94-146`)
```python
logp      = skill_dynamics.get_log_prob(input_obs,      cur_skill, target_obs)
logp_altz = skill_dynamics.get_log_prob(input_obs_altz, alt_skill, target_obs_altz)  # L=100 alt z's
logp_altz = np.array_split(logp_altz, num_reps)

intrinsic_reward = np.log(num_reps + 1) - np.log(
    1 + np.exp(np.clip(logp_altz - logp, -50, 50)).sum(axis=0)
)
```
Where `num_reps = 100` from `--random_skills=100`. Alt-z resampled per gradient step. **Hard `np.clip(±50)` is the only numerical guard** — no logsumexp. This is the "improper bound" of Appendix C.

### MPPI defaults (all configs use these — sparse preset NOT in repo)
| Param | Value |
|---|---|
| `planning_horizon` (H_P) | 1 |
| `primitive_horizon` (H_Z) | 10 |
| `num_candidate_sequences` (K) | 50 |
| `refine_steps` (R) | 10 |
| `mppi_gamma` | 10 |
| Init covariance | `1.5 · I` (fixed throughout, only means update) |
| Smoothing | EMA `β=0.9` |
| Test-time scoring | Uses **learned skill dynamics**, not env |
| Predict-state output | **MoG mean** (not sample) — `_use_modal_mean=True` |

### SAC backbone (off-DADS)
- `--agent_entropy=0.1`, implemented as `reward_scale_factor=1/α=10` (NOT standard SAC).
- `target_update_tau=0.005`, gamma=0.99 (Ant/DKitty) or 0.995 (Humanoid).
- **Replay buffer: 10k** for off-DADS (small).
- IS clip α=10: `np.clip(np.exp(cur_log_prob - old_log_prob), 1/10, 10)`. Used to reweight skill-dynamics MLE only, not SAC update.

### File:line citations
- Skill dynamics: `skill_dynamics.py:77-126, 278-279`
- Reward (clip+sum+log): `dads_agent.py:143-144`
- L=100: configs `*_offpolicy.txt:65`
- MPPI: `dads_off.py:892-1075`
- IS clip: `dads_off.py:346-353`

---

## DUSDi — `JiahengHu/DUSDi` (PyTorch)

### Critical default
**`anti: false` in shipped config** (`dusdi_diayn.yaml:45`). The headline disentanglement penalty is **off** unless explicitly enabled. Even when on, the `anti_coef` is a step function: 0 until `step >= anti_threshold = 1_000_000`, then jumps to 0.1 (`dusdi_diayn.py:308-310`).

### Two-discriminator architecture
Both per-factor:
- **Positive `q_phi^i`** (`diayn_modules.py:30-72`): `Linear(local_obs_size, 1024) → ReLU → Linear(1024, 1024) → ReLU → Linear(1024, skill_dim)`. Optional spectral norm (off by default).
- **Anti `q_psi^i`** (`diayn_modules.py:75-117`): Same architecture. Input: `s^¬i` constructed by `np.concatenate([range(0, partition[i]), range(partition[i+1], obs_dim)])`.

### Reward (verbatim, `dusdi_diayn.py:228-258`)
```python
reward = (d_pred_log_softmax[arange, z_hat] - log(1/skill_dim))
       - (d_anti_pred_log_softmax[arange, z_hat] - log(1/skill_dim)) * self.anti_coef
reward = reward.reshape(-1, diayn_skill_channel)   # NOT summed; per-factor
return reward * diayn_scale  # = 1.0
```
**Per-factor reward preserved as a vector** for per-factor Q heads. Sum across factors only when `monolithic_Q=True`.

Anti-loss is added **un-weighted** to discriminator loss (`d_loss + d_anti_loss`); only the reward is weighted by `anti_coef`.

### Q-decomposition (StateMaskCritic, `diayn_critics.py:272-327`)
- **Fully separate Q-networks per factor** via `FactoredValueHead` + `functorch.combine_state_for_ensemble` + `vmap`.
- Per factor: `Linear(obs+act, 512) → LayerNorm → Tanh → Linear(512, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 1)`.
- Twin Q (Q1, Q2). Target via parallel make_critics, soft-update tau=0.01.
- **`attn_logit` is fixed uniform** (`requires_grad=False` ones), not learned.
- TD loss per-head, summed (`ddpg.py:282-288`).

### Factor partitions (envs)
| Env | Partitions | n_factors | dim each | k |
|---|---|---|---|---|
| moma2d (2D Gunner) | `[0, 4, 8, 12]` | 3 | 4, 4, 4 | 5 |
| igibson | `[0, 3, 7, 10]` | 3 | 3, 4, 3 | 4 |
| particle | `range(0, 11)` | 9 (active) | 1 each | 5 |
| **DMC Walker** | **NOT in this codebase** | — | — | — |

### File:line citations
- Two-discriminator arch: `diayn_modules.py:30-117`
- Reward: `dusdi_diayn.py:228-258`
- Q-decomposition: `value_head.py:10-73`, `diayn_critics.py:272-327`
- `anti=False` default: `dusdi_diayn.yaml:45`

---

## SkiLD — `wangzizhao/SkiLD` (PyTorch + tianshou)

### Forward dynamics + pCMI
- **DynamicsGrad** (`Causal/grad.py:20-150`): channel multi-head self-attention with `num_output_heads=num_factors`. Per-factor MLP feature: `[128, 128]` ReLU. Attention: `num_heads=4`, `attn_dim=32`, `attn_out_dim=128`, `share_weight_across_kqv=True`. Per-factor predictor head (linear → categorical logits over `variable_longest`).
- **Discrete loss:** cross-entropy with `mixup_alpha=0.5`. Continuous would be regression on residual.
- **Mask training:** input slot `j` zeroed via `sa_feature * (1 - cmi_mask)`; ~50% chance of all-zero "no-mask" sampled during training.

### pCMI estimation (`Causal/grad.py:443-477`)
For each input slot j: compute `neg_logp_j` with mask=one_hot(j); compute `neg_logp_full` with mask=zeros. Then:
```python
cmi = (neg_logps - neg_logp[..., None]).clamp(min=0)
cmi_in_factor[..., factor_idx, :] = max over variables in factor   # variable→factor agg
graph = cmi_in_factor > cmi_threshold  # ε = 0.02
```
Post-process: if model wrong on factor i (`pred_correct=False`), zero its row; if factor unchanged, replace row with self-loop + action edge.

### Hierarchical components
- **Graph-selection PPO** (upper): tianshou discrete PPO over masked action space of `sample_action_space_n=256` slots indexing historically-observed unique subgraphs. Counts maintained as `(num_factors, 2^(num_factors+1))` integer arrays; graph→index hash via binary-vector dot with powers of 2. Reward `1/sqrt(count)` with count clipped at 1.
- **Skill-policy** (lower): **Rainbow DQN by default for discrete envs**, NOT PPO. Reward: `1[g_induced reached] · (graph_reward_scale + diayn_scale · log q(b|s', g) + diayn_scale · log K)` with `diayn_scale=0.5`, `graph_reward_scale=1`, `K=num_classes=4`.
- **DIAYN inner discriminator:** `Linear(input, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 4)`, optional spectral norm. Trained ONLY on `reached_graph=True` transitions.

### Training schedule
| Phase | Env steps | Active updates |
|---|---|---|
| 1: Dynamics warmup | 0 → 300k | Only dynamics; random actions everywhere |
| 2: Lower warmup | 300k → 500k | + Lower PPO + DIAYN; upper still random |
| 3: Joint | 500k → 20M | All three |

`option.timeout=L=20` env steps. Replay 3M, prioritized.

### File:line citations
- pCMI: `Causal/grad.py:443-477`
- Lower reward: `Option/Terminate/General/rtt_lower_graph.py:266-283`
- Count novelty: `Option/Terminate/General/rtt_upper_graph_count.py:21-52`
- Schedule: `train_HRL.py:53-78`, `Option/ihrl_trainer.py:158-185`

---

## Implementation implications for our SD-A → SD-E plan

### Already in our plan, confirmed correct
- DIAYN reward at sample time (`diayn.py:172-190`).
- DIAYN discriminator on current state s, not s'.
- Per-episode skill resampling (we deviate from DIAYN's per-epoch but matches D3's per-fixed-step intent).
- Replay-stores-raw-obs + skill_z separate (D3 stores option alongside obs).

### Defaults to update in SD-A plan
- DIAYN discriminator hidden default: `[256, 256] → [300, 300]` (DIAYN reference) or `[256, 256] elu SimBa` (D3 reference). **Pick D3-style** since we're hardware-targeting; SimBa-style residual MLP is an upgrade. For SD-A scaffolding, plain `[256, 256]` is fine — D3 chose ELU+SimBa, can swap later.
- DIAYN Adam lr default: `3e-4` (matches DIAYN reference and standard SAC).
- DIAYN num_skills test default: 4 (already in plan, fine for unit tests).

### Open Qs RESOLVED for SD-B
- Discriminator MLP default → `[256, 256]` ReLU plain is reasonable starting point; D3 uses SimBa+ELU upgrade for SD-C/E.
- Discriminator on `s` not `s'` (already in plan).
- per-skill rollout count M → use 10 sim, 3 hardware (our convention; no paper fixes M).

### Open Qs RESOLVED for SD-E
- METRA `dual_lam_init = 30` confirmed.
- METRA `dual_slack = 1e-3` (our v1 plan had `1e-5` — **must update**).
- METRA `dual_dist = 'one'` default (constant 1, NOT L2). Our SD-E impl should follow.
- METRA target phi network: **none**.
- METRA λ stored as `log(λ)` for positivity.
- METRA z continuous: `N(0, I) → unit sphere project`.
- D3 lr: `1e-3` adaptive (NOT 1e-4).
- D3 λ sampling: `F.normalize(|N(0,1)|^skew)` (NOT Dirichlet, NOT sum-1).
- D3 per-factor value: 6 fully separate critic MLPs.
- D3 already includes DUSDi-style negative-MI penalty (`lambda_skill_disentanglement=0.1`).

### New design decisions surfaced by source audit
1. **DIAYN's GMM K=4 policy is NOT load-bearing.** Modern SAC ships squashed Gaussian; our SD-B uses FastSAC with standard squashed Gaussian. (D3 is PPO with `[512, 256, 256]` MLP + diagonal Gaussian — different algo, also no GMM. The point: GMM mixture is a legacy artifact of pre-SAC-2018 code, not a DIAYN-required choice.)
2. **METRA's `dual_dist='one'` is the default but is the trivial constraint.** L2 / s2_from_s are paper-text-style ablations. For our SD-E port, ship 'one' as default and 'l2' as a flag. Document this surprising fact prominently.
3. **D3's `beta_advantage_UCB=0.0`.** UCB term wired but disabled. Keep wired in our port for future ablation; default to 0 to match.
4. **D3's λ is L2-normalized half-normal**, not Dirichlet/truncated-Gaussian. Big departure from paper text. Ship correctly.
5. **DUSDi's anti-discriminator default is OFF.** D3 turned it ON (`skill_disentanglement=True`). When we land DUSDi-style penalty, follow D3, not DUSDi reference.
6. **Eval scripts: D3 doesn't ship them.** We must build our own to replicate Tables 1/2/3.
7. **Hardware contract: D3 has no `hardware_ready` flag.** Deploy is sim-only via IsaacLab's `play.py`. Our SD-D `hardware_ready` gate is novel — keep it, it's a real-deploy safety improvement.

### Things to skip
- DIAYN's GMM K=4 policy (legacy SAC era).
- DIAYN's V-only target network (modern SAC has Q-targets).
- DADS's MoG K=4 dynamics model + MPPI (if we revisit, port directly; for now skip).
- SkiLD's Rainbow DQN lower policy (use PPO/SAC like everything else).
