# Glossary

Key terms used throughout this framework, grouped by domain.

---

## JAX

### JIT (Just-In-Time compilation)
JAX traces Python functions and compiles them to optimized XLA programs for GPU/TPU execution. The first call incurs compilation overhead; all subsequent calls with the same input shapes run the cached compiled code. Every algorithm in this framework wraps its `update` and `select_action` methods with `@jax.jit`.

### lax.scan
A JAX primitive that compiles a sequential loop into a single fused XLA operation. Required inside JIT'd functions because Python `for`-loops are unrolled at trace time and cannot depend on runtime values. PPO uses nested `lax.scan` for both epoch iteration and minibatch processing.

### Pytree
JAX's abstraction for arbitrarily nested containers (dicts, tuples, lists, dataclasses) of arrays. All `TrainingState` objects, optimizer states, and network parameters in this framework are pytrees, enabling `jax.tree.map` to apply transformations uniformly across their leaves.

### vmap
Vectorized map: transforms a function that operates on a single example into one that operates over a batch dimension, without rewriting the function body. Used in this framework to parallelize environment stepping and network inference across `num_envs` instances.

---

## RL Fundamentals

### Action space
The set of valid actions at each timestep. All environments in this framework use continuous action spaces bounded to [-1, 1]. The Go2 quadruped has a 12-dimensional action space (3 joints per leg x 4 legs), scaled by `action_scale` before converting to motor targets.

### Entropy (in RL)
The differential entropy of the policy's action distribution, H(pi(.|s)). SAC and its variants add an entropy bonus alpha * H to the objective, encouraging exploration. The temperature alpha is auto-tuned by optimizing toward a target entropy (see Target entropy).

### Episode
A trajectory from environment reset to termination or truncation. The undiscounted sum of rewards over an episode (the return) is the primary evaluation metric.

### Eval score
The mean undiscounted episodic return averaged over `num_eval_episodes` (default 10) rollouts using the deterministic policy. Reported periodically during training and used for algorithm comparison.

### GAE (Generalized Advantage Estimation)
Computes advantage estimates for PPO by exponentially weighting n-step TD errors with parameter lambda, interpolating between TD(0) (lambda=0, low variance, high bias) and Monte Carlo (lambda=1, high variance, low bias). Default `gae_lambda=0.95`. Implemented via backward `lax.scan` in `buffers/rollout.py`, with truncation steps zeroed out to prevent cross-episode leakage.

### Observation (obs)
The input vector the agent receives at each timestep. Go2 environments return a dictionary with `"state"` (48-dimensional deployable sensor readings: local velocity, gyro, gravity, joint positions/velocities, last action, command) and `"privileged_state"` (122-dimensional, adding clean sensor values, actuator forces, foot contacts/velocities, and external forces).

### Off-policy
Algorithms that learn from transitions stored in a replay buffer, decoupling data collection from optimization. SAC, TD3, FastSAC, FastTD3, and FlashSAC are all off-policy in this framework.

### On-policy
Algorithms that optimize using only data collected under the current policy, discarding it after each update. PPO is the sole on-policy algorithm in this framework.

### Policy
The function mapping observations to actions (or action distributions). In this framework, policies are Flax `nn.Module` instances: `Actor` (encoder + `GaussianHead`) for stochastic policies (PPO, SAC, FastSAC, FlashSAC) and `DeterministicActor` (encoder + `DeterministicHead`) for deterministic policies (TD3, FastTD3).

### Replay buffer
A GPU-resident circular FIFO buffer (`JaxReplayBuffer`) storing `(obs, action, reward, next_obs, done, truncation)` transitions for off-policy training. Both `add_batch` and `sample` are JIT'd. Default capacity is 4M transitions for SAC/FastSAC or 1M for TD3/FastTD3/FlashSAC.

### Reward shaping
Designing the reward function to guide learning toward desired behavior. Go2 locomotion rewards are a weighted sum of tracking terms (linear/angular velocity commands) and regularization costs (torques, energy, action rate, joint limits, foot clearance). All terms are defined declaratively via `RewardTerm` specs.

### Truncation
An episode ending due to a time limit rather than a terminal state (e.g., falling). The framework handles truncation distinctly from termination: in PPO's GAE, truncated timesteps have their TD error zeroed out; in off-policy algorithms, a truncation mask prevents learning from invalid bootstrap targets at episode boundaries.

### UTD ratio (Update-to-Data)
The number of gradient updates performed per environment step collected. Higher UTD improves sample efficiency but increases compute per step. SAC defaults to `grad_updates_per_step=8`; TD3 defaults to 1; FastSAC and FastTD3 default to 8.

---

## Algorithms & Architecture

### Actor-Critic
An architecture pairing an actor (policy network) that selects actions with a critic (value network) that estimates expected returns. PPO uses a V-function critic (`VCritic`); SAC and its variants use twin Q-function critics (`QHead` or `DistributionalQHead`). Actor and critic are separate networks with independent parameters.

### Asymmetric critic
A training configuration where the critic receives a superset of the actor's observations. The actor sees only the 48d deployable `"state"` observations, while the critic additionally receives simulator-only information (122d `"privileged_state"`: clean sensor readings, contact forces, actuator torques). This enables sim-to-real transfer since the deployed actor never depends on privileged data. See the [Asymmetric Critic tutorial](tutorials/asymmetric-critic.md).

### C51
A distributional RL method that represents Q(s,a) as a categorical distribution over a fixed set of evenly spaced atoms. Used by FastSAC and FastTD3 with 101 atoms over [-20, 20] by default, and by FlashSAC with 101 atoms over [-5, 5]. The critic is trained with categorical cross-entropy loss against Bellman-projected target distributions.

### Distributional RL
A family of methods that learn the full distribution of returns Z(s,a) rather than only the expected value Q(s,a) = E[Z(s,a)]. In this framework, C51 is the distributional method, implemented via `DistributionalQHead`. Distributional critics can improve learning stability, particularly in high-UTD regimes.

### MLP (Multi-Layer Perceptron)
A fully connected feedforward network. The default architecture for all actors and critics in this framework, implemented as `MlpEncoder`. FlashSAC is the exception, using inverted residual blocks with BatchNorm (`FlashSACBlock`) instead of plain MLPs.

### Preset
A `TrainConfig` instance with algorithm-specific sub-configs and proven hyperparameters for a specific environment. Defined in `configs/env_presets.py`. See [Environment Presets](reference/env-presets.md).

---

## Environments & Hardware

### Domain randomization (DR)
Varying physics parameters (friction, mass, motor strength, damping, armature, PD gain scales) across episodes so the policy learns to be robust to parameter uncertainty. Declared per-environment via `DRSpec` lists. Essential for sim-to-real transfer with Go2.

### MJX
MuJoCo's JAX-native physics backend (`impl="jax"`). Runs the full MuJoCo pipeline as differentiable JAX code on GPU. Used for DM Control benchmarks in this framework. The Go2 MJX environment is archived in favor of MuJoCo Warp.

### MuJoCo
A contact-rich physics simulator for robotics and RL. All environments in this framework are built on MuJoCo models (MJCF XML), with GPU execution provided by MJX or MuJoCo Warp backends.

### MuJoCo Warp
GPU-accelerated MuJoCo backend using NVIDIA Warp kernels (`impl="warp"`). The primary backend for Go2 locomotion, supporting full collision geometry (cylinders + boxes) that MJX's convex-only pipeline cannot handle efficiently.

### PD gains (Kp, Kd)
Proportional and derivative gains for the joint-level PD controller that converts policy actions to motor torques: `tau = Kp * (target - q) + Kd * (0 - dq)`. Go2 Warp defaults are `Kp=20.0`, `Kd=0.5`. These gains are tightly coupled to the simulation timestep (`sim_dt=0.004`) and solver iterations; changing one without adjusting the others causes instability. Domain randomization applies per-episode scale factors to both.

### Privileged state
Simulator-only observations available to the critic but not the deployed actor. For Go2, this includes clean (noise-free) sensor readings, actuator forces, foot contact states, foot velocities, air time, and external forces applied to the torso. See Asymmetric critic.

### Sim-to-real
Transferring a simulation-trained policy to a physical robot. Requires domain randomization for robustness, matched PD gains and action scaling between sim and real, and an actor that depends only on deployable observations (not privileged state). See the [Sim-to-Real tutorial](tutorials/sim2real.md).

### Target entropy
The entropy setpoint for SAC's automatic temperature tuning. Computed as `-target_entropy_scale * action_dim`. SAC defaults to `target_entropy_scale=0.5` (moderate exploration). FastSAC defaults to `target_entropy_scale=0.0` (target entropy = 0), which prevents entropy collapse at large batch sizes. FlashSAC uses a different formulation: `0.5 * action_dim * log(2*pi*e*sigma_target^2)` with `sigma_target=0.15`.
