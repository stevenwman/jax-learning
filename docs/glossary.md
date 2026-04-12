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
The differential entropy of the policy's action distribution, \(H(\pi(\cdot|s))\). SAC and its variants add an entropy bonus \(\alpha H\) to the objective, encouraging exploration. The temperature \(\alpha\) is auto-tuned by optimizing toward a [target entropy](#target-entropy).

### Episode
A trajectory from environment reset to termination or truncation. The undiscounted sum of rewards over an episode (the return) is the primary evaluation metric.

### Eval score
The mean undiscounted episodic return averaged over `num_eval_episodes` (default 10) rollouts using the deterministic policy. Reported periodically during training and used for algorithm comparison.

### GAE (Generalized Advantage Estimation)
Estimates how much better an action was compared to the average. GAE blends short-horizon estimates (low variance, may be biased) with long-horizon estimates (high variance, less biased) via a parameter lambda. Default `gae_lambda=0.95`. Used by PPO; implemented via backward `lax.scan` in `buffers/rollout.py`.

### Observation (obs)
The input vector the agent receives at each timestep. Go2 environments return a dictionary with `"state"` (51-dimensional deployable sensor readings: local velocity, gyro, gravity, linear velocity, accelerometer, joint positions/velocities, last action, command) and `"privileged_state"` (125-dimensional, adding clean sensor values, actuator forces, foot contacts/velocities, and external forces).

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

### Target entropy
How random the policy should be. SAC auto-tunes a temperature parameter to maintain this target — higher means more exploration, lower means more exploitation. Computed as \(\mathcal{H}_{\text{target}} = -\texttt{scale} \times \dim(\mathcal{A})\). SAC defaults to scale 0.5; FastSAC uses 0.0 (minimal exploration, stable at large batch sizes). FlashSAC uses a Gaussian entropy formula: \(\frac{1}{2} \dim(\mathcal{A}) \ln(2\pi e \, \sigma_{\text{target}}^2)\) with \(\sigma_{\text{target}} = 0.15\).

### UTD ratio (Update-to-Data)
The number of gradient updates performed per environment step collected. Higher UTD improves sample efficiency but increases compute per step. SAC defaults to `grad_updates_per_step=8`; TD3 defaults to 1; FastSAC and FastTD3 default to 8.

---

## Algorithms & Architecture

### Actor-Critic
An architecture pairing an actor (policy network) that selects actions with a critic (value network) that estimates expected returns. PPO uses a V-function critic (`VCritic`); SAC and its variants use twin Q-function critics (`QHead` or `DistributionalQHead`). Actor and critic are separate networks with independent parameters.

### Asymmetric critic
A training configuration where the critic receives a superset of the actor's observations. The actor sees only the 51d deployable `"state"` observations, while the critic additionally receives simulator-only information (125d `"privileged_state"`: clean sensor readings, contact forces, actuator torques). This enables sim-to-real transfer since the deployed actor never depends on privileged data. See the [Asymmetric Critic tutorial](tutorials/asymmetric-critic.md).

### C51
Instead of predicting a single expected return, C51 predicts a histogram (distribution) of possible returns using a fixed set of bins ("atoms"). This gives the critic richer learning signal. Used by FastSAC and FastTD3 with 101 atoms over [-20, 20], and by FlashSAC with 101 atoms over [-5, 5].

### Distributional RL
A family of methods that learn the full distribution of returns \(Z(s,a)\) rather than only the expected value \(Q(s,a) = \mathbb{E}[Z(s,a)]\). In this framework, C51 is the distributional method, implemented via `DistributionalQHead`. Distributional critics can improve learning stability, particularly in high-UTD regimes.

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
Proportional and derivative gains for the joint-level PD controller that converts policy actions to motor torques:

\[\tau = K_p(q^* - q) + K_d(0 - \dot{q})\]

Go2 Warp defaults are \(K_p = 20.0\), \(K_d = 0.5\). These gains are tightly coupled to the simulation timestep (`sim_dt=0.004`) and solver iterations; changing one without adjusting the others causes instability. Domain randomization applies per-episode scale factors to both.

### Privileged state
Simulator-only observations available to the critic but not the deployed actor. For Go2, this includes clean (noise-free) sensor readings, actuator forces, foot contact states, foot velocities, air time, and external forces applied to the torso. See [Asymmetric critic](#asymmetric-critic).

### Sim-to-real
Transferring a simulation-trained policy to a physical robot. Requires domain randomization for robustness, matched PD gains and action scaling between sim and real, and an actor that depends only on deployable observations (not privileged state). See the [Sim-to-Real tutorial](tutorials/sim2real.md).
