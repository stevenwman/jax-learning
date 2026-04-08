# Glossary

Key terms used throughout this framework.

### Action space
The set of actions an agent can take at each timestep. Locomotion tasks use continuous action spaces; the Go2 environment has a 12-dimensional action space (one per joint).

### Actor-Critic
An architecture where the *actor* (policy) selects actions and the *critic* estimates the value function. Both PPO and SAC use actor-critic architectures with separate MLPs.

### Asymmetric critic
A training setup where the critic receives more information than the actor. The critic accesses privileged state (contacts, friction) while the actor only sees deployable observations. See the [Asymmetric Critic tutorial](tutorials/asymmetric-critic.md).

### C51
A distributional RL algorithm that represents the Q-value as a categorical distribution over a fixed set of atoms. Used by FastSAC and FastTD3 in this framework.

### Distributional RL
An approach that models the full distribution of returns rather than just the expected value. Can improve stability in environments with high reward variance.

### Domain randomization (DR)
Randomizing physics parameters (friction, mass, motor strength) during training so the policy generalizes to real-world variation. Essential for sim-to-real transfer with Go2.

### Entropy (in RL)
A measure of action randomness. SAC maximizes reward plus an entropy bonus, encouraging exploration. The entropy coefficient is auto-tuned via a target entropy parameter.

### Episode
One rollout from environment reset to termination or truncation. Episode return is the primary evaluation metric.

### Eval score
The average undiscounted return over evaluation episodes. Reported during training and used to compare algorithm performance.

### GAE (Generalized Advantage Estimation)
A method for computing advantage estimates in PPO that interpolates between high-bias (TD) and high-variance (Monte Carlo) estimates via a parameter lambda. Default lambda is 0.95.

### JIT (Just-In-Time compilation)
JAX compiles Python functions to optimized XLA code for GPU execution. Analogous to `torch.compile` but required for performance. First call is slow (1-3 min); subsequent calls are fast.

### lax.scan
JAX primitive for compiled loops. Replaces Python `for`-loops inside JIT'd functions, since Python control flow cannot be traced. Critical for PPO throughput.

### MJX
The JAX-native MuJoCo physics backend (`impl="jax"`). Used for DM Control benchmarks. Go2 MJX environment is archived in favor of Warp.

### MLP (Multi-Layer Perceptron)
A fully connected neural network. The default architecture for both actor and critic in this framework.

### MuJoCo
A physics simulator widely used in robotics and RL research. This framework uses MuJoCo for all environments.

### MuJoCo Warp
GPU-accelerated physics backend using NVIDIA Warp kernels (`impl="warp"`). The primary backend for Go2 locomotion — higher throughput than MJX for complex scenes.

### Off-policy
Algorithms that store past transitions in a replay buffer and reuse them for training. SAC, TD3, FastSAC, FastTD3, and FlashSAC are off-policy.

### On-policy
Algorithms that use data only from the current policy and discard it after each update. PPO is the on-policy algorithm in this framework.

### Observation (obs)
The input the agent receives each timestep. Go2 environments use dictionary observations with `"state"` (deployable sensors) and `"privileged_state"` (simulator-only information) groups.

### PD gains (Kp, Kd)
Proportional and derivative gains for the joint-level motor controller. Tightly coupled to the physics solver timestep and iteration count — changing one without the other causes instability.

### Policy
The neural network that maps observations to actions. In this framework, policies are Flax `nn.Module` instances compiled with JAX.

### Preset
A pre-configured `(TrainConfig, AlgoConfig)` tuple with proven hyperparameters for a specific environment. See [Environment Presets](reference/env-presets.md).

### Privileged state
Extra simulator information (ground contacts, friction coefficients, body velocities) available to the critic during training but not to the deployed policy.

### Pytree
JAX's generic nested data structure (dicts, lists, tuples of arrays). Used for parameters, optimizer state, training state, and configs.

### Replay buffer
Stores past `(obs, action, reward, next_obs, done)` transitions for off-policy training. Larger buffers allow more data reuse but consume more memory.

### Reward shaping
Designing reward functions to guide the agent toward desired behavior. Locomotion rewards combine tracking terms (velocity, heading) with regularization terms (energy, smoothness).

### Sim-to-real
Transferring a policy trained in simulation to a physical robot. Requires domain randomization, matched PD gains, and careful observation design. See the [Sim-to-Real tutorial](tutorials/sim2real.md).

### Target entropy
SAC's entropy target for automatic temperature tuning. Defaults to `-dim(action_space)` for small-scale training. Set to `0` for large-scale training (1024+ envs).

### Truncation
An episode ending due to a time limit, not a terminal state. The framework bootstraps the value estimate at truncated timesteps, unlike termination where value is zero.

### UTD ratio (Update-to-Data)
Gradient updates per environment step. Higher UTD ratios improve sample efficiency at the cost of wall-clock time. FastSAC/FastTD3 use UTD 8-20.

### vmap
JAX's vectorized map — batches a function across an array dimension. Used to run parallel environments and vectorize network inference without explicit batch dimensions.
