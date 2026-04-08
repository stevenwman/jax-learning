<!-- JAX -->
*[JIT]: Just-In-Time compilation — JAX compiles Python to optimized GPU code. First call is slow; subsequent calls are fast.
*[vmap]: JAX's vectorized map — batches a function across an array dimension without explicit loops.
*[lax.scan]: JAX primitive for compiled loops — replaces Python for-loops inside JIT'd functions.
*[pytree]: JAX's generic nested data structure (dicts, lists, tuples of arrays).
*[XLA]: Accelerated Linear Algebra — the compiler backend JAX uses for GPU/TPU execution.

<!-- RL fundamentals -->
*[GAE]: Generalized Advantage Estimation — interpolates between high-bias and high-variance advantage estimates via lambda.
*[UTD]: Update-to-Data ratio — gradient updates per environment step. Higher UTD = better sample efficiency.
*[MLP]: Multi-Layer Perceptron — a fully connected neural network.
*[PPO]: Proximal Policy Optimization — on-policy RL algorithm with clipped surrogate objective.
*[SAC]: Soft Actor-Critic — off-policy RL with auto-tuned entropy regularization.
*[TD3]: Twin Delayed DDPG — off-policy RL with twin critics and delayed actor updates.

<!-- Architecture -->
*[C51]: Distributional RL algorithm representing Q-values as a categorical distribution over atoms.
*[BatchNorm]: Batch Normalization — normalizes layer inputs across the batch for training stability.

<!-- Environments & hardware -->
*[MJCF]: MuJoCo's XML scene description format.
*[MJX]: JAX-native MuJoCo physics backend (impl="jax").
*[DR]: Domain Randomization — randomizing physics parameters during training for sim-to-real transfer.
*[PD gains]: Proportional-derivative motor controller stiffness (Kp) and damping (Kd).
*[ONNX]: Open Neural Network Exchange — a portable model format (not yet implemented in this framework).
*[VRAM]: Video RAM — GPU memory available for training.

<!-- Tools -->
*[W&B]: Weights & Biases — experiment tracking and visualization platform.
