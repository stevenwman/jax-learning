# Buffers

Replay and rollout buffers for off-policy and on-policy algorithms. `JaxReplayBuffer` stores transitions for SAC/TD3; `RolloutBuffer` collects trajectories for PPO with GAE computation.

::: jax_rl.buffers.jax_replay_buffer.JaxReplayBuffer
    options:
      filters: ["!__init__"]

::: jax_rl.buffers.jax_replay_buffer.FrameStackConfig

::: jax_rl.buffers.rollout.RolloutBuffer
    options:
      filters: ["!__init__"]

::: jax_rl.buffers.rollout.RolloutBatch

::: jax_rl.buffers.rollout.compute_gae
