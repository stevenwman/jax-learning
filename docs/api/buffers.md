# Buffers

Replay and rollout buffers for off-policy and on-policy algorithms. `JaxReplayBuffer` stores transitions for SAC/TD3; `RolloutBuffer` collects trajectories for PPO with GAE computation.

## JaxReplayBuffer

::: jax_rl.buffers.jax_replay_buffer.JaxReplayBuffer

## FrameStackConfig

::: jax_rl.buffers.jax_replay_buffer.FrameStackConfig

## RolloutBuffer

::: jax_rl.buffers.rollout.RolloutBuffer

## RolloutBatch

::: jax_rl.buffers.rollout.RolloutBatch

## compute_gae

::: jax_rl.buffers.rollout.compute_gae
