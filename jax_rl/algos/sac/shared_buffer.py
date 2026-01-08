"""
Shared Replay Buffer - Memory Efficient Alternative

Instead of (num_envs, capacity, dim), uses (capacity, dim) shared buffer.
This reduces memory by ~num_envs× while maintaining similar performance.

Key insight: For off-policy RL (SAC), we don't need to track which env
produced each transition - all transitions are valid training data.
"""
import jax
import jax.numpy as jnp
from flax import struct

from .sac import Transition

@struct.dataclass
class SharedBufferState:
    data: Transition
    ptr: jax.Array      # Single pointer (scalar), not per-env
    count: jax.Array    # Single count (scalar), not per-env
    capacity: int

DTYPE_FLOAT = jnp.float32
DTYPE_INT = jnp.int32

class SharedReplayBuffer:
    """
    Shared replay buffer: single buffer for all environments.
    
    Memory: (capacity, dim) instead of (num_envs, capacity, dim)
    Speed: Actually FASTER for sampling (single random op vs two)
    """
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        # Single shared buffer shapes
        self.obs_shape = (capacity, obs_dim)
        self.act_shape = (capacity, act_dim)
        self.scalar_shape = (capacity,)
    
    def init(self):
        return SharedBufferState(
            data=Transition(
                obs=jnp.zeros(self.obs_shape, dtype=DTYPE_FLOAT),
                act=jnp.zeros(self.act_shape, dtype=DTYPE_FLOAT),
                rew=jnp.zeros(self.scalar_shape, dtype=DTYPE_FLOAT),
                next_obs=jnp.zeros(self.obs_shape, dtype=DTYPE_FLOAT),
                done=jnp.zeros(self.scalar_shape, dtype=DTYPE_FLOAT),
            ),
            ptr=jnp.array(0, dtype=DTYPE_INT),
            count=jnp.array(0, dtype=DTYPE_INT),
            capacity=self.capacity
        )
    
    def add(self, state: SharedBufferState, batch: Transition):
        """
        Add batch of transitions to shared buffer.
        
        Args:
            batch: Transition with shapes (num_envs, dim) or (num_envs,)
            state: Current buffer state
        
        Returns:
            Updated buffer state
        
        Note: Uses scatter operation for efficient parallel updates. Handles
        wraparound correctly by splitting updates if needed.
        """
        model_dtype = state.data.obs.dtype
        batch = jax.tree.map(lambda x: x.astype(model_dtype), batch)
        
        num_new = batch.obs.shape[0]  # Number of new transitions
        ptr = state.ptr
        count = state.count
        
        # Calculate indices where to insert (handling wraparound)
        # Use jax.lax.iota instead of jnp.arange to work with traced values
        # iota creates [0, 1, 2, ..., num_new-1] and works in traced contexts
        indices = (jax.lax.iota(DTYPE_INT, num_new) + ptr) % self.capacity
        
        # Use scatter to update buffer at these indices
        # JAX's scatter handles wraparound correctly - if indices wrap around,
        # it will update the correct positions
        def scatter_update(buf, new_vals, idxs):
            # buf: (capacity, ...)
            # new_vals: (num_new, ...)
            # idxs: (num_new,)
            # Result: buf with updates at idxs positions
            # Note: If idxs has duplicates or wraps around, JAX handles it correctly
            return buf.at[idxs].set(new_vals)
        
        new_data = Transition(
            obs=scatter_update(state.data.obs, batch.obs, indices),
            act=scatter_update(state.data.act, batch.act, indices),
            rew=scatter_update(state.data.rew, batch.rew, indices),
            next_obs=scatter_update(state.data.next_obs, batch.next_obs, indices),
            done=scatter_update(state.data.done, batch.done, indices),
        )
        
        new_ptr = (ptr + num_new) % self.capacity
        new_count = jnp.minimum(count + num_new, self.capacity)
        
        return state.replace(data=new_data, ptr=new_ptr, count=new_count)
    
    def sample(self, state: SharedBufferState, rng, batch_size):
        """
        Sample batch_size transitions from shared buffer.
        
        Much simpler than per-env buffer: just random indices!
        
        Note: For circular buffer, we sample uniformly from [0, count).
        When buffer wraps around, count stays at capacity, so we sample
        from the full buffer (which includes both old and new data).
        This is correct behavior for off-policy RL.
        
        Args:
            state: Current buffer state
            rng: JAX random key
            batch_size: Number of transitions to sample
        
        Returns:
            Transition batch with shape (batch_size, ...)
        """
        max_count = state.count
        
        # Safety check: if buffer is empty, return zeros (shouldn't happen after warmup)
        # But we need to handle it for JAX compilation
        def safe_sample():
            # Sample random indices uniformly from [0, max_count)
            # This ensures we only sample from valid buffer positions
            indices = jax.random.randint(rng, (batch_size,), 0, max_count)
            return Transition(
                obs=state.data.obs[indices],
                act=state.data.act[indices],
                rew=state.data.rew[indices],
                next_obs=state.data.next_obs[indices],
                done=state.data.done[indices]
            )
        
        def empty_sample():
            # Return zero-filled batch if buffer is empty
            # This should only happen during warmup or if buffer wasn't initialized
            obs_dim = state.data.obs.shape[-1]
            act_dim = state.data.act.shape[-1]
            return Transition(
                obs=jnp.zeros((batch_size, obs_dim), dtype=state.data.obs.dtype),
                act=jnp.zeros((batch_size, act_dim), dtype=state.data.act.dtype),
                rew=jnp.zeros((batch_size,), dtype=state.data.rew.dtype),
                next_obs=jnp.zeros((batch_size, obs_dim), dtype=state.data.next_obs.dtype),
                done=jnp.zeros((batch_size,), dtype=state.data.done.dtype),
            )
        
        # Use conditional to handle empty buffer case
        # This is necessary for JAX compilation - both branches must be defined
        return jax.lax.cond(max_count > 0, safe_sample, empty_sample)

