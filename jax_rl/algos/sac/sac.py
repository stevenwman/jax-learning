import jax
import jax.numpy as jnp
from flax import struct

# --- Data Structures ---
@struct.dataclass
class Transition:
    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    next_obs: jax.Array
    done: jax.Array 

@struct.dataclass
class BufferState:
    data: Transition
    ptr: jax.Array      # Shape (NumEnvs,)
    count: jax.Array    # Shape (NumEnvs,)
    capacity: int

DTYPE_FLOAT = jnp.float32
DTYPE_INT = jnp.int32

# --- The Logic ---
class BatchedReplayBuffer:
    def __init__(self, num_envs, capacity, obs_dim, act_dim):
        self.num_envs = num_envs
        self.capacity = capacity
        # We pre-allocate the memory on GPU
        self.obs_shape = (num_envs, capacity, obs_dim)
        self.act_shape = (num_envs, capacity, act_dim)
        self.scalar_shape = (num_envs, capacity)

    def init(self):
        return BufferState(
            data=Transition(
                obs=jnp.zeros(self.obs_shape,dtype=DTYPE_FLOAT),
                act=jnp.zeros(self.act_shape,dtype=DTYPE_FLOAT),
                rew=jnp.zeros(self.scalar_shape,dtype=DTYPE_FLOAT),
                next_obs=jnp.zeros(self.obs_shape,dtype=DTYPE_FLOAT),
                done=jnp.zeros(self.scalar_shape,dtype=DTYPE_FLOAT),
            ),
            ptr=jnp.zeros(self.num_envs, dtype=DTYPE_INT),
            count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            capacity=self.capacity
        )

    def add(self, state: BufferState, batch: Transition):

        model_dtype = state.data.obs.dtype
        batch = jax.tree.map(lambda x: x.astype(model_dtype), batch)
        # 1. Expand data to broadcast correctly: (NumEnvs, 1, Dim)
        # '...' detects remaining dims, making this part universal
        def fix(x): return x[:, None, ...] if x.ndim > 1 else x[:, None]
        
        # 2. Use vmap to insert into each env's circular buffer
        # This is the "Independent Buffer" logic        
        def insert_slice(buf, update, p):
            return jax.lax.dynamic_update_slice_in_dim(buf, update, start_index=p, axis=0)

        # We vmap over the "Environment" axis (0)
        new_data = Transition(
            obs      = jax.vmap(insert_slice)(state.data.obs     , fix(batch.obs)     , state.ptr),
            act      = jax.vmap(insert_slice)(state.data.act     , fix(batch.act)     , state.ptr),
            rew      = jax.vmap(insert_slice)(state.data.rew     , fix(batch.rew)     , state.ptr),
            next_obs = jax.vmap(insert_slice)(state.data.next_obs, fix(batch.next_obs), state.ptr),
            done     = jax.vmap(insert_slice)(state.data.done    , fix(batch.done)    , state.ptr)
        )

        new_ptr = (state.ptr + 1) % self.capacity
        new_count = jnp.minimum(state.count + 1, self.capacity)
        return state.replace(data=new_data, ptr=new_ptr, count=new_count)

    def sample(self, state: BufferState, rng, batch_size):
        k1, k2 = jax.random.split(rng)
        
        # 1. Randomly pick WHICH environments to sample from
        # Shape: (batch_size,)
        env_indices = jax.random.randint(k1, (batch_size,), 0, self.num_envs)
        
        # 2. Lookup the SPECIFIC count for each chosen environment
        # We use the indices from step 1 to grab the correct 'count' for that env.
        # Shape: (batch_size,) e.g. [20, 5, 100, 20, ...]
        batch_counts = state.count[env_indices]
        
        # 3. Sample time indices using the SPECIFIC bounds
        # jax.random.randint supports "Broadcasting". 
        # Since 'maxval' (batch_counts) matches the requested shape (batch_size,),
        # JAX will automatically use batch_counts[i] as the limit for item i.
        time_indices = jax.random.randint(k2, (batch_size,), 0, batch_counts)

        # 4. Gather the data
        def get(arr): return arr[env_indices, time_indices]
        
        return Transition(
            obs      = get(state.data.obs),
            act      = get(state.data.act),
            rew      = get(state.data.rew),
            next_obs = get(state.data.next_obs),
            done     = get(state.data.done)
        )

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from .core import Actor, Critic

# --- 1. The Actor Loss (Policy Optimization) ---
def actor_loss_fn(actor: Actor, critic: Critic, batch: Transition, alpha, rng):
    # 1. Sample actions from the CURRENT policy
    # We need a key here because the policy is stochastic (Gaussian)
    dist = actor(batch.obs)
    pi_action, log_prob = dist.sample_and_log_prob(seed=rng)
    
    # 2. Get Q-values for these NEW actions
    # Note: We use the 'critic' (not target) to grade the actor
    q1, q2 = critic(batch.obs, pi_action)
    min_q = jnp.minimum(q1, q2)
    
    # 3. SAC Objective: Maximize (MinQ - alpha * LogProb)
    # Since we are minimizing loss, we flip the signs:
    # Minimize: alpha * LogProb - MinQ
    loss = (alpha * log_prob - min_q).mean()
    
    # Diagnostics: Check for NaN/inf at each step
    has_nan = jnp.isnan(loss) | jnp.isinf(loss)
    has_nan_logprob = jnp.isnan(log_prob).any() | jnp.isinf(log_prob).any()
    has_nan_q = jnp.isnan(min_q).any() | jnp.isinf(min_q).any()
    has_nan_action = jnp.isnan(pi_action).any() | jnp.isinf(pi_action).any()
    has_nan_obs = jnp.isnan(batch.obs).any() | jnp.isinf(batch.obs).any()
    
    # Return diagnostics along with loss
    diagnostics = {
        "has_nan_loss": has_nan,
        "has_nan_logprob": has_nan_logprob,
        "has_nan_q": has_nan_q,
        "has_nan_action": has_nan_action,
        "has_nan_obs": has_nan_obs,
        "log_prob_min": jnp.min(log_prob),
        "log_prob_max": jnp.max(log_prob),
        "log_prob_mean": jnp.mean(log_prob),
        "q_min": jnp.min(min_q),
        "q_max": jnp.max(min_q),
        "q_mean": jnp.mean(min_q),
    }
    
    return loss, (-log_prob.mean(), diagnostics) # Return entropy and diagnostics for logging

# --- 2. The Critic Loss (Q-Learning) ---
def critic_loss_fn(critic: Critic, target_critic: Critic, actor: Actor, batch: Transition, alpha, rng):
    
    # 1. Generate Target Actions (from Next State)
    # We don't want gradients flowing through the target generation!
    dist = actor(batch.next_obs)
    next_action, next_log_prob = dist.sample_and_log_prob(seed=rng)
    
    # 2. Target Q-Values (Double Q-Learning)
    # Use target_critic weights
    target_q1, target_q2 = target_critic(batch.next_obs, next_action)
    target_min_q = jnp.minimum(target_q1, target_q2)
    
    # 3. The Bellman Backup (Soft Q)
    # Target = R + gamma * (1 - D) * (TargetQ - alpha * TargetEntropy)
    gamma = 0.99
    target_y = batch.rew + gamma * (1 - batch.done) * (target_min_q - alpha * next_log_prob)
    
    # 4. Current Q-Values
    # We execute both heads on the current batch
    current_q1, current_q2 = critic(batch.obs, batch.act)
    
    # 5. MSE Loss
    diff1 = current_q1 - target_y
    diff2 = current_q2 - target_y
    loss_q1 = (diff1 ** 2).mean()
    loss_q2 = (diff2 ** 2).mean()
    
    total_loss = loss_q1 + loss_q2
    
    # Diagnostics: Check for NaN/inf at each step
    has_nan_loss = jnp.isnan(total_loss) | jnp.isinf(total_loss)
    has_nan_target = jnp.isnan(target_y).any() | jnp.isinf(target_y).any()
    has_nan_target_q = jnp.isnan(target_min_q).any() | jnp.isinf(target_min_q).any()
    has_nan_current_q = jnp.isnan(current_q1).any() | jnp.isnan(current_q2).any()
    has_nan_next_logprob = jnp.isnan(next_log_prob).any() | jnp.isinf(next_log_prob).any()
    has_nan_reward = jnp.isnan(batch.rew).any() | jnp.isinf(batch.rew).any()
    has_nan_next_obs = jnp.isnan(batch.next_obs).any() | jnp.isinf(batch.next_obs).any()
    
    diagnostics = {
        "has_nan_loss": has_nan_loss,
        "has_nan_target": has_nan_target,
        "has_nan_target_q": has_nan_target_q,
        "has_nan_current_q": has_nan_current_q,
        "has_nan_next_logprob": has_nan_next_logprob,
        "has_nan_reward": has_nan_reward,
        "has_nan_next_obs": has_nan_next_obs,
        "target_min": jnp.min(target_y),
        "target_max": jnp.max(target_y),
        "target_mean": jnp.mean(target_y),
        "target_q_min": jnp.min(target_min_q),
        "target_q_max": jnp.max(target_min_q),
        "target_q_mean": jnp.mean(target_min_q),
        "next_logprob_min": jnp.min(next_log_prob),
        "next_logprob_max": jnp.max(next_log_prob),
        "reward_min": jnp.min(batch.rew),
        "reward_max": jnp.max(batch.rew),
    }
    
    return total_loss, (jnp.mean(target_min_q), diagnostics) # Log Q-vals and diagnostics

# --- 3. The Update Step (Combining Everything) ---
# This is the function that runs inside the Scan Loop
def train_step(
    actor: Actor, critic: Critic, target_critic: Critic,       # The Models
    actor_opt: nnx.Optimizer, critic_opt: nnx.Optimizer,       # The Optimizers
    batch: Transition,                                         # The Data
    key,                                                       # The Master Key
    alpha=0.2,                                                 # Fixed Entropy Temp (Simplified)
    polyak=0.995,                                              # Target Update Rate
):
    # A. Split Keys for the two stochastic operations
    # key -> (next_key, key_for_actor_loss, key_for_critic_loss)
    next_key, k1, k2 = jax.random.split(key, 3)
    
    # B. Update Critic
    # nnx.value_and_grad gives us both the loss value (for logs) and gradients
    (c_loss, (c_log, c_diag)), c_grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(
        critic, target_critic, actor, batch, alpha, k1
    )
    
    # Check for NaN in gradients
    def check_grad_nan(g):
        return jnp.isnan(g).any() | jnp.isinf(g).any()
    c_grads_has_nan = jax.tree.map(check_grad_nan, c_grads)
    c_grads_has_nan_any = jnp.array(list(jax.tree.leaves(c_grads_has_nan))).any()
    
    critic_opt.update(critic, c_grads)

    # C. Update Actor
    # Note: Actor update depends on Critic, so we do it second
    (a_loss, (entropy, a_diag)), a_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(
        actor, critic, batch, alpha, k2
    )
    
    # Check for NaN in gradients
    a_grads_has_nan = jax.tree.map(check_grad_nan, a_grads)
    a_grads_has_nan_any = jnp.array(list(jax.tree.leaves(a_grads_has_nan))).any()
    
    actor_opt.update(actor, a_grads)
    
    # D. Update Target Networks (Polyak Averaging)
    # Standard JAX tree map: new_target = polyak * target + (1-polyak) * source
    # We access the parameters via 'nnx.state(model, nnx.Param)'
    
    # Helper to smooth weights
    def soft_update(target_node, source_node):
        return polyak * target_node + (1.0 - polyak) * source_node
        
    # We update the state of target_critic IN PLACE (conceptually)
    # nnx.update performs the replacement safely
    current_params = nnx.state(critic, nnx.Param)
    target_params = nnx.state(target_critic, nnx.Param)
    
    new_target_params = jax.tree.map(soft_update, target_params, current_params)
    nnx.update(target_critic, new_target_params)

    # E. Return Logs and Key with diagnostics
    metrics = {
        "loss_critic": c_loss,
        "loss_actor": a_loss,
        "q_val": c_log,
        "entropy": entropy,
        # Critic diagnostics
        **{f"critic_{k}": v for k, v in c_diag.items()},
        # Actor diagnostics
        **{f"actor_{k}": v for k, v in a_diag.items()},
        # Gradient diagnostics (aggregate)
        "critic_grads_has_nan": c_grads_has_nan_any,
        "actor_grads_has_nan": a_grads_has_nan_any,
    }
    return next_key, metrics