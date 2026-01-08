import os
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import wandb
from collections import namedtuple
from datetime import datetime

# Playground Imports
from mujoco_playground import registry
from brax.envs import Wrapper, State

from algos.sac.core import Actor, Critic, create_checkpoint_manager, save_ckpt
from algos.sac.sac import train_step, Transition
from algos.sac.shared_buffer import SharedReplayBuffer

# --- 1. CONFIGURATION ---
jax.config.update("jax_enable_x64", True)
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'float32'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Helps avoid memory issues with large batches

# ENV_NAME = 'Go1JoystickFlatTerrain'
ENV_NAME = 'HumanoidRun'
# ENV_NAME = 'CartpoleSwingup'
NUM_ENVS = 2048
# NUM_ENVS = 10
TOTAL_STEPS = 1_000_000
BATCH_SIZE = 256
HIDDEN_SIZES = (256, 256)
LR = 3e-4
ALPHA = 0.2
WARMUP_STEPS = 1000
LOG_EVERY = 10
BUFFER_CAPACITY = 500_000  # Adjust as needed; shared buffer keeps memory reasonable
STEPS_PER_EPOCH = 5000

# --- REWARD SCALING ---
# Some environments need reward scaling to prevent Q-value explosion during training.
# Set to 1.0 to disable scaling, or adjust per environment as needed.
# Common values: 0.1 for HumanoidRun/Go1, 1.0 for CartpoleSwingup
# CRITICAL: Use environment-specific scaling to avoid distribution shift
# REWARD_SCALE = 0.1 if ENV_NAME in ['HumanoidRun', 'Go1JoystickFlatTerrain'] else 1.0
REWARD_SCALE = 1.0

# --- 2. ADAPTERS & HELPERS ---
_PseudoStateBase = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

class PseudoState(_PseudoStateBase):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

class BraxAdapter(Wrapper):
    def reset(self, rng):
        state = self.env.reset(rng)
        return self._to_brax_state(state)

    def step(self, state, action):
        mock_state = PseudoState(
            data=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=state.info
        )
        next_state = self.env.step(mock_state, action)
        return self._to_brax_state(next_state)

    def _to_brax_state(self, play_state):
        return State(
            pipeline_state=play_state.data,
            obs=play_state.obs,
            reward=play_state.reward,
            done=play_state.done,
            metrics=play_state.metrics,
            info=play_state.info
        )

# --- 3. THE CONTAINER (To keep scan clean) ---
# We use this to bundle everything that changes during training
class AgentState(nnx.Variable):
    # This is just a dummy class to hold references if we wanted
    # But simpler is just a Python Dataclass or Tuple.
    # Let's use a simple dictionary-like structure for JAX to carry.
    pass


def scan_loop(step_fn, models, init_carry, length):
    """
    Wraps jax.lax.scan to handle Flax NNX mutable objects automatically.
    
    Args:
        step_fn: Function(models, carry, _) -> (new_carry, outputs)
        models: A tuple/list/dict of NNX objects (Actor, Critic, Optimizers)
        init_carry: Your custom state (EnvState, BufferState, Key)
        length: Number of steps to run
    """
    # 1. SPLIT: Separate structure (Graph) from weights (State)
    graphdef, init_state = nnx.split(models)
    
    # Pack weights into the scan carry
    scan_carry = (init_state, init_carry)

    def wrapped_step(curr_scan_carry, _):
        state, user_carry = curr_scan_carry
        
        # 2. MERGE: Reconstruct mutable models inside the loop
        # We use the STATIC graphdef from outside + DYNAMIC state from carry
        current_models = nnx.merge(graphdef, state)
        
        # 3. RUN USER LOGIC
        # The user function modifies 'current_models' in-place
        new_user_carry, outputs = step_fn(current_models, user_carry)
        
        # 4. SPLIT AGAIN: Extract updated weights to pass to next step
        _, new_state = nnx.split(current_models)
        
        return (new_state, new_user_carry), outputs

    # Run the JAX primitive
    (final_state, final_user_carry), stacked_outputs = jax.lax.scan(
        wrapped_step, scan_carry, None, length=length
    )
    
    # 5. UPDATE: Update the original model objects with final weights
    # This ensures the objects outside this function are up-to-date
    nnx.update(models, final_state)
    
    return final_user_carry, stacked_outputs

# --- 4. MAIN LOOP ---
def main():

    wandb.init(
        project="jax-sac-playground",
        config={
            "env": ENV_NAME,
            "num_envs": NUM_ENVS,
            "total_steps": TOTAL_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "hidden_sizes": HIDDEN_SIZES
        }
    )

    # Create unique checkpoint directory with timestamp and environment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"./sac_checkpoints/{ENV_NAME}_{timestamp}"
    ckpt_manager = create_checkpoint_manager(ckpt_dir)
    print(f"Checkpoint directory: {os.path.abspath(ckpt_dir)}")

    print(f"Loading Playground Env: {ENV_NAME}")
    print(f"Reward Scaling: {REWARD_SCALE} {'(DISABLED)' if REWARD_SCALE == 1.0 else '(ENABLED)'}")
    base_env = registry.load(ENV_NAME)
    env = BraxAdapter(base_env)
    
    # Create both JIT and non-JIT versions
    # Non-JIT for use inside scan (scan compiles everything, so nested JIT causes issues)
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step)
    
    # JIT versions for one-off calls (e.g., initial reset)
    jit_reset = jax.jit(vmap_reset)
    jit_step = jax.jit(vmap_step)
    
    # Handle dictionary observation spaces
    raw_obs_size = env.observation_size
    use_dict_obs = isinstance(raw_obs_size, dict)
    
    if use_dict_obs:
        # For dict obs spaces, use 'state' key (or first key)
        # Go1JoystickFlatTerrain has {'state': (48,), 'privileged_state': (123,)}
        obs_key = 'state' if 'state' in raw_obs_size else list(raw_obs_size.keys())[0]
        obs_shape = raw_obs_size[obs_key]
        obs_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        print(f"Detected dict observation space: {raw_obs_size}")
        print(f"Using '{obs_key}' key with dim={obs_dim}")
    else:
        obs_dim = raw_obs_size
        obs_key = None
        print(f"Using flat observation space with dim={obs_dim}")
    
    # Helper to extract observations - must be JAX-traceable
    # After vmap, obs is either Array(N, obs_dim) or dict of Arrays
    def flatten_obs(obs):
        if use_dict_obs:
            # For dict obs, extract the key we want
            # This works because JAX can trace dict access
            return obs[obs_key]
        else:
            # For flat obs, return as-is
            return obs
    
    act_dim = env.action_size
    print(f"Obs dim: {obs_dim} | Act dim: {act_dim} | Envs: {NUM_ENVS}")

    # --- INITIALIZATION ---
    key = nnx.Rngs(0)
    actor = Actor(obs_dim, act_dim, HIDDEN_SIZES, rngs=key)
    critic = Critic(obs_dim, act_dim, HIDDEN_SIZES, rngs=key)
    target_critic = Critic(obs_dim, act_dim, HIDDEN_SIZES, rngs=nnx.Rngs(0))

    actor_opt = nnx.Optimizer(actor, optax.adam(LR), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, optax.adam(LR), wrt=nnx.Param)

    # Use shared replay buffer to reduce memory usage dramatically
    assert BUFFER_CAPACITY >= NUM_ENVS, "BUFFER_CAPACITY must be >= NUM_ENVS to avoid index collisions per step."
    buffer = SharedReplayBuffer(capacity=BUFFER_CAPACITY, obs_dim=obs_dim, act_dim=act_dim)
    buffer_state = buffer.init()

    models = (actor, critic, target_critic, actor_opt, critic_opt)

    def rollout_step(curr_models, carry):
        # 1. Unpack
        (actor, critic, target_critic, actor_opt, critic_opt) = curr_models
        env_state, buf_state, key = carry
        
        key, act_key, train_key = jax.random.split(key, 3)
        
        # 2. Action & Step (flatten observations if needed)
        # After vmap, env_state.obs is either Array(N, obs_dim) or dict of Arrays
        flat_obs = flatten_obs(env_state.obs)
        action = jax.vmap(actor.get_stochastic_action)(flat_obs, jax.random.split(act_key, NUM_ENVS))
        # Use vmap_step (not jit_step) inside scan_loop - scan_loop compiles everything
        next_env_state = vmap_step(env_state, action)
        
        # 3. Buffer Add (flatten observations)
        flat_next_obs = flatten_obs(next_env_state.obs)
        trans = Transition(
            obs=flat_obs, act=action, rew=next_env_state.reward, 
            next_obs=flat_next_obs, done=next_env_state.done
        )
        buf_state = buffer.add(buf_state, trans)
        
        # 4. Train
        batch = buffer.sample(buf_state, train_key, BATCH_SIZE)
        
        # Check batch for NaN/inf before training
        batch_has_nan_obs = jnp.isnan(batch.obs).any() | jnp.isinf(batch.obs).any()
        batch_has_nan_act = jnp.isnan(batch.act).any() | jnp.isinf(batch.act).any()
        batch_has_nan_rew = jnp.isnan(batch.rew).any() | jnp.isinf(batch.rew).any()
        batch_has_nan_next_obs = jnp.isnan(batch.next_obs).any() | jnp.isinf(batch.next_obs).any()
        
        # Debug: Check if buffer is actually storing data (after warmup)
        # This helps verify the shared buffer is working correctly
        buffer_count = buf_state.count
        batch_is_zero = (jnp.abs(batch.obs).max() < 1e-6) & (buffer_count > 1000)  # Only check after warmup
        
        # --- REWARD SCALING ---
        # Apply reward scaling (configured at top of file)
        batch_scaled = batch.replace(rew=batch.rew * REWARD_SCALE)
        
        _, metrics = train_step(
            actor, critic, target_critic, actor_opt, critic_opt, 
            batch_scaled, train_key, ALPHA
        )
        
        # Add batch-level diagnostics
        metrics["batch_has_nan_obs"] = batch_has_nan_obs
        metrics["batch_has_nan_act"] = batch_has_nan_act
        metrics["batch_has_nan_rew"] = batch_has_nan_rew
        metrics["batch_has_nan_next_obs"] = batch_has_nan_next_obs
        metrics["buffer_count"] = buffer_count
        metrics["batch_is_zero"] = batch_is_zero  # Warning flag if buffer seems empty
        # Log rewards: raw (before scaling) and check if they're non-zero
        raw_reward_mean = next_env_state.reward.mean()
        raw_reward_sum = next_env_state.reward.sum()
        metrics["per_step_reward"] = raw_reward_mean
        metrics["per_step_reward_sum"] = raw_reward_sum
        metrics["per_step_reward_max"] = next_env_state.reward.max()
        metrics["per_step_reward_min"] = next_env_state.reward.min()
        
        return (next_env_state, buf_state, key), metrics

    # --- 6. EXECUTION ---
    print("Initializing state...")
    master_key = jax.random.PRNGKey(42)
    master_key, reset_key = jax.random.split(master_key)
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    env_state = jit_reset(reset_keys)
    
    # Verify observation shape
    test_obs = flatten_obs(env_state.obs)
    expected_shape = (NUM_ENVS, obs_dim)
    if test_obs.shape != expected_shape:
        raise ValueError(f"Observation shape mismatch! Got {test_obs.shape}, expected {expected_shape}")
    print(f"✓ Observation shape verified: {test_obs.shape}")

    # Warmup
    print("Warmup...")
    def warmup_fn(carry, _):
        es, bs, k = carry
        k, ak = jax.random.split(k)
        action = jax.random.uniform(ak, (NUM_ENVS, act_dim), minval=-1, maxval=1)
        # Use vmap_step (not jit_step) inside scan - scan compiles everything
        nes = vmap_step(es, action)
        # Flatten observations for buffer
        flat_obs = flatten_obs(es.obs)
        flat_next_obs = flatten_obs(nes.obs)
        trans = Transition(flat_obs, action, nes.reward, flat_next_obs, nes.done)
        bs = buffer.add(bs, trans)
        return (nes, bs, k), None
        
    (env_state, buffer_state, master_key), _ = jax.lax.scan(
        warmup_fn, (env_state, buffer_state, master_key), None, length=WARMUP_STEPS
    )

    # Evaluation function: run deterministic episodes to measure policy performance
    def evaluate_policy(actor, num_episodes=10, max_steps=1000, eval_seed_offset=0):
        """
        Evaluate policy using deterministic actions. Returns average episode return.
        
        Args:
            eval_seed_offset: Offset to add to base seed (999) to get different eval seeds per epoch
                             This helps reduce variance by using different initial states each time
        """
        # Use epoch-dependent seed to reduce variance while still being deterministic
        base_seed = 999 + eval_seed_offset
        eval_key = jax.random.PRNGKey(base_seed)
        eval_key, reset_key = jax.random.split(eval_key)
        reset_keys = jax.random.split(reset_key, num_episodes)
        
        # Create evaluation environments
        eval_env_state = jit_reset(reset_keys)
        
        def eval_step(carry, _):
            eval_es, k = carry
            
            # Use deterministic actions for evaluation
            flat_obs = flatten_obs(eval_es.obs)
            action = jax.vmap(actor.get_deterministic_action)(flat_obs)
            next_es = vmap_step(eval_es, action)
            
            # Return rewards and done flags - we'll process these later
            return (next_es, k), (next_es.reward, next_es.done.astype(bool))
        
        # Run evaluation and collect all rewards and done flags
        (final_es, _), (all_rewards, all_dones) = jax.lax.scan(
            eval_step, (eval_env_state, eval_key), 
            None, length=max_steps
        )
        # all_rewards shape: (max_steps, num_episodes)
        # all_dones shape: (max_steps, num_episodes)
        
        # Post-process: compute episode returns from collected rewards and done flags
        # Cumulative sum of rewards over time: shape (max_steps, num_episodes)
        cumsum_rewards = jnp.cumsum(all_rewards, axis=0)
        
        # Find first time each episode completes (first True in done array for each episode)
        # argmax returns first occurrence (True=1 > False=0), but if all False, it returns 0 (first index)
        done_indices = jnp.argmax(all_dones.astype(jnp.int32), axis=0)  # Shape: (num_episodes,)
        episodes_completed = jnp.any(all_dones, axis=0)  # Shape: (num_episodes,) - True if episode completed
        
        # For completed episodes: done_indices is the step (0-indexed) where it first completed
        # For non-completed episodes: done_indices will be 0 (because argmax on all False returns 0)
        # But we check episodes_completed to distinguish - if False, episode never completed
        
        # Episode lengths: completed = done_index + 1 (include completion step), non-completed = max_steps
        episode_lengths = jnp.where(
            episodes_completed, 
            done_indices.astype(jnp.float32) + 1.0,  # +1 because we include the completion step
            float(max_steps)
        )
        
        # Get cumulative reward at completion time (or final reward if never completed)
        # For completed: use cumsum at done_index (this includes the reward from the completion step)
        # For non-completed: use cumsum at max_steps - 1 (final step, index 999)
        step_indices = jnp.where(episodes_completed, done_indices, max_steps - 1)
        
        # Use vmap or explicit indexing to get rewards at the right indices
        # This is equivalent to: episode_returns[ep_idx] = cumsum_rewards[step_indices[ep_idx], ep_idx]
        def get_return_at_index(ep_idx):
            step_idx = step_indices[ep_idx]
            return cumsum_rewards[step_idx, ep_idx]
        
        episode_returns = jax.vmap(get_return_at_index)(jnp.arange(num_episodes))
        
        # Average return over ALL episodes
        avg_return = jnp.mean(episode_returns)
        avg_length = jnp.mean(episode_lengths)
        
        # Count completed episodes (those that had at least one done=True)
        num_completed = jnp.sum(episodes_completed.astype(jnp.int32))
        
        return float(avg_return), int(num_completed), float(avg_length)

    # Train
    print("Training...")
    num_epochs = TOTAL_STEPS // STEPS_PER_EPOCH
    
    # We construct the MEGA CARRY tuple
    carry = (env_state, buffer_state, master_key)

    for epoch in range(num_epochs):

        carry, metrics_history = scan_loop(
            rollout_step, models, carry, length=STEPS_PER_EPOCH
        )
        
        # <--- 4. AGGREGATE & LOG
        # We average the 1000 steps to get one data point for WandB
        # jax.tree.map applies jnp.mean to every item in the dict
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), metrics_history)
        
        # Convert JAX arrays to standard Python types for WandB
        log_dict = {}
        for k, v in avg_metrics.items():
            try:
                if isinstance(v, (jnp.ndarray, jnp.generic)):
                    # Convert scalar arrays to Python floats
                    if v.ndim == 0:
                        # Handle boolean scalars from diagnostics
                        if v.dtype == jnp.bool_:
                            val = bool(v)
                        else:
                            val = float(v) if jnp.isfinite(v) else 0.0
                    else:
                        # Average for non-scalar arrays, but check for NaN first
                        finite_v = v[jnp.isfinite(v)]
                        val = float(jnp.mean(finite_v)) if finite_v.size > 0 else 0.0
                elif isinstance(v, (list, tuple)):
                    # Handle lists/tuples (e.g., gradient diagnostics)
                    val = []
                    for x in v:
                        if isinstance(x, (jnp.ndarray, jnp.generic)):
                            if x.ndim == 0:
                                val.append(float(x) if jnp.isfinite(x) else 0.0)
                            else:
                                val.append(float(jnp.mean(x)) if jnp.isfinite(x).any() else 0.0)
                        else:
                            val.append(x)
                else:
                    val = v
                log_dict[k] = val
            except Exception as e:
                # If conversion fails, log as 0.0 to avoid breaking training
                print(f"Warning: Failed to convert {k}: {e}")
                log_dict[k] = 0.0
        log_dict["step"] = (epoch + 1) * STEPS_PER_EPOCH
        
        # Check for NaN warnings and print them (only for actual NaN detections, not conversion artifacts)
        nan_warnings = []
        for key, val in log_dict.items():
            if "has_nan" in key.lower():
                # Boolean True means NaN was detected
                if isinstance(val, bool) and val:
                    nan_warnings.append(key)
                # Float > 0.5 means averaged boolean was mostly True
                elif isinstance(val, float) and val > 0.5:
                    nan_warnings.append(f"{key}={val:.2%}")
                # Non-zero int means some NaNs were detected
                elif isinstance(val, (int, float)) and val > 0:
                    nan_warnings.append(f"{key}={val}")
        
        if nan_warnings:
            print(f"⚠️ Epoch {epoch}: NaN detected! Check: {', '.join(nan_warnings[:10])}")
        
        # Check for buffer issues
        buffer_count = log_dict.get('buffer_count', 0)
        batch_is_zero = log_dict.get('batch_is_zero', False)
        if batch_is_zero and buffer_count > 1000:
            print(f"⚠️ Epoch {epoch}: WARNING - Sampled batch appears to be all zeros despite buffer_count={buffer_count:.0f}")
            print(f"   This suggests the buffer may not be storing data correctly!")
        
        # Evaluate policy performance (deterministic actions)
        # Extract actor from models tuple - scan_loop updates models in-place via nnx.update
        env_state_epoch, buf_state_epoch, master_key_epoch = carry
        # Models are updated in-place by scan_loop, so actor is already the latest
        eval_actor, _, _, _, _ = models
        
        # Use epoch number as seed offset to get different eval trajectories each time
        # This reduces variance while still being deterministic
        avg_episode_return, num_episodes_completed, avg_episode_length = evaluate_policy(
            eval_actor, num_episodes=10, max_steps=1000, eval_seed_offset=epoch
        )
        log_dict["eval_episode_return"] = avg_episode_return
        log_dict["eval_episodes_completed"] = num_episodes_completed
        log_dict["eval_episode_length"] = avg_episode_length
        
        wandb.log(log_dict)
        
        per_step_reward = log_dict.get('per_step_reward', 0.0)
        per_step_reward_max = log_dict.get('per_step_reward_max', 0.0)
        per_step_reward_min = log_dict.get('per_step_reward_min', 0.0)
        print(f"Epoch {epoch}: Eval Return={avg_episode_return:.2f} ({num_episodes_completed}/{10} eps, len={avg_episode_length:.1f}), "
              f"Train Reward=[{per_step_reward_min:.4f}, {per_step_reward:.4f}, {per_step_reward_max:.4f}], "
              f"Q={log_dict['q_val']:.2f}, Loss=[C:{log_dict['loss_critic']:.4f}, A:{log_dict['loss_actor']:.4f}]")
    
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            print("logging")
            
            # 1. Get the state Pytree (Using the cleaner alternative from your screenshot)
            state_payload = {
                "actor": nnx.state(actor),
                "critic": nnx.state(critic),
                "target_critic": nnx.state(target_critic),
                "actor_opt": nnx.state(actor_opt),
                "critic_opt": nnx.state(critic_opt),
            }
            
            # 2. Pass strict State object to the saver
            save_ckpt(ckpt_manager, (epoch + 1) * STEPS_PER_EPOCH, state_payload)

    # Final Save
    final_payload = {
        "actor": nnx.state(actor),
        "critic": nnx.state(critic),
        "target_critic": nnx.state(target_critic),
        "actor_opt": nnx.state(actor_opt),
        "critic_opt": nnx.state(critic_opt),
    }
    save_ckpt(ckpt_manager, TOTAL_STEPS, final_payload)

if __name__ == "__main__":
    main()