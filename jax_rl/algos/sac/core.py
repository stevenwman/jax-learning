import jax
import jax.numpy as jnp
from flax import nnx
import distrax
import os
from typing import Sequence, Callable
import orbax.checkpoint as ocp

def create_checkpoint_manager(ckpt_dir, max_to_keep=None):
    """
    Create checkpoint manager - using explicit checkpointer for compatibility.
    
    Args:
        ckpt_dir: Directory path for checkpoints
        max_to_keep: Maximum number of checkpoints to keep. None = keep all checkpoints.
    """
    abs_path = os.path.abspath(ckpt_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    # Use explicit checkpointer to avoid "Unknown key default" error
    # The deprecation warning is just informational - this API still works
    checkpointer = ocp.StandardCheckpointer()
    manager = ocp.CheckpointManager(abs_path, checkpointer, options=options)
    if max_to_keep is None:
        print(f"Checkpoint Manager ready at: {abs_path} (keeping all checkpoints)")
    else:
        print(f"Checkpoint Manager ready at: {abs_path} (keeping {max_to_keep} most recent checkpoints)")
    return manager

def save_ckpt(manager, step, state_to_save):
    """
    Save checkpoint using the simple API (matches notebook version).
    state_to_save should be a dict with keys like 'actor', 'critic', etc.
    """
    print(f"Saving checkpoint step {step}...")
    try:
        # Get the actual directory path for debugging
        ckpt_dir = os.path.abspath(manager.directory) if hasattr(manager, 'directory') else 'unknown'
        print(f"Checkpoint directory: {ckpt_dir}")
        
        # Try the positional API (matches notebook) - this should work
        manager.save(step, state_to_save)
        
        # Wait for async save to complete
        manager.wait_until_finished()
        
        # Verify the save worked by checking if directory exists
        step_dir = os.path.join(ckpt_dir, str(step))
        if os.path.exists(step_dir):
            print(f"✓ Checkpoint saved successfully at step {step} in {step_dir}")
        else:
            print(f"⚠ Warning: Checkpoint directory {step_dir} not found after save!")
            
    except Exception as e:
        print(f"✗ ERROR saving checkpoint at step {step}: {e}")
        import traceback
        traceback.print_exc()
        raise

# --- Helper: The Universal MLP Builder ---
# Matches Spinning Up's 'mlp' function but handles NNX RNGs
def build_mlp(
    sizes: Sequence[int], 
    activation: Callable, 
    rngs: nnx.Rngs, 
    output_activation: Callable = None
):
    layers = []
    for i in range(len(sizes) - 1):
        # Add Linear Layer
        layers.append(nnx.Linear(
            sizes[i], sizes[i+1], 
            kernel_init=nnx.initializers.orthogonal(1.414),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
            )
        )
        if i < len(sizes) - 2:
            layers.append(activation)
        elif output_activation is not None:
            layers.append(output_activation)
            
    return nnx.Sequential(*layers)


# --- 1. The Flexible Critic ---
class Critic(nnx.Module):
    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        hidden_sizes: Sequence[int] = (256, 256), 
        activation: Callable = nnx.relu,
        rngs: nnx.Rngs = None
    ):
        # Input to Q-net is Obs + Act
        input_dim = obs_dim + act_dim
        
        # Full architecture: [Input, ...Hidden..., 1]
        # We append [1] because Q-function outputs a single scalar value
        layer_sizes = [input_dim] + list(hidden_sizes) + [1]
        self.net1 = build_mlp(sizes=layer_sizes, activation=activation, rngs=rngs, output_activation=None)
        self.net2 = build_mlp(sizes=layer_sizes, activation=activation, rngs=rngs, output_activation=None)

    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        # Squeeze output to be (Batch,) instead of (Batch, 1)
        return self.net1(x).squeeze(-1), self.net2(x).squeeze(-1)
    

class Actor(nnx.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), rngs=None):
        # 1. Base Network
        self.net = build_mlp([obs_dim] + list(hidden_sizes), nnx.relu, rngs, nnx.relu)
        
        # 2. Heads
        last_size = hidden_sizes[-1]
        self.mu_layer = nnx.Linear(last_size, act_dim, 
                                   kernel_init=nnx.initializers.orthogonal(0.01), rngs=rngs)
        self.log_std_layer = nnx.Linear(last_size, act_dim, 
                                        kernel_init=nnx.initializers.orthogonal(0.01), rngs=rngs)
        self.act_limit = 1.0 # Standard for Brax/Gym

    def __call__(self, obs):
        """Returns the Distribution object (used for Training Loss)"""
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, -20, 2)
        
        base_dist = distrax.MultivariateNormalDiag(mu, jnp.exp(log_std))
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), ndims=1))

    def get_deterministic_action(self, obs):
        """Used for Evaluation (No Noise)"""
        x = self.net(obs)
        mu = self.mu_layer(x)
        # We just squash the mean directly. No sampling.
        return jnp.tanh(mu) * self.act_limit

    def get_stochastic_action(self, obs, key):
        """Used for Rollouts (With Noise)"""
        dist = self(obs)
        # Sample and scale (Distrax Tanh handles the squashing)
        # We act_limit is usually 1.0, but good to keep explicit
        return dist.sample(seed=key) * self.act_limit