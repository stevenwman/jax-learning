import os
os.environ["MUJOCO_GL"] = "egl"  # Headless rendering
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Helps avoid memory issues with large batches
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import mujoco
import numpy as np
import imageio
import orbax.checkpoint as ocp
from collections import namedtuple

# Playground Imports
from mujoco_playground import registry
from brax.envs import Wrapper, State

from algos.sac.core import Actor, Critic, create_checkpoint_manager

# --- 1. CONFIGURATION ---
# ENV_NAME = 'HumanoidRun'  # Change this to match your trained model
# ENV_NAME = 'Go1JoystickFlatTerrain'
ENV_NAME = 'CartpoleSwingup'
CHECKPOINT_DIR = "./sac_checkpoints/CartpoleSwingup_20260107_214453"  # Must match your training checkpoint directory
CHECKPOINT_STEP = None  # None = use latest checkpoint, or specify step number (e.g., 50000)
HIDDEN_SIZES = (256, 256)  # Must match training
LR = 3e-4  # Only needed for optimizer initialization, not used
OUTPUT_NAME = None  # None = auto-generate from ENV_NAME
NUM_STEPS = 500  # Number of simulation steps to record
FPS = 60  # Video framerate

# --- 2. ADAPTERS ---
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

# --- 3. CHECKPOINT RESTORATION ---
def restore_ckpt(manager, actor, critic, target_critic, actor_opt, critic_opt, step=None):
    """
    Restore checkpoint and update models in-place. Returns step number.
    
    Args:
        manager: Checkpoint manager
        actor, critic, target_critic, actor_opt, critic_opt: Models to restore
        step: Specific step to restore, or None to use latest
    """
    if step is None:
        step = manager.latest_step()
        if step is None:
            print("No checkpoint found. Starting fresh.")
            return 0
        print(f"Using latest checkpoint: step {step}")
    else:
        # Check if the specified step exists
        available_steps = manager.all_steps()
        if step not in available_steps:
            latest = manager.latest_step()
            raise ValueError(
                f"Checkpoint step {step} not found. "
                f"Available steps: {sorted(available_steps)[-10:]} (showing last 10). "
                f"Latest: {latest}. "
                f"Please specify a valid step number."
            )
        print(f"Restoring from specified step {step}...")

    # Create structure template matching the saved format
    target_payload = {
        'actor': nnx.state(actor),
        'critic': nnx.state(critic),
        'target_critic': nnx.state(target_critic),
        'actor_opt': nnx.state(actor_opt),
        'critic_opt': nnx.state(critic_opt),
    }

    restored = manager.restore(step, items=target_payload)

    # Update objects in-place
    nnx.update(actor, restored['actor'])
    nnx.update(critic, restored['critic'])
    nnx.update(target_critic, restored['target_critic'])
    nnx.update(actor_opt, restored['actor_opt'])
    nnx.update(critic_opt, restored['critic_opt'])

    print(f"✓ Successfully restored checkpoint from step {step}.")
    return step

# --- 4. MAIN RECORDING FUNCTION ---
def main():
    print(f"Initializing Env: {ENV_NAME}")
    raw_env = registry.load(ENV_NAME)
    env = BraxAdapter(raw_env)
    
    # Handle dictionary observation spaces (e.g., Go1JoystickFlatTerrain)
    raw_obs_size = env.observation_size
    use_dict_obs = isinstance(raw_obs_size, dict)
    
    if use_dict_obs:
        obs_key = 'state' if 'state' in raw_obs_size else list(raw_obs_size.keys())[0]
        obs_shape = raw_obs_size[obs_key]
        obs_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        print(f"Detected dict observation space: {raw_obs_size}")
        print(f"Using '{obs_key}' key with dim={obs_dim}")
    else:
        obs_dim = raw_obs_size
        obs_key = None
        print(f"Using flat observation space with dim={obs_dim}")
    
    act_dim = env.action_size
    print(f"Obs: {obs_dim} | Act: {act_dim}")
    
    # Helper to extract observations - must be JAX-traceable
    def flatten_obs(obs):
        if use_dict_obs:
            return obs[obs_key]
        else:
            return obs

    # --- INIT ACTOR & CRITICS, OPTIMIZERS ---
    key = nnx.Rngs(0)
    actor = Actor(obs_dim, act_dim, HIDDEN_SIZES, rngs=key)
    critic = Critic(obs_dim, act_dim, HIDDEN_SIZES, rngs=key)
    target_critic = Critic(obs_dim, act_dim, HIDDEN_SIZES, rngs=nnx.Rngs(0))

    actor_opt = nnx.Optimizer(actor, optax.adam(LR), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, optax.adam(LR), wrt=nnx.Param)

    # --- SETUP CHECKPOINT MANAGER & RESTORE ---
    ckpt_manager = create_checkpoint_manager(CHECKPOINT_DIR)
    restored_step = restore_ckpt(ckpt_manager, actor, critic, target_critic, actor_opt, critic_opt, step=CHECKPOINT_STEP)

    if restored_step == 0:
        raise ValueError(
            f"No checkpoints found in {os.path.abspath(CHECKPOINT_DIR)}. "
            "Please train a model first or check the checkpoint directory path."
        )

    # --- SETUP RENDERING ---
    mj_model = raw_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    @jax.jit
    def get_action(obs):
        # Flatten observation if needed
        flat_obs = flatten_obs(obs)
        obs_batched = flat_obs[None, :]
        action_batched = actor.get_deterministic_action(obs_batched)
        return action_batched[0]

    # --- SIMULATE & RECORD ---
    print("Simulating...")
    frames = []
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)

    episode_ended = False
    for i in range(NUM_STEPS):
        action = get_action(state.obs)
        state = jit_step(state, action)
        
        # Update MuJoCo data structures for rendering
        mj_data.qpos = np.array(state.pipeline_state.qpos)
        mj_data.qvel = np.array(state.pipeline_state.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        # Render frame
        renderer.update_scene(mj_data)
        frames.append(renderer.render())
        
        if state.done and not episode_ended:
            print(f"Episode ended at step {i}")
            episode_ended = True
            # Optionally reset: state = jit_reset(rng)

    # --- SAVE VIDEO ---
    output_name = OUTPUT_NAME if OUTPUT_NAME else f'{ENV_NAME}_sac.mp4'
    print(f"Saving video to: {output_name}...")
    imageio.mimsave(output_name, frames, fps=FPS)
    print(f"✓ Video saved to: {output_name} ({len(frames)} frames @ {FPS} fps)")

if __name__ == "__main__":
    main()

