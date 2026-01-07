import jax
import mujoco
import numpy as np
import imageio
from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# --- 1. THE FIXED PSEUDOSTATE (Must match training) ---
# We define the base tuple structure
_PseudoStateBase = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

# We extend it to add the .replace() method that Playground expects
class PseudoState(_PseudoStateBase):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

# --- 2. THE ADAPTER ---
class BraxAdapter(Wrapper):
    def reset(self, rng):
        state = self.env.reset(rng)
        return self._to_brax_state(state)

    def step(self, state, action):
        # Unpack Brax State -> Pseudo Playground State
        mock_state = PseudoState(
            data=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=state.info
        )
        
        # Call Playground Step
        next_state = self.env.step(mock_state, action)
        
        # Repack -> Brax State
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

# --- 3. LOAD ENV ---
env_name = 'Go1JoystickFlatTerrain'
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

print("Loading Policy...")
# Dummy training call to reconstruct network structure
make_inference_fn, _, _ = ppo.train(
    environment=env,
    num_timesteps=0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    num_envs=1,
    seed=0
)

# Load the trained parameters
params = model.load_params('/tmp/mjx_go1_policy')
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# --- 4. RECORD ---
print("Rendering Go1... (This may take 30s to process frames)")
mj_model = raw_env.mj_model
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, height=720, width=1280)

frames = []
rng = jax.random.PRNGKey(1) 
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Initialize
state = jit_reset(rng)

# Run for 1000 steps (approx 20 seconds)
for i in range(1000):
    rng, key = jax.random.split(rng)
    
    # Inference & Step
    action, _ = jit_inference_fn(state.obs, key)
    state = jit_step(state, action)
    
    # Render every 2nd step to save time/space (30fps output)
    if i % 2 == 0:
        mj_data.qpos = np.array(state.pipeline_state.qpos)
        mj_data.qvel = np.array(state.pipeline_state.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        # Tracking camera
        renderer.update_scene(mj_data, camera="track")
        frames.append(renderer.render())

# Save
output_name = 'go1_walking.mp4'
imageio.mimsave(output_name, frames, fps=30)
print(f"Video saved to: {output_name}")