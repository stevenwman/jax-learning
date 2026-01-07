import jax
import os
import mujoco
import numpy as np
import imageio
from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# --- 1. CRITICAL CONFIG ---
jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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

# --- 3. LOAD ---
env_name = 'HumanoidRun'
print(f"Loading {env_name} (x64)...")
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

print("Loading Policy...")
# Reconstruct network
make_inference_fn, _, _ = ppo.train(
    environment=env,
    num_timesteps=0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    num_envs=1,
    seed=0
)
params = model.load_params('/tmp/mjx_humanoid_policy')
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# --- 4. RENDER SETUP ---
print("Rendering Humanoid... (This may take 45s)")
mj_model = raw_env.mj_model

# FIX 1: Increase Framebuffer for HD Video
mj_model.vis.global_.offwidth = 1280
mj_model.vis.global_.offheight = 720

mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, height=720, width=1280)

frames = []
rng = jax.random.PRNGKey(1)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(rng)

# Run for 1000 steps
for i in range(1000):
    rng, key = jax.random.split(rng)
    action, _ = jit_inference_fn(state.obs, key)
    state = jit_step(state, action)
    
    if i % 2 == 0:
        mj_data.qpos = np.array(state.pipeline_state.qpos)
        mj_data.qvel = np.array(state.pipeline_state.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        # FIX 2: Use default camera (Removed 'camera="track"')
        # This will follow the "Free Camera" view.
        renderer.update_scene(mj_data) 
        
        frames.append(renderer.render())

output_name = 'humanoid_run.mp4'
imageio.mimsave(output_name, frames, fps=30)
print(f"Video saved to: {output_name}")