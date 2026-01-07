import jax
import mujoco
import numpy as np
import imageio
from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# --- 1. ADAPTER (Must match training exactly) ---
PseudoState = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

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

# --- 2. SETUP & LOAD ---
env_name = 'CartpoleBalance'
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

print("Loading Policy...")
# Dummy call to reconstruct network structure
make_inference_fn, _, _ = ppo.train(
    environment=env,
    num_timesteps=0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    num_envs=1,
    seed=0
)
params = model.load_params('/tmp/mjx_cartpole_policy')
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# --- 3. VIDEO RECORDING LOOP ---
print("Simulating and Recording...")

# We use the CPU model for rendering
mj_model = raw_env.mj_model
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, height=480, width=640)

frames = []
rng = jax.random.PRNGKey(0)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Initialize
state = jit_reset(rng)

# Run for 500 steps (approx 8 seconds at 60fps)
for i in range(500):
    rng, key = jax.random.split(rng)
    
    # --- CRITICAL FIXES HERE ---
    # 1. Pass 'state.obs' (Array), not 'state' (Object)
    # 2. Unpack 'action, _' because the policy returns (action, extras)
    action, _ = jit_inference_fn(state.obs, key)
    
    # Step Physics (GPU)
    state = jit_step(state, action)
    
    # Sync to CPU for Rendering
    mj_data.qpos = np.array(state.pipeline_state.qpos)
    mj_data.qvel = np.array(state.pipeline_state.qvel)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Render Frame
    renderer.update_scene(mj_data)
    frames.append(renderer.render())

# --- 4. SAVE ---
output_name = 'cartpole_policy.mp4'
imageio.mimsave(output_name, frames, fps=60)
print(f"Video saved to: {output_name}")