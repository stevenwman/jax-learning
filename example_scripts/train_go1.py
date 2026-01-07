import jax
import os
from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# --- 1. THE FIXED PSEUDOSTATE ---
# We subclass namedtuple to add the missing '.replace()' method
# JAX still treats this as a tuple, so it works with jit/scan automatically.
_PseudoStateBase = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

class PseudoState(_PseudoStateBase):
    def replace(self, **kwargs):
        # Map .replace() -> ._replace()
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

# --- 3. SETUP ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def progress(num_steps, metrics):
    reward = metrics['eval/episode_reward']
    # Try to grab velocity reward if available, otherwise just total
    print(f'Step {num_steps:,}: Reward = {reward:.2f}')

# --- 4. LOAD ENV ---
env_name = 'Go1JoystickFlatTerrain'
# Note: The first time you run this, it downloads ~100MB of meshes.
print(f"Loading {env_name}...") 
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

# --- 5. TRAIN ---
print("Starting PPO Training (Go1)...")
make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=60_000_000,
    num_evals=20,
    progress_fn=progress,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=1024,
    seed=0
)

# --- 6. SAVE ---
model_path = '/tmp/mjx_go1_policy'
model.save_params(model_path, params)
print(f"Saved Go1 policy to {model_path}")