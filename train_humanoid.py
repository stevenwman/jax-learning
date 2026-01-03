import jax
import os

# --- 1. ANTI-NAN CONFIG (Must be at the top) ---
jax.config.update("jax_enable_x64", True)
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# --- 2. ROBUST PSEUDOSTATE ---
_PseudoStateBase = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

class PseudoState(_PseudoStateBase):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

# --- 3. ADAPTER ---
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

def progress(num_steps, metrics):
    reward = metrics['eval/episode_reward']
    print(f'Step {num_steps:,}: Reward = {reward:.2f}')

# --- 4. LOAD HUMANOID ---
env_name = 'HumanoidRun' 
print(f"Loading {env_name} (x64 mode)...")
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

# --- 5. TRAIN ---
print("Starting Training...")
make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=100_000_000,
    num_evals=20,
    progress_fn=progress,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=4096,
    batch_size=2048,
    seed=0
)

# --- 6. SAVE ---
model_path = '/tmp/mjx_humanoid_policy'
model.save_params(model_path, params)
print(f"Saved Humanoid policy to {model_path}")