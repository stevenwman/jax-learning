import jax
import os

# --- 1. ANTI-NAN CONFIG ---
jax.config.update("jax_enable_x64", True)
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

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

def progress(num_steps, metrics):
    print(f'Step {num_steps:,}: Reward = {metrics["eval/episode_reward"]:.2f}')

# --- 3. RUN ---
env_name = 'HumanoidRun'
print(f"Loading {env_name} (Stable Config)...")
env = BraxAdapter(registry.load(env_name))

print("Starting Stable Training (30M steps)...")
make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=30_000_000,      
    num_evals=15,                  
    progress_fn=progress,
    reward_scaling=0.01,           # CRITICAL: Scale huge rewards down to ~1.0 range
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,              # Longer unroll helps it see "future" stability
    num_minibatches=32,
    num_updates_per_batch=4,       # Reduced from 8 to prevent overfitting
    discounting=0.99,              # Increased from 0.97 for long-term balance
    learning_rate=5e-5,            # Reduced from 3e-4 to stop "spazzing"
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    seed=0
)

# --- 4. SAVE ---
model_path = '/tmp/mjx_humanoid_policy'
model.save_params(model_path, params)
print(f"SUCCESS: Saved policy to {model_path}")