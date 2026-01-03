import jax
import os
from collections import namedtuple
from mujoco_playground import registry
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.envs import Wrapper, State

# 0. HELPER FOR JAX COMPATIBILITY
# We define a complete structure that mimics Playground's expected State
PseudoState = namedtuple('PseudoState', ['data', 'obs', 'reward', 'done', 'metrics', 'info'])

# --- 1. THE COMPLETE ADAPTER ---
class BraxAdapter(Wrapper):
    def reset(self, rng):
        state = self.env.reset(rng)
        return self._to_brax_state(state)

    def step(self, state, action):
        # 1. UNPACK: Brax State -> Pseudo Playground State
        # We must populate ALL fields, because the env might read 'info' or 'metrics'
        mock_state = PseudoState(
            data=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=state.info
        )
        
        # 2. Call Playground Step
        next_state = self.env.step(mock_state, action)
        
        # 3. REPACK: Playground State -> Brax State
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

# --- 2. SETUP ---
# Helps avoid NaN errors on RTX cards
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def progress(num_steps, metrics):
    reward = metrics['eval/episode_reward']
    print(f'Step {num_steps:,}: Reward = {reward:.2f}')

# --- 3. LOAD & WRAP ---
env_name = 'CartpoleBalance'
raw_env = registry.load(env_name)
env = BraxAdapter(raw_env)

print(f"Training {env_name} on GPU...")

# --- 4. TRAIN ---
make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=10_000_000,
    num_evals=20,
    progress_fn=progress,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=1024,
    seed=0
)

# --- 5. SAVE ---
model_path = '/tmp/mjx_cartpole_policy'
model.save_params(model_path, params)
print(f"Saved policy to {model_path}")