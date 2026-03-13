from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training
import jax

num_envs = 4

# Load base env
env = dm_control_suite.load('CartpoleBalance')

# Playground's wrap_for_brax_training applies:
#   VmapWrapper -> EpisodeWrapper -> BraxAutoResetWrapper
# Uses state.data (MJX) instead of state.pipeline_state (Brax)
env = wrap_for_brax_training(env, episode_length=1000)

# VmapWrapper.reset expects a batch of keys: (num_envs, 2)
key = jax.random.PRNGKey(0)
reset_keys = jax.random.split(key, num_envs)
state = env.reset(reset_keys)

# Print state fields and shapes
print(f"obs:        {state.obs.shape}")
print(f"reward:     {state.reward.shape}")
print(f"done:       {state.done.shape}")
print(f"info keys:  {list(state.info.keys())}")
print(f"truncation: {state.info['truncation'].shape}")
print(f"steps:      {state.info['steps'].shape}")

# Step with random actions
key, subkey = jax.random.split(key)
action = jax.random.uniform(subkey, (num_envs, env.action_size), minval=-1, maxval=1)
next_state = env.step(state, action)

print(f"\nAfter step:")
print(f"obs:        {next_state.obs.shape}")
print(f"reward:     {next_state.reward.shape}")
print(f"done:       {next_state.done.shape}")
print(f"truncation: {next_state.info['truncation'].shape}")
print(f"steps:      {next_state.info['steps'].shape}")
