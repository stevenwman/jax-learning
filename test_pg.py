from mujoco_playground import dm_control_suite
import jax
import jax.numpy as jnp

env = dm_control_suite.load('CartpoleBalance')
state = env.reset(jax.random.PRNGKey(0))

# Step a few times
for _ in range(10):
    state = env.step(state, jnp.zeros(env.action_size))

# Check render output
frames = env.render(state)
print(type(frames), frames.shape if hasattr(frames, 'shape') else '')