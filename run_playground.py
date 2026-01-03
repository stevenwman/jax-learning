import jax
import jax.numpy as jnp
from mujoco_playground import registry

# 1. Load the Playground Cartpole
# The wrapper automatically handles backend setup
env = registry.load('CartpoleBalance')
print(f"Environment loaded: {type(env)}")  # Fixed print line

# 2. JIT Compile
jit_step = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

# 3. Run a quick loop
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

print("Running 1000 steps...")
for _ in range(1000):
    rng, key = jax.random.split(rng)
    
    # Check if 'action_size' exists (Brax style) or infer from spec
    act_size = getattr(env, 'action_size', 1) 
    
    action = jax.random.uniform(key, shape=(act_size,), minval=-1, maxval=1)
    state = jit_step(state, action)

# Block to ensure GPU finished
state.data.qpos.block_until_ready()
print("Success! Playground is running on GPU.")