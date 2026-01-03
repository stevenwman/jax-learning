import time
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco_playground import registry

# 1. Load the Environment
print("Initializing environment...")
env = registry.load('CartpoleBalance')

# 2. Setup the Viewer (CPU side)
# We need the standard CPU mujoco model to drive the visualization
mj_model = env.mj_model
mj_data = mujoco.MjData(mj_model)

# 3. JIT Compile the GPU functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# 4. Initialize Simulation
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

print("Launching viewer... (Press ESC to quit)")

# 5. The Visualization Loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # We'll target 60 FPS for viewing (even though physics runs faster)
    target_dt = 1.0 / 60.0
    
    while viewer.is_running():
        loop_start = time.time()

        # --- GPU PHYSICS ---
        rng, key = jax.random.split(rng)
        # Random action just to see it wiggle
        action = jax.random.uniform(key, shape=(env.action_size,), minval=-1, maxval=1)
        state = jit_step(state, action)
        
        # --- SYNC TO CPU ---
        # We copy the joint positions (qpos) and velocities (qvel) from GPU to CPU
        # np.array() forces the transfer
        mj_data.qpos = np.array(state.data.qpos)
        mj_data.qvel = np.array(state.data.qvel)
        
        # We must call mj_forward on CPU to update the geometry positions 
        # based on the new qpos we just pulled
        mujoco.mj_forward(mj_model, mj_data)
        
        # --- RENDER ---
        viewer.sync()

        # --- TIMING ---
        # Sleep to keep it viewable (otherwise it runs at 20,000 FPS)
        elapsed = time.time() - loop_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)