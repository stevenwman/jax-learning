import mujoco
import mujoco.mjx as mjx
import jax

# 1. Check JAX devices (should say 'gpu' not 'cpu')
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# 2. Try compiling a simple MJX model
xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <geom type="box" size=".1 .1 .1"/>
      <joint type="free"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
try:
    mjx_model = mjx.put_model(model)
    print("MJX model compiled successfully!")
except Exception as e:
    print(f"MJX compilation failed: {e}")