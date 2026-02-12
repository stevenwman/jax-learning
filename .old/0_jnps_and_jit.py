import jax
import jax.numpy as jnp

jax.config.update("jax_log_compiles", True)

@jax.jit
def simple_update(params, x):
    return params * x + 1.0

params = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([0.5, 0.5, 0.5])
result = simple_update(params, x)

print(result)