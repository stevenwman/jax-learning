# Algorithms

Six RL algorithms, each self-contained with no shared base class.

??? note "Why closures instead of methods?"
    JAX's JIT compiler traces Python functions and captures the values they close over. If we used regular methods (`self.update`), JAX would try to trace `self`, which is a mutable Python object — this breaks JIT.

    Instead, all algorithms define JIT'd functions as closures inside `__init__` that capture only JAX-compatible values (networks, configs, constants), then assign them to `self._update`, `self.select_action`, etc. This pattern is standard for JAX RL implementations (Brax, PureJaxRL use the same approach).

::: jax_rl.algos.ppo.PPO
    options:
      filters: ["!__init__"]

::: jax_rl.algos.sac.SAC
    options:
      filters: ["!__init__"]

::: jax_rl.algos.td3.TD3
    options:
      filters: ["!__init__"]

::: jax_rl.algos.fast_sac.FastSAC
    options:
      filters: ["!__init__"]

::: jax_rl.algos.fast_td3.FastTD3
    options:
      filters: ["!__init__"]

::: jax_rl.algos.flash_sac.FlashSAC
    options:
      filters: ["!__init__"]
