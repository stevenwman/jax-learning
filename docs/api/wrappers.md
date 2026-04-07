# Wrappers

Environment wrappers for observation transforms, action delays, vectorization, and episode management. The pipeline module provides a declarative way to compose wrappers from config.

::: jax_rl.envs.wrappers.frame_stack.FrameStackWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.action_delay.ActionDelayWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.pipeline.build_wrapper_pipeline

::: jax_rl.envs.wrappers.pipeline.apply_wrapper_pipeline

::: jax_rl.envs.wrappers.training.VmapWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.training.EpisodeWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.training.AutoResetWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.training.DomainRandomizationVmapWrapper
    options:
      filters: ["!__init__"]

::: jax_rl.envs.wrappers.training.wrap_for_training
