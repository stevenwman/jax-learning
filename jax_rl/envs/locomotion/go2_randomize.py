"""Domain randomization for Go2 environment.

Aggressive DR for MJX→CPU→real transfer robustness.
Ranges informed by walk-these-ways Go2 fork, unitree_rl_lab, and Isaac Lab.
"""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1  # "base" in Go2


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomize physics parameters per-env for vmapped training."""

  @jax.vmap
  def rand_dynamics(rng):
    # Friction: randomize ALL geoms (MuJoCo uses max-combine, so floor-only
    # doesn't work if foot friction caps it). Range from WTW: [0.05, 4.5].
    rng, key = jax.random.split(rng)
    fric_val = jax.random.uniform(key, minval=0.05, maxval=4.5)
    # Set tangential friction on all geoms uniformly
    geom_friction = model.geom_friction.at[:, 0].set(fric_val)

    # Scale DOF friction loss: *U(0.7, 1.5).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(12,), minval=0.7, maxval=1.5
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(0.9, 1.3).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(12,), minval=0.9, maxval=1.3
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale DOF damping: *U(0.7, 2.0) — covers MJX/CPU gap without being too extreme.
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[6:] * jax.random.uniform(
        key, shape=(12,), minval=0.7, maxval=2.0
    )
    dof_damping = model.dof_damping.at[6:].set(damping)

    # Jitter torso COM: +U(-0.08, 0.08).
    rng, key = jax.random.split(rng)
    dpos = jax.random.uniform(key, (3,), minval=-0.08, maxval=0.08)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )

    # Scale all link masses: *U(0.8, 1.2) — wider (was 0.9-1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.8, maxval=1.2
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add payload mass to torso: +U(-1.0, 3.0) kg.
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=3.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Motor strength: *U(0.9, 1.1) — models battery sag / motor variation.
    rng, key = jax.random.split(rng)
    motor_strength = jax.random.uniform(
        key, shape=(12,), minval=0.9, maxval=1.1
    )
    # Apply as actuator gain scaling (gainprm[0] is the torque multiplier)
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
        model.actuator_gainprm[:, 0] * motor_strength
    )

    # Jitter initial joint positions: +U(-0.1, 0.1) — wider (was ±0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(12,), minval=-0.1, maxval=0.1)
    )

    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
    )

  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
  })

  return model, in_axes
