"""Domain randomization for Go2 Bongo Handstand environment.

Randomizes per-env:
  - Feet-board friction via explicit <pair> elements (bypasses max-combine)
  - Board mass
  - Robot link masses, motor strength, joint damping/armature/frictionloss
  - Torso COM jitter
"""

import jax
from mujoco import mjx


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    torso_body_id: int = 1,
    board_body_id: int = 18,
):
    """Randomize physics parameters per-env for vmapped training.

    Args:
        model: MJX model (unbatched).
        rng: (num_envs,2) array of per-env PRNG keys.
        torso_body_id: robot torso body index.
        board_body_id: bongo board body index.
    """

    @jax.vmap
    def rand_dynamics(rng):
        # ── Feet-board friction (via <pair> elements) ──────────────────
        # pair_friction is (npair, 5): [tangent1, tangent2, spin, roll1, roll2]
        # Pairs 0,1 = FL-board, FR-board (from scene XML order).
        rng, key = jax.random.split(rng)
        feet_board_fric = jax.random.uniform(key, minval=0.5, maxval=1.2)
        pair_friction = model.pair_friction
        pair_friction = pair_friction.at[0, :2].set(feet_board_fric)
        pair_friction = pair_friction.at[1, :2].set(feet_board_fric)

        # ── Board mass: *U(0.8, 1.2) ──────────────────────────────────
        rng, key = jax.random.split(rng)
        board_mass_scale = jax.random.uniform(key, minval=0.8, maxval=1.2)
        body_mass = model.body_mass.at[board_body_id].set(
            model.body_mass[board_body_id] * board_mass_scale
        )

        # ── Robot link masses: *U(0.8, 1.2) ───────────────────────────
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.8, maxval=1.2
        )
        # Apply to robot bodies only (skip board bodies already handled).
        # Simpler: scale all, then re-apply board separately.
        body_mass = body_mass.at[:].set(body_mass * dmass)
        # Re-apply board mass (already scaled once above).
        body_mass = body_mass.at[board_body_id].set(
            model.body_mass[board_body_id] * board_mass_scale
        )

        # ── Torso COM jitter: +U(-0.05, 0.05) ────────────────────────
        # Smaller range than Go2 joystick (±0.08) — handstand is more sensitive.
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
        body_ipos = model.body_ipos.at[torso_body_id].set(
            model.body_ipos[torso_body_id] + dpos
        )

        # ── Motor strength: *U(0.9, 1.1) ─────────────────────────────
        rng, key = jax.random.split(rng)
        motor_strength = jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            model.actuator_gainprm[:, 0] * motor_strength
        )

        # ── Joint damping: *U(0.7, 2.0) ──────────────────────────────
        rng, key = jax.random.split(rng)
        damping = model.dof_damping[6:18] * jax.random.uniform(
            key, shape=(12,), minval=0.7, maxval=2.0
        )
        dof_damping = model.dof_damping.at[6:18].set(damping)

        # ── Joint armature: *U(0.9, 1.3) ─────────────────────────────
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:18] * jax.random.uniform(
            key, shape=(12,), minval=0.9, maxval=1.3
        )
        dof_armature = model.dof_armature.at[6:18].set(armature)

        # ── Joint friction loss: *U(0.7, 1.5) ────────────────────────
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:18] * jax.random.uniform(
            key, shape=(12,), minval=0.7, maxval=1.5
        )
        dof_frictionloss = model.dof_frictionloss.at[6:18].set(frictionloss)

        return (
            pair_friction,
            body_mass,
            body_ipos,
            actuator_gainprm,
            dof_damping,
            dof_armature,
            dof_frictionloss,
        )

    (
        pair_friction,
        body_mass,
        body_ipos,
        actuator_gainprm,
        dof_damping,
        dof_armature,
        dof_frictionloss,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0,
        "body_mass": 0,
        "body_ipos": 0,
        "actuator_gainprm": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "dof_frictionloss": 0,
    })

    model = model.tree_replace({
        "pair_friction": pair_friction,
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "actuator_gainprm": actuator_gainprm,
        "dof_damping": dof_damping,
        "dof_armature": dof_armature,
        "dof_frictionloss": dof_frictionloss,
    })

    return model, in_axes
