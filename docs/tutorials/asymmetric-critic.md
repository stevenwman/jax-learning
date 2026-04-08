# Asymmetric Critic

*Intermediate — assumes familiarity with actor-critic RL ([Concepts](../getting-started/concepts.md)).*

An asymmetric critic is a training technique where the critic (value function) sees more information than the actor (policy). During training, the critic uses privileged observations -- terrain info, contact forces, noise-free sensors -- to provide better gradient signals. At deployment, only the actor runs, using the limited observations available on the real robot.

## How It Works

The environment returns two observation groups:

| Group | Dims | Contents | Used by |
|-------|------|----------|---------|
| `"state"` | 48d | Noisy joint positions, velocities, gyro, gravity, commands, last action | Actor (policy) |
| `"privileged_state"` | 122d | Everything in "state" + clean sensors, actuator forces, contacts, foot velocities, air times, external forces | Critic (value/Q function) |

The actor learns to map 48d noisy observations to actions. The critic learns to evaluate state-action pairs using 122d privileged observations. Because the critic is only used during training (to compute TD targets or advantages), the extra information improves learning speed without affecting what the deployed policy needs.

## Algorithm Support

### PPO

PPO has built-in asymmetric support. The actor and critic are separate networks with separate observation paths:

- Actor: `obs["state"]` (48d) --> action
- Critic: `obs["privileged_state"]` (122d) --> value estimate

This is automatic when the environment returns dict observations.

### Off-Policy (FastSAC / SAC / TD3)

Off-policy algorithms also support asymmetric observations:

- Actor network: receives 48d `"state"` observations
- Critic network: receives 122d `"privileged_state"` observations

The config handles the routing -- the training loop passes the correct observation slice to each network.

## A/B Results on Go2

We tested asymmetric vs symmetric critics on `Go2WarpJoystickFlat` with FastSAC:

| Metric | Symmetric (48d/48d) | Asymmetric (48d/122d) |
|--------|---------------------|-----------------------|
| Steps to 270+ eval | ~9M | ~5M |
| Final eval score | 276 | 279 |
| Training wall time | ~8 min | ~8 min |

The asymmetric critic reaches the 270+ performance threshold roughly **2x faster** (5M vs 9M steps). However, the final performance ceiling is the same within noise (279 vs 276). The bottleneck for final performance is the actor's 48d observation space, not the critic's information.

!!! note "Why the same ceiling?"
    The critic helps the actor learn faster by providing better value estimates early in training. But the actor can only learn behaviors that are achievable with its 48d observations. Once the actor has extracted all useful information from its inputs, additional critic information doesn't help.

## When to Use Asymmetric Critics

**Always use for PPO on locomotion.** There's no downside -- the critic is discarded at deployment, and the extra information speeds up training.

**Use for off-policy when early learning speed matters.** If you're iterating on reward functions and want fast feedback, asymmetric critics cut iteration time in half. If you're doing a final long training run, the benefit is smaller since both approaches converge to the same ceiling.

**Use when privileged information is available.** If your environment has information that's available in simulation but not on hardware (terrain maps, contact forces, object poses), put it in the privileged observations. The critic uses it during training; the actor learns to infer what it can from limited sensors.

!!! tip
    There is no observed downside to using asymmetric critics.

## Frame Stacking

A related technique is frame stacking -- feeding the actor multiple timesteps of observations to provide temporal context. Results vary by task:

### Locomotion (Go2 Joystick)

Frame stacking (3 frames) did **not** help:

| Config | Eval Score |
|--------|-----------|
| No stacking (baseline) | 276.5 |
| 3-frame stacking | 271.3 |

The `last_action` term in the observation already provides sufficient temporal context for locomotion. Stacking triples the actor's input dimensionality (48d to 144d) without adding useful information, slightly hurting performance.

### Balance Tasks (Bongo Board)

Frame stacking **is critical** for balance:

| Config | Eval Score |
|--------|-----------|
| No stacking | 24 |
| 3-frame stacking | 47 |

Balance tasks require tracking velocity trends and oscillation patterns that a single frame can't capture. The board tilt rate and roller acceleration are implicit in consecutive frames but not available as single-frame observations.

!!! tip "Rule of thumb"
    If your task involves dynamic balance or requires estimating velocities/accelerations that aren't directly observed, try frame stacking. For locomotion with proprioceptive observations that already include velocities and last action, skip it.

## Next Steps

- [**Custom Rewards**](custom-rewards.md) -- swap reward terms and observation groups
- [**Sim-to-Real**](sim2real.md) -- deploy your trained policy on real hardware
- [**Glossary**](../glossary.md) -- definitions for actor-critic, privileged state, and other terms
