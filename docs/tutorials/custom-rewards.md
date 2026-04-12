# Custom Rewards and Observations

*Intermediate — assumes familiarity with the [Concepts](../getting-started/concepts.md) page.*

This tutorial explains the **RewardSpec** and **ObsSpec** systems — composable building blocks for defining what the agent optimizes (rewards) and what it sees (observations). Both follow the same pattern: define a list of named terms, each backed by a function.

## RewardSpec

### How It Works

A reward specification is a list of `RewardTerm` objects. Each term has a name and a function:

```python
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards

@dataclass
class RewardTerm:
    name: str
    fn: Callable[..., Any]  # (**kwargs) -> scalar jax.Array
```

At each timestep, `compute_rewards()` calls every function and returns a dict of **unweighted** scalars:

```python
def compute_rewards(terms: list[RewardTerm], **kwargs) -> dict[str, Any]:
    return {term.name: term.fn(**kwargs) for term in terms}
```

The env's `step()` method then applies weights from `reward_config.scales`:

```python
rewards = compute_rewards(self._reward_spec, data=data, action=action,
                          info=info, done=done)
rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)
```

This separation means:

- Reward functions are pure math -- no weights baked in
- You can tune weights in the config without touching reward code
- Per-term weighted values are logged for diagnostics

### Adding a Reward Term

Write a function that takes keyword args and returns a scalar JAX array, then add it to the spec:

```python
def my_custom_reward(data, info, **kw) -> jax.Array:
    """Reward the robot for keeping its base at a target height."""
    base_z = data.subtree_com[1][2]  # torso COM z-position
    return jp.exp(-10.0 * jp.square(base_z - 0.3))

# In _post_init():
self._reward_spec.append(
    RewardTerm("target_height", my_custom_reward)
)
```

Then add the weight to your config:

```python
reward_config=config_dict.create(
    scales=config_dict.create(
        # ... existing terms ...
        target_height=5.0,  # positive = reward (maximize)
    ),
)
```

!!! tip "Convention"
    Positive weights mean "maximize this" (rewards). Negative weights mean "minimize this" (costs/penalties). For example, `tracking_lin_vel=10.0` rewards accurate velocity tracking, while `torques=-0.0002` penalizes high motor torques.

### Removing or Swapping Terms

Since the reward spec is just a Python list, you can modify it in `_post_init()`:

```python
# Remove a term by name
self._reward_spec = [t for t in self._reward_spec if t.name != "feet_slip"]

# Replace a term
for i, t in enumerate(self._reward_spec):
    if t.name == "orientation":
        self._reward_spec[i] = RewardTerm("orientation", my_better_orientation_fn)
        break
```

### Data-Driven Reward Spec

The bongo board env takes this further -- it defines **all possible** reward terms in a dict, then selects only the ones listed in the config:

```python
all_terms = {
    "survival": RewardTerm("survival", lambda **kw: jp.float32(1.0)),
    "orientation_cost": RewardTerm("orientation_cost", lambda data, **kw: ...),
    "board_tilt_cost": RewardTerm("board_tilt_cost", lambda data, **kw: ...),
    # ... more terms ...
}

# Only include terms that have weights in the config:
scales = self._config.reward_config.scales
self._reward_spec = [all_terms[k] for k in scales.keys()]
```

This lets you switch between reward presets (e.g., `config_b2()` vs `config_c()`) by changing only the config -- no code changes needed.

## ObsSpec

### How It Works

Observations follow the same composable pattern. An observation specification is a dict mapping group names to lists of `ObsTerm` objects:

```python
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs

@dataclass
class ObsTerm:
    name: str
    fn: Callable[..., Any]       # (**kwargs) -> jax.Array
    noise_scale: float = 0.0    # 0.0 = no noise

@dataclass
class IncludeGroup:
    group_name: str  # include another group's output
```

`compute_obs()` evaluates all terms, applies noise, and concatenates per group:

```python
def compute_obs(
    groups: dict[str, list[ObsTerm | IncludeGroup]],
    noise_level: float,
    rng: jax.Array,
    **kwargs,
) -> tuple[dict[str, jax.Array], jax.Array]:
    ...
```

It returns `(obs_dict, new_rng)` where `obs_dict` maps group names to concatenated arrays.

### The Two Standard Groups

Most environments define two observation groups:

- **"state"** -- what the policy (actor) sees at deployment. Includes noisy sensor readings to simulate real hardware. Typically 42-48 dimensions.
- **"privileged_state"** -- what the critic sees during training. Includes everything in "state" plus clean sensor readings, contact forces, and other information unavailable on the real robot. Typically 96-122 dimensions.

### Adding an Observation

Append an `ObsTerm` to the appropriate group:

```python
# Add base height to the state observations
self._obs_groups["state"].append(
    ObsTerm("base_height",
            lambda data, **kw: data.subtree_com[self._torso_body_id][2:3],
            noise_scale=0.01)
)
```

!!! note
    The function must return a JAX array. For scalar values, use slicing (`[2:3]`) or `.reshape(1)` to return a 1d array -- `compute_obs()` concatenates all terms with `jp.concatenate()`.

### IncludeGroup: Composing Groups

`IncludeGroup("state")` tells `compute_obs()` to paste the already-computed "state" array into the current group. This is how "privileged_state" includes all policy observations without duplicating term definitions:

```python
self._obs_groups = {
    "state": [
        ObsTerm("gyro", lambda data, **kw: self.get_gyro(data), noise_scale=0.2),
        ObsTerm("joint_pos", ..., noise_scale=0.03),
        # ...
    ],
    "privileged_state": [
        IncludeGroup("state"),  # paste the full 51d state vector here
        ObsTerm("actuator_force", lambda data, **kw: data.actuator_force),
        ObsTerm("contact_forces", ...),
        # ...
    ],
}
```

Groups are processed in dict insertion order, so "state" must be defined before "privileged_state".

### Noise

Each `ObsTerm` has an optional `noise_scale`. When `noise_level > 0` (set in env config), uniform noise is added:

```
noise = uniform(-1, 1) * noise_level * noise_scale
obs_value = clean_value + noise
```

The global `noise_level` (default 1.0) is a multiplier -- set it to 0.0 to disable all noise (useful for debugging).

### Example: Adding a Skill Vector for DIAYN

To add a fixed skill vector `z` (e.g., for diversity-driven exploration), it's one line:

```python
self._obs_groups["state"].append(
    ObsTerm("skill_z", lambda z=z, **kw: z)
)
```

The lambda captures `z` at definition time. No noise is applied.

## Putting It Together

Here's a minimal example combining both systems in a custom env:

```python
def _post_init(self):
    self._obs_groups = {
        "state": [
            ObsTerm("joint_pos", lambda data, **kw: data.qpos[7:], noise_scale=0.03),
            ObsTerm("joint_vel", lambda data, **kw: data.qvel[6:], noise_scale=1.5),
            ObsTerm("last_act", lambda info, **kw: info["last_act"]),
        ],
        "privileged_state": [
            IncludeGroup("state"),
            ObsTerm("contact", lambda info, **kw: info["contact"].astype(jp.float32)),
        ],
    }

    self._reward_spec = [
        RewardTerm("alive", lambda **kw: jp.float32(1.0)),
        RewardTerm("upright", lambda data, **kw:
            jp.exp(-jp.sum(self.get_gravity(data)[:2] ** 2))),
        RewardTerm("energy", lambda data, **kw:
            jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))),
    ]
```

With corresponding config:

```python
reward_config=config_dict.create(
    scales=config_dict.create(
        alive=1.0,
        upright=5.0,
        energy=-0.001,
    ),
)
```

## Next Steps

- [**Asymmetric Critic**](asymmetric-critic.md) -- give the critic access to privileged observations for faster learning
- [**Sim-to-Real**](sim2real.md) -- deploy your trained policy on real hardware
- [**Glossary**](../glossary.md) -- definitions for reward shaping, domain randomization, and other terms
