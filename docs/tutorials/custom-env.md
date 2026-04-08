# Custom Environment

*Advanced — requires familiarity with Python dataclasses, MuJoCo XML, and JAX basics.*

This tutorial walks through adding a new Go2 environment, using the **bongo board handstand** task as a worked example.

<video autoplay loop muted playsinline style="width: 100%; max-width: 640px; border-radius: 8px;">
  <source src="/assets/videos/go2_bongo_handstand.mp4" type="video/mp4">
</video>

## Overview

Adding a new environment involves these files:

```
jax_rl/envs/locomotion/
├── xmls/go2_bongo_scene.xml          # 1. Scene XML
├── go2_bongo_handstand.py            # 2. Env class
jax_rl/training/env_setup.py          # 3. Registration
jax_rl/configs/env_presets.py         # 4. Preset config
```

## Step 1: Create the Scene XML

Your scene XML combines the robot MJCF, task objects, and sensors into a single simulation. For the bongo board, this means:

- The Go2 robot (included from vendored `unitree_go2/go2.xml`)
- A bongo board (flat platform) and cylindrical roller
- Contact sensors for detecting foot-board, body-floor, and board-floor contacts
- A keyframe defining the initial handstand pose

The scene XML lives at `jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml`. Key structure:

```xml
<mujoco model="go2_bongo">
  <!-- Include the robot model -->
  <include file="unitree_go2/go2.xml"/>

  <!-- Task objects: board + roller -->
  <body name="board" pos="0 0 0.34">
    <freejoint name="board_joint"/>
    <geom name="board_top" type="box" size="0.25 0.15 0.01" mass="1.0"/>
  </body>
  <body name="roller" pos="0 0 0.05">
    <joint name="roller_slide" type="slide" axis="1 0 0"/>
    <geom type="cylinder" size="0.05 0.15"/>
  </body>

  <!-- Contact sensors (touch detection via MuJoCo sensor API) -->
  <sensor>
    <touch name="FL_board_found" site="FL_foot"/>
    <touch name="board_floor_found" site="board_edge"/>
    <!-- ... more contact sensors ... -->
  </sensor>

  <!-- Initial pose keyframe -->
  <keyframe>
    <key name="handstand" qpos="..."/>
  </keyframe>
</mujoco>
```

!!! tip
    Use MuJoCo's `touch` sensors for contact detection rather than iterating over `data.contact`. Sensors are faster in Warp and give you named, stable indices.

## Step 2: Write the Env Class

Subclass `Go2WarpEnv` (from `go2_warp_base.py`) and implement the core methods. Here's the structure from `BongoHandstand`:

```python
from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


class BongoHandstand(go2_warp_base.Go2WarpEnv):

    def __init__(self, task, config=default_config(), config_overrides=None):
        super().__init__(
            xml_path=consts.BONGO_SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        """Set up obs groups, reward spec, body IDs, sensor addresses."""
        ...

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Initialize qpos/qvel from keyframe, create info dict."""
        ...

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """PD control, physics step, compute obs/reward/done."""
        ...

    def _get_obs(self, data, info) -> Dict[str, jax.Array]:
        """Delegate to compute_obs() with self._obs_groups."""
        ...

    def _get_termination(self, data) -> jax.Array:
        """Return True if episode should end."""
        ...

    def _get_reward(self, data, action, info, done) -> dict:
        """Delegate to compute_rewards() with self._reward_spec."""
        ...
```

### _post_init: Observations

Define observation groups as lists of `ObsTerm`. Each term has a name, a function that extracts the observation, and an optional noise scale:

```python
noise = self._config.noise_config.scales
self._obs_groups = {
    "state": [
        ObsTerm("gyro", lambda data, **kw: self.get_gyro(data),
                noise_scale=noise.gyro),
        ObsTerm("gravity", lambda data, **kw: self.get_gravity(data),
                noise_scale=noise.gravity),
        ObsTerm("joint_pos_offset", lambda data, **kw: data.qpos[7:19] - self._default_pose,
                noise_scale=noise.joint_pos),
        ObsTerm("joint_vel", lambda data, **kw: data.qvel[6:18],
                noise_scale=noise.joint_vel),
        ObsTerm("last_act", lambda info, **kw: info["last_act"]),
        # Task-specific obs: board tilt, roller position/velocity
        ObsTerm("board_tilt", lambda data, **kw: self._get_board_tilt(data)),
        ObsTerm("roller_pos", lambda data, **kw: data.qpos[self._roller_slide_qposadr].reshape(1)),
        ObsTerm("roller_vel", lambda data, **kw: data.qvel[self._roller_slide_dofadr].reshape(1)),
    ],
    "privileged_state": [
        IncludeGroup("state"),  # includes all of "state" automatically
        ObsTerm("gyro_clean", lambda data, **kw: self.get_gyro(data)),
        ObsTerm("actuator_force", lambda data, **kw: data.actuator_force),
        ObsTerm("foot_board_contact", lambda data, **kw: ...),
        ObsTerm("com_rel_board", lambda data, **kw:
            data.subtree_com[self._torso_body_id][:2] - data.xpos[self._board_body_id][:2]),
    ],
}
```

### _post_init: Rewards

Define reward terms as a list of `RewardTerm(name, fn)`. The bongo env uses a data-driven approach -- only terms whose names appear in `reward_config.scales` are included:

```python
all_terms = {
    "survival": RewardTerm("survival", lambda **kw: jp.float32(1.0)),
    "orientation_cost": RewardTerm("orientation_cost", lambda data, **kw:
        jp.clip(jp.sum((self.get_gravity(data) - target_gravity) ** 2) / 4.0, 0.0, 1.0)),
    "board_tilt_cost": RewardTerm("board_tilt_cost", lambda data, **kw:
        jp.clip(jp.sum(self._get_board_tilt(data) ** 2) / 0.5, 0.0, 1.0)),
    "termination": RewardTerm("termination", lambda done, **kw: done),
    # ... more terms ...
}

# Only include terms that have weights in the config:
scales = self._config.reward_config.scales
self._reward_spec = [all_terms[k] for k in scales.keys()]
```

!!! note
    `compute_rewards()` returns **unweighted** values. The `step()` method applies weights from `reward_config.scales` afterward.

### step: Physics Loop

The step method follows a standard pattern:

1. Compute motor targets from action
2. Run PD control at physics rate via `jax.lax.scan`
3. Compute observations, termination, and rewards
4. Apply reward weights and clip

```python
def step(self, state, action):
    motor_targets = self._default_pose + action * self._config.action_scale

    # PD control at physics rate (5 substeps per control step)
    def substep(data, _):
        tau = kp * (motor_targets - data.qpos[7:19]) + kd * (0.0 - data.qvel[6:18])
        data = data.replace(ctrl=tau[a2j])  # remap joint->actuator order
        return mjx.step(model, data), None

    data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)
    rewards = compute_rewards(self._reward_spec, data=data, action=action,
                              info=state.info, done=done)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)
    ...
```

## Step 3: Register the Environment

Add your env to `_register_custom_envs()` in `jax_rl/training/env_setup.py`:

```python
from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand
from jax_rl.envs.locomotion.go2_bongo_handstand import default_config as bongo_default_config

if "Go2BongoHandstand" not in pg_locomotion._envs:
    pg_locomotion.register_environment(
        "Go2BongoHandstand",
        functools.partial(BongoHandstand, task="bongo_handstand"),
        bongo_default_config,
    )
```

This makes the env available to all training scripts via the `--env Go2BongoHandstand` flag.

## Step 4: Add a Config Preset

Add your env's training hyperparameters to `jax_rl/configs/env_presets.py`. This is optional but saves you from specifying everything on the command line:

```python
FAST_SAC_PRESETS: dict[str, tuple[TrainConfig, FastSACConfig]] = {
    # ... existing presets ...
    "Go2BongoHandstand": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2BongoHandstand"),
        _FAST_SAC_BASE_ALGO,
    ),
}
```

## Step 5: Smoke Test

Run a short training to verify everything works:

```bash
uv run python train_offpolicy.py \
    --algo fast_sac \
    --env Go2BongoHandstand \
    --num-envs 64 \
    --total-timesteps 200000
```

You should see:

- No import errors or crashes
- Observations being computed (printed shapes in the first few steps)
- Rewards starting near zero and changing (even if not improving much in 200k steps)

!!! warning
    If you get `KeyError` on a sensor name, double-check that your scene XML defines all the sensors your env class references. Sensor names must match exactly between the XML and Python code.

## Summary

| Step | File | What to do |
|------|------|------------|
| 1 | `xmls/<scene>.xml` | Combine robot MJCF + task objects + sensors |
| 2 | `go2_<task>.py` | Subclass `Go2WarpEnv`, implement core methods |
| 3 | `env_setup.py` | Register with `pg_locomotion.register_environment()` |
| 4 | `env_presets.py` | Add training hyperparameter preset |
| 5 | CLI | Smoke test with `--num-envs 64 --total-timesteps 200000` |
