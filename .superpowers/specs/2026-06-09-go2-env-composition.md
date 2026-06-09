# Go2 Warp env composition — host + pluggable Controller / Terrain / Actuation

**Date:** 2026-06-09 · **Status:** DRAFT (awaiting sign-off) · **Branch:** go2-osc-impedance

## Problem

Go2 env variation happens along **three orthogonal axes**, but each is currently
handled by a *different* mechanism, so combinations multiply classes + registry
names (24→18 envs for ~2 real classes):

| axis | values | current mechanism |
|---|---|---|
| controller | joint-PD / OSC / variable-impedance(+damping) | **subclass** (`_apply_control`, `action_size`) |
| terrain | flat / rough-heightfield | **mixin** (`_scene_xml`, `_customize_mj_model`) |
| actuation | torque-only / motor-model | **config flags** (`torque_speed_model`, `physical_armature`) |

Getting a combination means a mixin class (`WarpOscRoughHF(_RoughHFMixin,
WarpOscJoystick)`) × a registry name × a config closure. Adding a controller or
terrain grows the cross-product.

## Verified facts (what makes this safe)

- The seams are **tiny and non-overlapping**: controller = `(action_size,
  _apply_control)`; terrain = `(_scene_xml, customize-part)`; actuation =
  `(torque clip, armature-part)`. `obs / reward / step / reset` live entirely in
  `WarpJoystick` and are **never overridden** by OSC/var classes — they're the
  invariant task.
- **Only `mjx_backend.py` references the env classes** (registration). The one
  `isinstance` check (`mjx_backend.py:607`) is on `WarpJoystickCurriculum`, a
  separate class not in scope. `record_video`/tests load by **name** or use the
  controller **functions** (`compute_leg_impedance_torque`, `impedance_gains`) —
  not the classes. So keeping NAMES stable protects all downstream.

## Proposed design — one host, three components

`Go2WarpEnv` becomes the **host**: owns model-load, obs, reward, step, reset (the
invariant). Three pluggable component objects, selected at construction:

```python
class Controller(Protocol):
    def action_size(self, env) -> int: ...
    def setup(self, env) -> None: ...                 # was _post_init (cache site/dof ids, gains)
    def apply(self, env, data, action) -> mjx.Data:   # was _apply_control

class Terrain(Protocol):
    scene_xml: Path
    def customize_model(self, mj_model) -> None: ...  # fill hfield_data (no-op for flat)

class Actuation(Protocol):
    def customize_model(self, mj_model) -> None: ...  # set per-joint armature (no-op for torque-only)
    def clip_torque(self, tau, dq): ...               # torque-speed curve (identity for torque-only)
```

Host delegation (the only base changes):
```python
@property
def _scene_xml(self):          return self.terrain.scene_xml
def _customize_mj_model(self): self.terrain.customize_model(self._mj_model)
                               self.actuation.customize_model(self._mj_model)
@property
def action_size(self):         return self.controller.action_size(self)
def _apply_control(self, d, a):return self.controller.apply(self, d, a)
# substep torque path:         tau = self.actuation.clip_torque(tau, dq)
def _post_init(self):          ...; self.controller.setup(self)
```

**Implementations** (each small):
- Controller: `JointPD`, `OSC(kp, kd, use_lambda, target_mode, ridge)`,
  `VarImpedance(granularity, damping_action, s_range, z_range)`.
- Terrain: `Flat`, `RoughHF(profile, amp, seed)`.
- Actuation: `TorqueOnly`, `MotorModel(armature='mjlab', torque_speed=True)`.

**Component params vs config:** controller/terrain/actuation-specific params live
on the component objects (so the `_make_osc_kp_config(0.5)` closures dissolve into
`OSC(kp=[1500,1500,2000], kd=[78,78,92])`). The shared **task** config
(`ctrl_dt`, `sim_dt`, reward scales, push, obs noise, `action_scale`) stays as the
env `ConfigDict`. *(Recommendation: components own their params; revisit only if
some param is read in both obs and control.)*

## Registration — name → triple

The 18 names become a small table of triples (names UNCHANGED → checkpoints +
train/record keep working):

```python
ENVS = {
  "Go2WarpOscJoystickFlat":          (OSC(BASE_KP, BASE_KD),               Flat(),               TorqueOnly()),
  "Go2WarpOscJoystickFlatKp05":      (OSC(0.5*BASE_KP, ...),               Flat(),               TorqueOnly()),
  "Go2WarpOscRoughUni":              (OSC(SOFT_KP, SOFT_KD),               RoughHF("uniform",.07),MotorModel()),
  "Go2WarpOscVarDampingFlatPhysical":(VarImpedance("per_foot", damping=True),Flat(),             MotorModel()),
  "Go2WarpJoystickFlat":             (JointPD(),                           Flat(),               TorqueOnly()),
  ...
}
for name,(c,t,a) in ENVS.items(): register(name, lambda c=c,t=t,a=a: Go2WarpEnv(c,t,a, task_config()))
```

This table lives in a new `go2_warp_env_registry.py` next to the components;
`mjx_backend` just calls `register_go2_envs(pg_locomotion)`.

## Migration plan

1. Add the three Protocols + implementations (port logic verbatim from
   `_apply_control` / `_RoughHFMixin` / the two flags — no behavior change).
2. Add the delegation seams to `Go2WarpEnv`; keep `WarpJoystick` obs/reward as-is
   (it becomes / folds into the host).
3. Replace the env-class subclasses with the host + component table.
4. Keep all 18 NAMES; map each to a triple.
5. Delete the now-empty subclasses (`WarpOscJoystick`, `WarpOscVarImpedance`,
   rough mixins) once the table covers them.

## Risks & verification

- **Behavior drift** (the real risk): port `_apply_control`/`_post_init` logic
  *verbatim*. Verify with the existing tests (`test_go2_osc`, `test_torque_speed_model`,
  `test_var_impedance_damping`, `test_go2_osc_env`) — they exercise the controller
  math directly and must stay green.
- **Name resolution**: re-import → assert the same 18 env names resolve and build
  (action sizes match: 12/16/20/24/36).
- **Checkpoint load**: load one existing checkpoint through `record_video` → still
  runs (proves the name→triple host matches the old class).
- **Within-run equivalence**: for one env, roll the OLD class vs NEW host with the
  same seed/actions → identical qpos trajectory (GPU nondeterminism caveat: use
  same-process, same-key compare, not cross-run bit-identity).

## Out of scope

- Config-override loading (`--env Go2Warp --set ...`) — the triple table makes it
  natural later, but not this change.
- The richer electrical actuator model (held off; `Actuation` leaves room for a
  future `ElectricalMotor` impl).
- No physics/results change — pure restructure.
