# mud_eval/archive — staged for deletion

These scripts are **not part of the live Newton mud eval** and are staged here for
removal (kept temporarily as a development/reproduction trail). The live eval is
`../record_traverse_maxfwd.py` + the `../mud_{model,costep,cpu,osc,jax_policy}.py`
modules; `../probe_forces.py` is kept as the traction-loss force-probe diagnostic.

NOTE: the modules these scripts `import` (`mud_costep`, `mud_model`, …) now live in
the parent dir, so archived scripts won't run as-is from here (they predate / aren't
on the eval path). That's expected — they're history, not runnable.

## Superseded record scripts (replaced by `record_traverse_maxfwd.py`)
- `record_traverse.py` — earlier traverse recorder (no parity-spawn fix, no mass)
- `record_mud.py`, `record_falloff.py` — early single-clip recorders
- `run_mud_eval.py` — early eval runner

## M0–M3 build-time validation gates + diagnostics (one-off)
- `m0_smoke.py` — M0 smoke (robot + MPM finite, torso_z)
- `gate_ground{,_cpu,_urdf}.py`, `gate_mjcf_coupling.py` — ground-contact / coupling gates
- `gate_osc_{jac,torque,loop}.py` — OSC J/M/torque validation gates
- `inspect_{collision,contacts,go2xml,ground}.py` — model/contact inspectors
- `mud_diag.py` — misc diagnostics

History/context for these is in `../HANDOFF.md` (M0–M3 sections).
