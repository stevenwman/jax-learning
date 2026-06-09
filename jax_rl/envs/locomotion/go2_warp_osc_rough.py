"""Go2 on a ROUGH HEIGHTFIELD floor (Warp), with joint-PD / OSC / variable-
impedance controllers — for testing whether foot compliance helps on terrain
that is *actually* rough (unlike the box-curriculum, which left the robot on the
flat border).

The floor is a MuJoCo heightfield (`go2_warp_scene_rough.xml`); its elevation
grid (`hfield_data`) is filled per config in `_customize_mj_model`. Two profiles
(toggle via `rough_profile`):
  - "perlin_hf" : higher-frequency fractal-Perlin — smooth rounded foot-scale
                  bumps (the "A" preview). Borrowed from mjlab's perlin gen.
  - "uniform"   : per-cell random uniform — jagged foot-scale bumps (the harsh
                  "uni7" preview). Matches mjlab's random_rough.
  - "smooth_uniform" : uniform + light gaussian blur — rounded random bumps
                  (the "between" option).

MJX-Warp heightfield collision is confirmed working (the Go2 rests on the bumps;
hfield CCD kernels just take a one-time ~9 s JIT compile).
"""
import numpy as np
from ml_collections import config_dict
from scipy import ndimage

from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion.go2_warp_joystick import (
    WarpJoystick, default_config as joystick_default_config,
)
from jax_rl.envs.locomotion.go2_warp_osc_joystick import (
    WarpOscJoystick, default_config as osc_default_config,
)
from jax_rl.envs.locomotion.go2_warp_osc_var_impedance import (
    WarpOscVarImpedance,
    default_config as var_default_config,
    default_config_per_axis as var_axis_default_config,
)

ROUGH_SCENE = consts.ROOT_PATH / "go2_warp_scene_rough.xml"


# ── Noise generators ─────────────────────────────────────────────────────────
def _norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _fractal_perlin_noise_2d(nx, ny, rng, octaves=4, persistence=0.5,
                             lacunarity=2.0, scale=14.0):
    """Borrowed from mjlab terrains/heightfield_terrains.py (pure numpy)."""
    def lerp(a, b, x): return a + x * (b - a)
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def grad(h, x, y):
        h = h % 4
        return np.where(h == 0, x + y, np.where(h == 1, x - y,
                        np.where(h == 2, -x + y, -x - y)))
    def perlin(x, y, p):
        xi = x.astype(int) % 256; yi = y.astype(int) % 256
        xf = x - x.astype(int); yf = y - y.astype(int)
        u = fade(xf); v = fade(yf)
        n00 = grad(p[p[xi] + yi], xf, yf)
        n01 = grad(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = grad(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = grad(p[p[xi + 1] + yi], xf - 1, yf)
        return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
    p = np.arange(256, dtype=int); rng.shuffle(p); p = np.stack([p, p]).flatten()
    noise = np.zeros((nx, ny)); amp = 1.0; freq = scale; tot = 0.0
    xx, yy = np.meshgrid(np.linspace(0, nx, nx, endpoint=False),
                         np.linspace(0, ny, ny, endpoint=False), indexing="ij")
    for _ in range(octaves):
        noise += amp * perlin(xx * freq / nx, yy * freq / ny, p)
        tot += amp; amp *= persistence; freq *= lacunarity
    return noise / tot


def _make_heightfield(profile, nrow, ncol, seed):
    rng = np.random.default_rng(seed)
    if profile == "perlin_hf":
        return _norm01(_fractal_perlin_noise_2d(nrow, ncol, np.random.default_rng(seed)))
    if profile == "uniform":
        return rng.uniform(0.0, 1.0, (nrow, ncol))
    if profile == "smooth_uniform":
        return _norm01(ndimage.gaussian_filter(rng.uniform(0, 1, (nrow, ncol)), 1.5))
    raise ValueError(f"unknown rough_profile {profile!r}")


# ── Rough-heightfield mixin ──────────────────────────────────────────────────
class _RoughHFMixin:
    """Loads the rough scene and fills hfield_data from config (profile, amp)."""
    _scene_xml = ROUGH_SCENE

    def _customize_mj_model(self) -> None:
        m = self._mj_model
        hid = m.hfield("rough").id
        nrow = int(m.hfield_nrow[hid]); ncol = int(m.hfield_ncol[hid])
        zmax = float(m.hfield_size[hid, 2])
        profile = str(getattr(self._config, "rough_profile", "uniform"))
        amp = float(getattr(self._config, "rough_amplitude", 0.07))
        seed = int(getattr(self._config, "rough_seed", 0))
        noise01 = _make_heightfield(profile, nrow, ncol, seed)            # [0,1]
        data = (noise01 * (amp / zmax)).astype(np.float32)               # scale to amp
        adr = int(m.hfield_adr[hid])
        m.hfield_data[adr:adr + nrow * ncol] = data.flatten()


def _add_rough(cfg, profile, amplitude, seed=0):
    cfg.rough_profile = profile
    cfg.rough_amplitude = amplitude
    cfg.rough_seed = seed
    return cfg


# ── Config factories (gains matched to the flat runs for zero-shot) ──────────
def joint_rough_config(profile="uniform", amp=0.07):
    return _add_rough(joystick_default_config(), profile, amp)


def osc_soft_rough_config(profile="uniform", amp=0.07):
    cfg = osc_default_config()
    cfg.osc.kp = [1500.0, 1500.0, 2000.0]   # s=0.5 (matches flat fixed-soft)
    cfg.osc.kd = [78.0, 78.0, 92.0]
    return _add_rough(cfg, profile, amp)


def var_rough_config(profile="uniform", amp=0.07):
    return _add_rough(var_default_config(), profile, amp)        # per_foot, baseline kp


def var_axis_rough_config(profile="uniform", amp=0.07):
    return _add_rough(var_axis_default_config(), profile, amp)


# ── Envs ─────────────────────────────────────────────────────────────────────
class WarpRoughHF(_RoughHFMixin, WarpJoystick):
    """Joint-PD on a rough heightfield."""


class WarpOscRoughHF(_RoughHFMixin, WarpOscJoystick):
    """Fixed-impedance OSC on a rough heightfield."""


class WarpOscVarRoughHF(_RoughHFMixin, WarpOscVarImpedance):
    """Variable-impedance OSC on a rough heightfield (per_foot or per_axis)."""
