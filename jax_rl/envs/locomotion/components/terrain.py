"""Terrain component — shapes the scene floor (Flat | RoughHF). One of the four
orthogonal axes the Go2 Warp env delegates to (see the ``components`` package
docstring). Logic moved VERBATIM from the former ``go2_warp_components`` — no
behaviour change.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy import ndimage


# Heightfield noise generators (moved verbatim from the deleted rough-config
# module's private helpers).
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


class Terrain:
    """Shapes the scene floor. `apply(spec)` mutates the MjSpec before compile
    (structural: geom type + hfield asset); `customize_model(mj_model)` runs
    after compile (data: hfield elevation). Either may be a no-op."""

    def apply(self, spec) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def customize_model(self, mj_model) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Flat(Terrain):
    """Flat plane floor — already declared by the base scene; nothing to do."""

    def apply(self, spec) -> None:
        pass

    def customize_model(self, mj_model) -> None:
        pass


class RoughHF(Terrain):
    """Procedural rough heightfield floor.

    `apply` adds the hfield asset and switches the base scene's ``floor`` geom to
    it IN THE SPEC — so the second scene XML (``go2_warp_scene_rough.xml``) is no
    longer needed. MjSpec requires non-empty ``userdata`` to compile an hfield AND
    renormalizes it to [0,1] (rescaling the amplitude). So `apply` sets the
    elevation as a placeholder (lets compile succeed) and `customize_model`
    re-writes the EXACT elevation into ``hfield_data`` AFTER compile — overwriting
    the renormalized values, giving a surface BIT-IDENTICAL to the former
    ``_RoughHFMixin`` post-compile poke. nrow/ncol/size match the old rough XML.
    """

    NROW = NCOL = 160                # matches go2_warp_scene_rough.xml <hfield>
    SIZE = (5.0, 5.0, 0.10, 0.5)     # x/y half-extent, max height, base thick (m)

    def __init__(self, profile: str = "uniform", amplitude: float = 0.07,
                 seed: int = 0):
        self.profile = profile
        self.amplitude = amplitude
        self.seed = seed

    def _elevation(self, nrow, ncol, zmax) -> np.ndarray:
        noise01 = _make_heightfield(self.profile, nrow, ncol, self.seed)   # [0,1]
        return (noise01 * (self.amplitude / zmax)).astype(np.float32)      # scale to amp

    def apply(self, spec) -> None:
        hf = spec.add_hfield()
        hf.name = "rough"
        hf.nrow, hf.ncol = self.NROW, self.NCOL
        hf.size = list(self.SIZE)
        # Placeholder elevation so compile succeeds; customize_model overwrites it
        # post-compile with the exact (un-renormalized) values.
        data = self._elevation(self.NROW, self.NCOL, self.SIZE[2])
        hf.userdata = data.flatten().tolist()
        floor = next(g for g in spec.worldbody.geoms if g.name == "floor")
        floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
        floor.hfieldname = "rough"

    def customize_model(self, mj_model) -> None:
        m = mj_model
        hid = m.hfield("rough").id
        nrow = int(m.hfield_nrow[hid]); ncol = int(m.hfield_ncol[hid])
        zmax = float(m.hfield_size[hid, 2])
        data = self._elevation(nrow, ncol, zmax)
        adr = int(m.hfield_adr[hid])
        m.hfield_data[adr:adr + nrow * ncol] = data.flatten()


def terrain_from_config(config) -> Terrain:
    """Build the Terrain from the legacy config keys (`rough_profile`,
    `rough_amplitude`, `rough_seed`). No `rough_profile` set → Flat. Bridges the
    old config-driven construction to the component model during migration."""
    profile = getattr(config, "rough_profile", None)
    if profile is None:
        return Flat()
    return RoughHF(
        profile=str(profile),
        amplitude=float(getattr(config, "rough_amplitude", 0.07)),
        seed=int(getattr(config, "rough_seed", 0)),
    )
