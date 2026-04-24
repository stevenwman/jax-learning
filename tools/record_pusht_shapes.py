"""Record minimal_logbar policy rollouts on 4 new block shapes + T.

Vibes test — no coverage metric validity (goal polygon is shape-matched but
identity pose calibration is T-specific). Observes shape-agnostic 5d state
and lets the policy zero-shot push each shape toward its goal pose.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pygame
import pymunk
from PIL import Image

from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.envs.manipulation.pusht import PushTEnv
from train_pusht import _make_env, policy_to_gym_action

MASK = pymunk.ShapeFilter.ALL_MASKS()


def _make_poly_shape(body, verts, color):
    s = pymunk.Poly(body, verts)
    s.color = pygame.Color(color)
    s.filter = pymunk.ShapeFilter(mask=MASK)
    return s


def _multi_poly(space, polys, position, angle, color):
    inertia_total = sum(pymunk.moment_for_poly(1, vertices=p) for p in polys)
    body = pymunk.Body(1, inertia_total)
    shapes = [_make_poly_shape(body, p, color) for p in polys]
    body.angle, body.position = angle, position
    space.add(body, *shapes)
    return body, shapes


def _ring_polys(center, inner_r, outer_r, theta_start, theta_end, n):
    cx, cy = center
    thetas = np.linspace(theta_start, theta_end, n + 1)
    polys = []
    for i in range(n):
        t0, t1 = thetas[i], thetas[i + 1]
        polys.append([
            (cx + inner_r * np.cos(t0), cy + inner_r * np.sin(t0)),
            (cx + outer_r * np.cos(t0), cy + outer_r * np.sin(t0)),
            (cx + outer_r * np.cos(t1), cy + outer_r * np.sin(t1)),
            (cx + inner_r * np.cos(t1), cy + inner_r * np.sin(t1)),
        ])
    return polys


# ── Shape builders — must match signature: (space, position, angle, ...) → (body, shapes)

def _add_ellipse(space, position, angle, semi_major=60, semi_minor=40, color="LightSlateGray"):
    n = 32
    verts = [(semi_major * np.cos(t), semi_minor * np.sin(t))
             for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    inertia = pymunk.moment_for_poly(1, vertices=verts)
    body = pymunk.Body(1, inertia)
    shape = _make_poly_shape(body, verts, color)
    body.angle, body.position = angle, position
    space.add(body, shape)
    return body, [shape]


def _add_triangle(space, position, angle, color="LightSlateGray"):
    verts = [(-60, -40), (60, -40), (0, 40)]
    inertia = pymunk.moment_for_poly(1, vertices=verts)
    body = pymunk.Body(1, inertia)
    shape = _make_poly_shape(body, verts, color)
    body.angle, body.position = angle, position
    space.add(body, shape)
    return body, [shape]


def _add_s(space, position, angle, color="LightSlateGray", n_per_ring=10):
    s = 1.953125 * 0.7
    block_height = (38 - 22) * s
    up_shift = block_height * 3 / 14
    arc_trim = 30
    upper = _ring_polys(
        center=(0, 27 * s + up_shift), inner_r=22 * s, outer_r=38 * s,
        theta_start=np.radians(255),
        theta_end=np.radians(195 + 360 - arc_trim),
        n=n_per_ring - 1,
    )
    lower = _ring_polys(
        center=(0, -27 * s), inner_r=22 * s, outer_r=38 * s,
        theta_start=np.radians(75),
        theta_end=np.radians(15 + 360 - arc_trim),
        n=n_per_ring - 1,
    )
    return _multi_poly(space, upper + lower, position, angle, color)


def _add_u(space, position, angle, color="LightSlateGray", n_ring=10):
    s = 1.367
    inner_r = 22 * s
    outer_r = 38 * s
    ring = _ring_polys(
        center=(0, 0), inner_r=inner_r, outer_r=outer_r,
        theta_start=np.radians(180), theta_end=np.radians(360),
        n=n_ring,
    )
    arm_top = 50 * s
    left_arm = [(-outer_r, 0), (-inner_r, 0), (-inner_r, arm_top), (-outer_r, arm_top)]
    right_arm = [(inner_r, 0), (outer_r, 0), (outer_r, arm_top), (inner_r, arm_top)]
    return _multi_poly(space, ring + [left_arm, right_arm], position, angle, color)


SHAPE_BUILDERS = {
    "tee":      PushTEnv.add_tee,
    "ellipse":  _add_ellipse,
    "triangle": _add_triangle,
    "s":        _add_s,
    "u":        _add_u,
}


class _ShapedPushTEnv(PushTEnv):
    """PushTEnv with swappable block shape via class attribute `_shape_builder`.

    Coverage stubbed to 0 — shapely chokes on unioning overlapping convex pieces
    in S/U rings. Vibes-only: policy doesn't use coverage for action selection.
    """
    _shape_builder = PushTEnv.add_tee

    def _get_coverage(self):
        try:
            return super()._get_coverage()
        except Exception:
            return 0.0

    def _setup(self):
        # Mirror parent _setup but use custom shape builder instead of add_tee.
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = None
        self.render_buffer = []

        from jax_rl.envs.manipulation.pusht.pusht import PushTEnv as _B
        # Re-run original setup (walls + agent + collision handler), then swap block.
        _B._setup(self)
        # Remove T block added by _setup.
        self.space.remove(self.block, *self.block.shapes)
        # Add new shape at same spawn location.
        self.block, self._block_shapes = self.__class__._shape_builder(
            self.space, (256, 300), 0,
        )


def _make_shaped_env(shape_name):
    """Build env for the named shape, matching train_pusht.py's wrapper stack."""
    # Subclass per shape — class attribute picks the builder at _setup time.
    env_cls = type(f"PushTEnv_{shape_name}", (_ShapedPushTEnv,),
                   {"_shape_builder": staticmethod(SHAPE_BUILDERS[shape_name])})
    import gymnasium as gym
    from train_pusht import NormalizeObsWrapper, ActionRepeatWrapper
    env = env_cls(obs_type="state", reward_mode="contact_gated", render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
    env = NormalizeObsWrapper(env)
    env = ActionRepeatWrapper(env, k=2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints/20260420_133136_pusht_sac_contact_gated_seed0/actor_params_best.npy")
    ap.add_argument("--shapes", nargs="+",
                    default=["tee", "ellipse", "triangle", "s", "u"])
    ap.add_argument("--n-episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out-dir", type=str, default=".temp")
    args = ap.parse_args()

    actor_params = np.load(args.ckpt, allow_pickle=True).item()

    algo = SAC(SACConfig(hidden_dim=(256, 256), critic_hidden_dim=(256, 256)),
               obs_dim=5, action_dim=2,
               optimizer=optax.adam(1e-3), alpha_optimizer=optax.adam(1e-3))

    @jax.jit
    def act(params, obs, key):
        return algo.select_action(params, obs, key, deterministic=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)

    for shape in args.shapes:
        print(f"\n== shape={shape} ==")
        env = _make_shaped_env(shape)
        render_env = _ShapedPushTEnv(
            obs_type="state", reward_mode="contact_gated", render_mode="rgb_array",
        )
        type(render_env)._shape_builder = staticmethod(SHAPE_BUILDERS[shape])

        for ep in range(args.n_episodes):
            render_env.reset(seed=1000 + ep)
            obs, _ = env.reset(seed=1000 + ep)
            frames = [render_env.render()]
            for _ in range(args.max_steps):
                key, ak = jax.random.split(key)
                a_pm1 = np.asarray(act(actor_params, jnp.asarray(obs[None]), ak))[0]
                a_gym = policy_to_gym_action(a_pm1).astype(np.float32)
                obs, r, term, trunc, info = env.step(a_gym)
                for _ in range(2):
                    _, _, rt, ru, _ = render_env.step(a_gym)
                    frames.append(render_env.render())
                    if rt or ru:
                        break
                if term or trunc:
                    break
            cov = info.get("coverage", 0.0)
            print(f"  ep {ep}: cov={cov:.3f}  frames={len(frames)}")

            tmp = out / f"_pusht_{shape}_ep{ep}"
            tmp.mkdir(exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(tmp / f"{i:04d}.png")
            mp4 = out / f"pusht_shape_{shape}_ep{ep}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "20",
                "-i", str(tmp / "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(mp4),
            ], check=True, capture_output=True)
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()
        env.close()
        render_env.close()


if __name__ == "__main__":
    main()
