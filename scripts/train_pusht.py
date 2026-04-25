"""Train FastSAC on vendored PushTEnv (gym-pusht + modular rewards).

Single-process parallel gym vector env + FastSAC. gym-pusht is CPU-only
(pymunk), so we can't vmap physics. We run N parallel envs via
gym.vector.SyncVectorEnv and batch the single-step transitions into a
numpy replay buffer.

Rough throughput: ~2-4k sps per env, N=8 parallel ≈ 15-25k sps aggregate.
Training ~1M steps takes 1-2 hours.

Usage:
    uv run python train_pusht.py --reward-mode dense --total-timesteps 1000000
    uv run python train_pusht.py --reward-mode coverage --total-timesteps 500000 --wandb
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.envs.manipulation.pusht import PushTEnv


# ═════════════════════════════════════════════════════════════════════
# Action scaling: FastSAC outputs [-1, 1]² via tanh → gym-pusht [0, 512]²
# ═════════════════════════════════════════════════════════════════════

_GYM_LOW = 0.0
_GYM_HIGH = 512.0


def policy_to_gym_action(action_pm1: np.ndarray) -> np.ndarray:
    """[-1, 1] → [0, 512]."""
    return (action_pm1 + 1.0) * 0.5 * (_GYM_HIGH - _GYM_LOW) + _GYM_LOW


# ═════════════════════════════════════════════════════════════════════
# Simple numpy replay buffer
# ═════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.truncations = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, action, reward, next_obs, done, truncation):
        n = obs.shape[0]
        idx = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward.reshape(-1, 1)
        self.dones[idx] = done.reshape(-1, 1)
        self.truncations[idx] = truncation.reshape(-1, 1)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=batch_size)
        batch = {
            "obs": jnp.asarray(self.obs[idx]),
            "next_obs": jnp.asarray(self.next_obs[idx]),
            "action": jnp.asarray(self.actions[idx]),
            "reward": jnp.asarray(self.rewards[idx]),
            "done": jnp.asarray(self.dones[idx]),
            "truncation": jnp.asarray(self.truncations[idx]),
        }
        # FastSAC expects critic_obs / critic_next_obs — alias to obs (symmetric).
        batch["critic_obs"] = batch["obs"]
        batch["critic_next_obs"] = batch["next_obs"]
        return batch


# ═════════════════════════════════════════════════════════════════════
# Vectorized gym env builder
# ═════════════════════════════════════════════════════════════════════

class NormalizeObsWrapper(gym.ObservationWrapper):
    """Rescale raw pixel-coord obs [0, 512] → roughly [-1, 1].

    gym-pusht returns agent_xy and block_xy in [0, 512] pixel coords and
    angles in [0, 2π]. Raw values feed 100+ magnitudes into the network,
    which destabilizes SAC's critic. Rescale so obs ≈ zero-centered unit-scale.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        self._center = (low + high) / 2.0
        self._scale = (high - low) / 2.0 + 1e-6
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=low.shape, dtype=np.float32,
        )

    def observation(self, obs):
        return ((np.asarray(obs, dtype=np.float32) - self._center) / self._scale).astype(np.float32)


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat action K times, accumulate reward, return last obs.

    Implicit temporal extension — critical for multi-contact manipulation
    where each "decision" should cover multiple physics steps of committed
    motion (FiGAR-style, Atari frame-skip).
    """
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k

    def step(self, action):
        total_r = 0.0
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += float(r)
            if term or trunc:
                break
        return obs, total_r, term, trunc, info


def _make_env(reward_mode: str, obs_type: str, frame_stack: int = 1, action_repeat: int = 1,
               max_episode_steps: int = 300, coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee"):
    env = PushTEnv(obs_type=obs_type, reward_mode=reward_mode, render_mode="rgb_array",
                   coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
    # CRITICAL: gym.make("gym_pusht/PushT-v0") auto-wraps with TimeLimit(300)
    # but direct PushTEnv(...) does not. Without this, failed episodes run
    # indefinitely, SAC target Q bootstraps infinite future, critic explodes.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if obs_type == "environment_state_agent_pos":
        env = gym.wrappers.FlattenObservation(env)
    env = NormalizeObsWrapper(env)
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, k=action_repeat)
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        env = gym.wrappers.FlattenObservation(env)
    return env


def make_vec_env(n_envs: int, reward_mode: str, obs_type: str = "state",
                  frame_stack: int = 1, action_repeat: int = 1, coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee"):
    def make_single():
        def _thunk():
            return _make_env(reward_mode, obs_type, frame_stack, action_repeat,
                             coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
        return _thunk
    return gym.vector.SyncVectorEnv([make_single() for _ in range(n_envs)])


def make_eval_env(reward_mode: str, obs_type: str = "state",
                  frame_stack: int = 1, action_repeat: int = 1, coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee"):
    return _make_env(reward_mode, obs_type, frame_stack, action_repeat,
                     coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)


# ═════════════════════════════════════════════════════════════════════
# Eval
# ═════════════════════════════════════════════════════════════════════

def _rollout_once(algo, actor_params, reward_mode, obs_type, frame_stack, action_repeat, seed, max_steps, deterministic, key, coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee"):
    env = make_eval_env(reward_mode, obs_type, frame_stack, action_repeat, coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
    obs, _ = env.reset(seed=seed)
    ep_r = 0.0
    info = {}
    for _ in range(max_steps):
        key, ak = jax.random.split(key)
        a_pm1 = np.asarray(algo.select_action(
            actor_params, jnp.asarray(obs[None]),
            ak, deterministic=deterministic,
        ))[0]
        obs, r, term, trunc, info = env.step(policy_to_gym_action(a_pm1))
        ep_r += float(r)
        if term or trunc:
            break
    env.close()
    return ep_r, float(info.get("coverage", 0.0)), bool(info.get("is_success", False))


def evaluate(algo, actor_params, reward_mode: str, obs_type: str = "state",
             frame_stack: int = 1, action_repeat: int = 1,
             n_episodes: int = 5, max_steps: int = 300, coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee") -> dict:
    # max_steps is in ENV-action units; divide when using action_repeat.
    max_policy_steps = max_steps // max(action_repeat, 1)
    det_ret, det_cov, det_succ = [], [], []
    sto_ret, sto_cov, sto_succ = [], [], []
    key = jax.random.PRNGKey(7777)
    for i in range(n_episodes):
        r, c, s = _rollout_once(algo, actor_params, reward_mode, obs_type, frame_stack, action_repeat,
                                 1000 + i, max_policy_steps, True, key, coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
        det_ret.append(r); det_cov.append(c); det_succ.append(s)
        key, sk = jax.random.split(key)
        r, c, s = _rollout_once(algo, actor_params, reward_mode, obs_type, frame_stack, action_repeat,
                                 1000 + i, max_policy_steps, False, sk, coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
        sto_ret.append(r); sto_cov.append(c); sto_succ.append(s)
    return {
        "eval_return_mean": float(np.mean(det_ret)),
        "eval_return_std": float(np.std(det_ret)),
        "eval_coverage_mean": float(np.mean(det_cov)),
        "eval_success_rate": float(np.mean(det_succ)),
        "eval_sto_return_mean": float(np.mean(sto_ret)),
        "eval_sto_coverage_mean": float(np.mean(sto_cov)),
        "eval_sto_success_rate": float(np.mean(sto_succ)),
    }


# ═════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════

def train(
    reward_mode: str = "dense",
    total_timesteps: int = 1_000_000,
    num_envs: int = 8,
    buffer_size: int = 500_000,
    min_buffer_size: int = 4_000,
    batch_size: int = 512,
    grad_updates_per_step: int = 1,
    lr: float = 3e-4,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
    grad_clip_norm: float | None = None,
    target_entropy_scale: float = 1.0,
    obs_type: str = "state",
    frame_stack: int = 1,
    action_repeat: int = 1,
    coverage_shape: str = "linear", coverage_eps: float = 0.01, success_threshold: float = 0.95, success_bonus: float = 50.0, block_shape: str = "tee",
    seed: int = 0,
    eval_every_n_steps: int = 50_000,
    ckpt_dir: str | None = None,
    use_wandb: bool = False,
):
    print("=" * 70)
    print(f"FastSAC on PushTEnv (vendored gym-pusht)")
    print("=" * 70)
    print(f"  reward_mode={reward_mode}  total_steps={total_timesteps:,}")
    print(f"  num_envs={num_envs}  batch_size={batch_size}  buffer={buffer_size:,}")
    print(f"  lr={lr}  gamma={gamma}  seed={seed}")
    print()

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # Env
    env = make_vec_env(num_envs, reward_mode, obs_type, frame_stack, action_repeat, coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
    print(f"  obs_type={obs_type}  frame_stack={frame_stack}  action_repeat={action_repeat}  "
          f"obs_dim={int(env.single_observation_space.shape[0])}")
    obs_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])

    # Vanilla SAC — scalar Q head, no v_min/v_max constraints.
    # Better suited for bounded low-horizon tasks like push-T than FastSAC
    # (whose C51 distributional critic is tuned for unbounded locomotion Q).
    algo_cfg = SACConfig(
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        batch_size=batch_size,
        grad_updates_per_step=grad_updates_per_step,
        hidden_dim=(256, 256),
        critic_hidden_dim=(256, 256),
        tau=0.005,
        target_entropy_scale=target_entropy_scale,
    )
    # Optional gradient clipping (helps with early critic blowup on
    # variable-scale reward signals like contact_gated).
    if grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr),
        )
    else:
        optimizer = optax.adam(lr)
    alpha_optimizer = optax.adam(lr)
    algo = SAC(algo_cfg, obs_dim, action_dim, optimizer, alpha_optimizer, gamma=gamma)

    # Init training state via algo's factory
    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    # Buffer
    buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

    # Explore: sample stochastic action from actor
    @jax.jit
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key, deterministic=False)

    @jax.jit
    def jit_update(training_state, batch):
        return algo.update(training_state, batch)

    # Checkpoint dir
    if ckpt_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = f"checkpoints/{ts}_pusht_sac_{reward_mode}_seed{seed}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"  ckpt_dir={ckpt_dir}")

    # W&B
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="jax-rl",
            name=f"pusht_sac_{reward_mode}_seed{seed}",
            config={"reward_mode": reward_mode, "total_timesteps": total_timesteps,
                    "num_envs": num_envs, "batch_size": batch_size,
                    "buffer_size": buffer_size, "lr": lr, "gamma": gamma, "seed": seed},
        )

    # Rollout
    obs, _ = env.reset(seed=seed)
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    best_eval = -float("inf")
    t0 = time.time()

    for step in range(0, total_timesteps, num_envs):
        # Action selection
        if buffer.size < min_buffer_size:
            # Warmup: random uniform in [-1, 1]
            action_pm1 = rng.uniform(-1, 1, size=(num_envs, action_dim)).astype(np.float32)
        else:
            key, exp_key = jax.random.split(key)
            action_pm1 = np.asarray(explore(training_state.actor_params, jnp.asarray(obs), exp_key))

        # Map to gym action
        action_gym = policy_to_gym_action(action_pm1).astype(np.float32)

        # Step
        next_obs, reward, term, trunc, info = env.step(action_gym)
        done = term.astype(np.float32)
        truncation = trunc.astype(np.float32)
        ep_returns += reward.astype(np.float32)

        # In vector env, auto-reset; info["final_observation"] contains pre-reset obs.
        # Use next_obs as-is (already post-reset if done), which is correct for SAC.
        buffer.add_batch(obs, action_pm1,
                         (reward_scale * reward.astype(np.float32)),
                         next_obs, done, truncation)

        # Track episode ends
        any_done = (term | trunc)
        if any_done.any():
            for i in np.where(any_done)[0]:
                ep_returns[i] = 0.0

        obs = next_obs

        # Train
        if buffer.size >= min_buffer_size:
            for _ in range(grad_updates_per_step):
                batch = buffer.sample(batch_size, rng)
                training_state, metrics = jit_update(training_state, batch)

        # Log
        if (step // num_envs) % 100 == 0:
            elapsed = time.time() - t0
            sps = int((step + num_envs) / elapsed) if elapsed > 0 else 0
            if buffer.size >= min_buffer_size:
                q1 = float(metrics["q1_mean"])
                ent = float(metrics.get("entropy", 0))
                alpha = float(metrics.get("alpha", 0))
                actor_loss = float(metrics.get("actor_loss", 0))
                q1_loss = float(metrics.get("q1_loss", 0))
                recent_r_mean = float(ep_returns.mean())
                print(f"Step {step+num_envs:>10,} | Q1={q1:+7.3f} "
                      f"Q1L={q1_loss:.2e} AL={actor_loss:+6.2f} "
                      f"Ent={ent:+.2f} Alpha={alpha:.4f} "
                      f"ep_r_avg={recent_r_mean:+.2f} | {sps:,} sps | {elapsed:.0f}s")
                if wandb_run:
                    wandb_run.log({"step": step + num_envs, "sps": sps,
                                   **{k: float(v) for k, v in metrics.items() if not isinstance(v, (dict, tuple))}})
            else:
                print(f"Step {step+num_envs:>10,} | buffer={buffer.size:>7,}/{buffer_size:,} | "
                      f"warming up | {sps:,} sps | {elapsed:.0f}s")

        # Eval
        if (step + num_envs) % eval_every_n_steps < num_envs and buffer.size >= min_buffer_size:
            eval_stats = evaluate(algo, training_state.actor_params, reward_mode, obs_type, frame_stack, action_repeat, coverage_shape=coverage_shape, coverage_eps=coverage_eps, success_threshold=success_threshold, success_bonus=success_bonus, block_shape=block_shape)
            print(f"  EVAL @ step {step+num_envs:,}: "
                  f"det: r={eval_stats['eval_return_mean']:.2f}±{eval_stats['eval_return_std']:.2f} "
                  f"cov={eval_stats['eval_coverage_mean']:.3f} "
                  f"s={eval_stats['eval_success_rate']*100:.0f}% "
                  f"| sto: r={eval_stats['eval_sto_return_mean']:.2f} "
                  f"cov={eval_stats['eval_sto_coverage_mean']:.3f} "
                  f"s={eval_stats['eval_sto_success_rate']*100:.0f}%")
            if wandb_run:
                wandb_run.log({"step": step + num_envs, **eval_stats})

            if eval_stats["eval_return_mean"] > best_eval:
                best_eval = eval_stats["eval_return_mean"]
                np.save(Path(ckpt_dir) / "actor_params_best.npy",
                        jax.device_get(training_state.actor_params),
                        allow_pickle=True)
                print(f"    ↑ new best eval, saved.")

    env.close()

    # Final save
    np.save(Path(ckpt_dir) / "actor_params_final.npy",
            jax.device_get(training_state.actor_params),
            allow_pickle=True)
    print(f"\nTraining complete. Final checkpoint: {ckpt_dir}")
    print(f"  Best eval return: {best_eval:.2f}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", type=str, default="dense",
                    choices=["coverage", "sparse", "shaped", "approach", "dense", "contact_gated"])
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--grad-updates-per-step", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reward-scale", type=float, default=1.0,
                    help="Multiplier on env reward before replay. Use 0.1 for contact_gated.")
    ap.add_argument("--grad-clip-norm", type=float, default=None,
                    help="Global grad norm clip (e.g. 1.0). None = off.")
    ap.add_argument("--target-entropy-scale", type=float, default=1.0,
                    help="SAC target entropy = -scale * action_dim. Bigger = more explore.")
    ap.add_argument("--obs-type", type=str, default="state",
                    choices=["state", "keypoints", "environment_state_agent_pos"],
                    help="state=5d; keypoints=25d (5d state + 10 dense arc-length KPs per shape); "
                         "environment_state_agent_pos=18d (T-only keypoints + agent).")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="Stack N consecutive obs. Implicit velocity; flattened to obs_dim × N.")
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="Repeat each action K env steps (frame skip). Commits policy to direction, "
                         "classic RL trick for multi-contact manipulation (FiGAR / Atari frame skip).")
    ap.add_argument("--coverage-shape", type=str, default="linear",
                    choices=["linear", "log_barrier"],
                    help="r_coverage shape. 'linear' = raw coverage. 'log_barrier' = "
                         "-log(1 - cov + eps): unbounded near goal, amplifies final-mile precision.")
    ap.add_argument("--coverage-eps", type=float, default=0.01,
                    help="Epsilon for log_barrier (sets max reward ceiling: ε=0.01 → r_max≈4.6).")
    ap.add_argument("--success-threshold", type=float, default=0.95,
                    help="Coverage threshold for terminated=True. DP paper uses 0.95 (above "
                         "human teleop peak 0.9489). Lower to 0.85 for tractable success events.")
    ap.add_argument("--success-bonus", type=float, default=50.0,
                    help="Terminal reward on success (contact_gated mode only). Default 50.")
    ap.add_argument("--block-shape", type=str, default="tee",
                    choices=["tee", "l", "k", "s", "ellipse", "triangle", "dr"],
                    help="Block shape. 'dr' samples per episode from letter set {tee, l, k, s}.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every-n-steps", type=int, default=50_000)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    train(
        reward_mode=args.reward_mode,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        grad_updates_per_step=args.grad_updates_per_step,
        lr=args.lr,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        grad_clip_norm=args.grad_clip_norm,
        target_entropy_scale=args.target_entropy_scale,
        obs_type=args.obs_type,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        coverage_shape=args.coverage_shape,
        coverage_eps=args.coverage_eps,
        success_threshold=args.success_threshold,
        success_bonus=args.success_bonus,
        block_shape=args.block_shape,
        seed=args.seed,
        eval_every_n_steps=args.eval_every_n_steps,
        use_wandb=args.wandb,
    )
