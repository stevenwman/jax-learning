"""MPPI planner: trajectory rollout, elite weighting, action selection, batched plan_fn.

Implements the Model Predictive Path Integral (MPPI) planner used at action-time.
Network modules (Dynamics, Reward, QEnsemble, PolicyPrior) are passed as arguments —
no import dependency on jax_rl.algos.tdmpc2.networks."""
import jax
import jax.numpy as jnp

from jax_rl.utils.twohot import two_hot_inv


# ------------------ MPPI core ------------------


def mppi_rollout(
    plan_params,
    z_0: jax.Array,           # (N, latent_dim) — N = num_samples per env
    actions_seq: jax.Array,   # (horizon, N, action_dim)
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> jax.Array:
    """Score N candidate action trajectories by rolling the world model forward.

    Score formula (source tdmpc2.py:128-136):
      score(τ) = Σ_{h=0..H-1} γ^h · r̂(z_h, a_h) + γ^H · Q_avg_of_2(z_H, π(z_H))

    Reward/Q decoded through two_hot_inv(apply_symexp=True).
    Online Q ensemble used (NOT target Q — MPPI is inference-time).

    Returns: (N,) predicted returns.
    """
    N = z_0.shape[0]
    gamma = cfg.discount

    def step(carry, a):
        z, discount_factor, G = carry
        r_logits = reward_net.apply(plan_params["reward"], z, a)
        r_probs = jax.nn.softmax(r_logits, axis=-1)
        r_hat = two_hot_inv(
            r_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
        ).squeeze(-1)  # (N,)
        G_next = G + discount_factor * r_hat
        z_next = dynamics.apply(plan_params["dynamics"], z, a)
        return (z_next, discount_factor * gamma, G_next), None

    initial_carry = (z_0, jnp.ones(N), jnp.zeros(N))
    (z_final, discount_final, G_reward), _ = jax.lax.scan(step, initial_carry, actions_seq)

    # Terminal Q bootstrap: γ^H · Q_avg_of_2(z_final, π(z_final))
    key_pi, key_q = jax.random.split(key, 2)
    a_terminal, _ = policy_net.apply(plan_params["policy"], z_final, key_pi)
    q_logits = q_ensemble_net.apply(
        plan_params["q_ensemble"], z_final, a_terminal, deterministic=True,
    )  # (num_q, N, num_bins)
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    q_selected = q_logits[perm]  # (2, N, num_bins)
    q_probs = jax.nn.softmax(q_selected, axis=-1)
    q_decoded = two_hot_inv(
        q_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
    ).squeeze(-1)  # (2, N)
    q_terminal = q_decoded.mean(axis=0)  # (N,)

    return G_reward + discount_final * q_terminal


def mppi_iteration(
    mean: jax.Array,          # (horizon, action_dim)
    std: jax.Array,           # (horizon, action_dim)
    plan_params,
    z_0: jax.Array,           # (latent_dim,) — single env
    pi_trajs: jax.Array,      # (horizon, num_pi_trajs, action_dim)
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One MPPI iteration: sample, score, select elites, update (mean, std).

    Args:
        mean, std: current sampling distribution, shape (horizon, action_dim).
        pi_trajs: (horizon, num_pi_trajs, action_dim) — policy-seeded samples (from F2).
        z_0: single-env starting latent.

    Returns (new_mean, new_std, elite_actions, weights):
        new_mean, new_std: (horizon, action_dim)
        elite_actions: (horizon, num_elites, action_dim) — kept for F4 Gumbel action selection
        weights: (num_elites,) softmax-over-scores
    """
    key_sample, key_rollout = jax.random.split(key, 2)

    # Gaussian samples: i.i.d. across (horizon, num_samples - num_pi_trajs, action_dim)
    N_gauss = cfg.num_samples - cfg.num_pi_trajs
    eps = jax.random.normal(key_sample, (cfg.horizon, N_gauss, cfg.action_dim))
    gauss_actions = jnp.clip(
        mean[:, None, :] + std[:, None, :] * eps,
        -1.0, 1.0,
    )
    # Stack: [pi_trajs, gauss_actions] → (horizon, num_samples, action_dim)
    actions = jnp.concatenate([pi_trajs, gauss_actions], axis=1)

    # Broadcast z_0 to (num_samples, latent_dim) and score
    z_0_broadcast = jnp.broadcast_to(z_0, (cfg.num_samples,) + z_0.shape)
    scores = mppi_rollout(
        plan_params, z_0_broadcast, actions, cfg, key_rollout,
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble_net, policy_net=policy_net,
    )  # (num_samples,)

    # Elite selection: top-K by score
    _, elite_idx = jax.lax.top_k(scores, cfg.num_elites)  # (num_elites,)
    elite_scores = scores[elite_idx]
    elite_actions = actions[:, elite_idx, :]  # (horizon, num_elites, action_dim)

    # Elite weights: softmax with temperature (numerically stable)
    # Source: tdmpc2.py:191 — exp(temperature * delta), NOT exp(delta / temperature).
    # With cfg.mppi_temperature=0.5, the multiply form is softer (coef 0.5);
    # the divide form would be 4× sharper (coef 2.0) and over-concentrates on top elite.
    max_score = elite_scores.max()
    exp_scores = jnp.exp(cfg.mppi_temperature * (elite_scores - max_score))
    weights = exp_scores / (exp_scores.sum() + 1e-9)  # (num_elites,)

    # Update mean/std (elite-weighted)
    new_mean = (weights[None, :, None] * elite_actions).sum(axis=1)  # (horizon, action_dim)
    var = (weights[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
    new_std = jnp.clip(jnp.sqrt(var), cfg.mppi_min_std, cfg.mppi_max_std)

    return new_mean, new_std, elite_actions, weights


def sample_pi_trajectories(
    plan_params,
    z_0: jax.Array,          # (latent_dim,) — single env
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    policy_net: "PolicyPrior",
) -> jax.Array:
    """Seed MPPI population with num_pi_trajs trajectories from the policy prior.

    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:155-165.

    Loop structure — CRITICAL:
      For h = 0..horizon-1: sample a_h = π(z_h)
      For h = 0..horizon-2: advance z_{h+1} = dynamics(z_h, a_h)
      Net: `horizon` policy samples, `horizon - 1` dynamics advances.
      A naive range(horizon) both-loop over-advances latent one step and produces OOD
      final samples.

    Returns: (horizon, num_pi_trajs, action_dim) — pi-seeded action sequences.
    """
    N = cfg.num_pi_trajs
    # Broadcast z_0 to (N, latent_dim)
    z = jnp.broadcast_to(z_0, (N,) + z_0.shape)

    def step(carry, h_idx):
        z, key = carry
        key, sk = jax.random.split(key)
        a, _ = policy_net.apply(plan_params["policy"], z, sk)  # (N, action_dim)
        # Advance dynamics ONLY when h < horizon - 1 (last step: sample a but don't advance).
        z_advanced = dynamics.apply(plan_params["dynamics"], z, a)
        z_next = jnp.where(h_idx < cfg.horizon - 1, z_advanced, z)
        return (z_next, key), a

    _, actions = jax.lax.scan(step, (z, key), jnp.arange(cfg.horizon))
    # actions: (horizon, N, action_dim)
    return actions


def init_mppi_mean(prev_mean: jax.Array, t0: jax.Array,
                    horizon: int, action_dim: int) -> jax.Array:
    """Warm-start MPPI mean for a single env.

    If t0 is True, return zeros. Otherwise shift: new[:-1] = prev[1:], new[-1] = 0.
    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:167-168.

    Args:
        prev_mean: (horizon, action_dim) — last optimized mean.
        t0: scalar bool — True on new episode.
    """
    shifted = jnp.concatenate([prev_mean[1:], jnp.zeros((1, action_dim))], axis=0)
    return jnp.where(t0, jnp.zeros_like(shifted), shifted)


def init_mppi_mean_batched(prev_mean: jax.Array, t0: jax.Array,
                            horizon: int, action_dim: int) -> jax.Array:
    """Per-env warm-start.

    Args:
        prev_mean: (num_envs, horizon, action_dim)
        t0:        (num_envs,) bool
    Returns:
        (num_envs, horizon, action_dim)
    """
    return jax.vmap(
        lambda p, t: init_mppi_mean(p, t, horizon, action_dim)
    )(prev_mean, t0)


# ------------------ MPPI plan loop + action selection ------------------


def gumbel_sample_elite(
    key: jax.Array,
    weights: jax.Array,        # (num_elites,)
    elite_actions: jax.Array,  # (horizon, num_elites, action_dim)
) -> jax.Array:
    """Sample a single elite trajectory via Gumbel-softmax argmax, return its t=0 action.

    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:201-205.
    NOT elite-weighted mean — a SINGLE elite is sampled, its t=0 action is returned.

    Returns: (action_dim,)
    """
    logits = jnp.log(weights + 1e-9)
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-9) + 1e-9)
    idx = jnp.argmax(logits + gumbels)
    return elite_actions[0, idx]


def plan(
    plan_params,
    z_0: jax.Array,          # (latent_dim,)
    prev_mean: jax.Array,    # (horizon, action_dim)
    t0: jax.Array,           # scalar bool
    cfg,
    key: jax.Array,
    eval_mode: bool = False,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> tuple[jax.Array, jax.Array]:
    """Full MPPI planner for a single env. Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:plan().

    Returns (action, new_prev_mean):
      action: (action_dim,) — action to execute this step.
      new_prev_mean: (horizon, action_dim) — stored for next step's warm-start.
    """
    key_pi_seed, key_iter, key_elite, key_noise = jax.random.split(key, 4)

    # 1. Warm-start mean + init std
    mean = init_mppi_mean(prev_mean, t0, cfg.horizon, cfg.action_dim)
    std = jnp.full((cfg.horizon, cfg.action_dim), cfg.mppi_max_std)

    # 2. Sample pi-seed trajectories once at start (source samples once, reuses)
    pi_trajs = sample_pi_trajectories(
        plan_params, z_0, cfg, key_pi_seed,
        dynamics=dynamics, policy_net=policy_net,
    )

    # 3. MPPI iteration loop. iterations = base + 2 if action_dim >= 20 (source line 35).
    iterations = cfg.mppi_iterations + (2 if cfg.action_dim >= 20 else 0)
    iter_keys = jax.random.split(key_iter, iterations)

    def iter_body(carry, k_i):
        mean_c, std_c = carry
        new_mean, new_std, elite_actions, weights = mppi_iteration(
            mean_c, std_c, plan_params, z_0, pi_trajs, cfg, k_i,
            dynamics=dynamics, reward_net=reward_net,
            q_ensemble_net=q_ensemble_net, policy_net=policy_net,
        )
        return (new_mean, new_std), (elite_actions, weights)

    (final_mean, final_std), (all_elites, all_weights) = jax.lax.scan(
        iter_body, (mean, std), iter_keys
    )
    # Use final iteration's elites + weights
    elite_actions = all_elites[-1]  # (horizon, num_elites, action_dim)
    weights = all_weights[-1]       # (num_elites,)

    # 4. Gumbel-sample single elite, take t=0 action
    action = gumbel_sample_elite(key_elite, weights, elite_actions)  # (action_dim,)

    # 5. Exploration noise (skip in eval_mode): std[0] * ε
    noise = jax.random.normal(key_noise, (cfg.action_dim,)) * final_std[0]
    action = jnp.where(eval_mode, action, action + noise)
    action = jnp.clip(action, -1.0, 1.0)

    return action, final_mean


def make_plan_batched(
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
):
    """Return a jit+vmap'd `plan` over num_envs.

    Module instances are closed over (cannot be vmapped). Returned callable has signature:
        plan_fn(plan_params, z_0_b, prev_mean_b, t0_b, cfg, keys, eval_mode) → (actions, new_prev_means)
    where _b suffix = per-env leading dim.

    JIT is essential: without it, every env-step re-traces the full MPPI scan (measured
    ~800ms/call CPU, ~10-30s/call under GPU contention). With jit, 2nd+ calls drop to <10ms.
    """
    def single_plan(plan_params, z_0, prev_mean, t0, cfg, key, eval_mode):
        return plan(
            plan_params, z_0, prev_mean, t0, cfg, key, eval_mode=eval_mode,
            dynamics=dynamics, reward_net=reward_net,
            q_ensemble_net=q_ensemble_net, policy_net=policy_net,
        )
    # vmap over (z_0, prev_mean, t0, key); plan_params/cfg/eval_mode shared
    vmapped = jax.vmap(single_plan, in_axes=(None, 0, 0, 0, None, 0, None))
    # JIT with cfg (idx 4) and eval_mode (idx 6) as static. Positional-call compatible.
    # (cfg is frozen/hashable; eval_mode is bool; both required to be static for jit cache.)
    return jax.jit(vmapped, static_argnums=(4, 6))


__all__ = ["make_plan_batched"]
