"""TDMPC2 sub-package. See .superpowers/specs/2026-04-26-tdmpc2-refactor-design.md.

Re-exports the public API. As the refactor progresses, individual symbols move
from `tdmpc2_old.py` into focused modules (`networks.py`, `losses.py`, `mppi.py`,
`agent.py`); this `__init__.py` keeps the import path
`from jax_rl.algos.tdmpc2 import X` stable throughout."""

from jax_rl.algos.tdmpc2.networks import (
    NormedLinear,
    Encoder, Dynamics, Reward, QHead, QEnsemble, PolicyPrior,
    bound_log_std, squash_log_prob_correction, gaussian_log_prob,
    compute_scaled_entropy,
)
from jax_rl.algos.tdmpc2.losses import (
    compute_all_latents, compute_td_target,
    world_model_loss, policy_loss,
)
from jax_rl.algos.tdmpc2_old import (
    TDMPC2State,
    make_plan_batched,
    make_update_step,
    build_world_model_optimizer,
    build_policy_optimizer,
    # internal symbols still in tdmpc2_old (move out in later tasks):
    mppi_rollout, mppi_iteration, sample_pi_trajectories,
    init_mppi_mean, init_mppi_mean_batched,
    gumbel_sample_elite, plan,
)

__all__ = [
    "TDMPC2State",
    "make_plan_batched",
    "make_update_step",
    "Encoder", "Dynamics", "Reward", "QEnsemble", "PolicyPrior",
    "build_world_model_optimizer", "build_policy_optimizer",
    "NormedLinear",
    "bound_log_std", "squash_log_prob_correction", "gaussian_log_prob",
    "compute_scaled_entropy",
    "compute_all_latents", "compute_td_target",
    "world_model_loss", "policy_loss",
    "mppi_rollout", "mppi_iteration", "sample_pi_trajectories",
    "init_mppi_mean", "init_mppi_mean_batched",
    "gumbel_sample_elite", "plan",
]
