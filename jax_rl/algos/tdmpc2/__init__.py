"""TDMPC2 sub-package. See .superpowers/specs/2026-04-26-tdmpc2-refactor-design.md.

Re-exports the public API from the legacy implementation file. As the refactor
progresses, individual symbols move from `tdmpc2_old.py` into focused modules
(`networks.py`, `losses.py`, `mppi.py`, `agent.py`); this `__init__.py` keeps
the import path `from jax_rl.algos.tdmpc2 import X` stable throughout.
"""

from jax_rl.algos.tdmpc2_old import (
    # Production-facing API
    TDMPC2State,
    make_plan_batched,
    make_update_step,
    Encoder,
    Dynamics,
    Reward,
    QEnsemble,
    PolicyPrior,
    build_world_model_optimizer,
    build_policy_optimizer,
    # Internal symbols imported by tests/test_tdmpc2.py — re-exported for back-compat
    NormedLinear,
    bound_log_std,
    squash_log_prob_correction,
    gaussian_log_prob,
    compute_scaled_entropy,
    compute_all_latents,
    compute_td_target,
    world_model_loss,
    policy_loss,
    mppi_rollout,
    mppi_iteration,
    sample_pi_trajectories,
    init_mppi_mean,
    init_mppi_mean_batched,
    gumbel_sample_elite,
    plan,
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
