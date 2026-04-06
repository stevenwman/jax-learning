"""Composable reward specification.

Reward terms are standalone functions that receive keyword args (data, action,
info, done, etc.) and return a scalar. The env builds a list of RewardTerms
in _post_init() and calls compute_rewards() in step().

compute_rewards() returns UNWEIGHTED values. The env's step() applies weights
from reward_config.scales as before. This keeps the refactor zero-behavior-change.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RewardTerm:
    """A single reward component."""
    name: str
    fn: Callable[..., Any]  # (**kwargs) -> scalar


def compute_rewards(terms: list[RewardTerm], **kwargs: Any) -> dict[str, Any]:
    """Compute all reward terms, return unweighted dict.

    Args:
        terms: List of RewardTerms to evaluate.
        **kwargs: Passed to each reward function (data, action, info, etc.)

    Returns:
        Dict of {term_name: unweighted_scalar}.
    """
    return {term.name: term.fn(**kwargs) for term in terms}
