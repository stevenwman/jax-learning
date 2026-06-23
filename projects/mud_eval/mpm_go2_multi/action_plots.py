"""Plotting helpers for the MPM Go2 example."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_action_norm(history: list[float], output_path: str) -> None:
    """Save a line plot of the policy action L2 norm per step."""
    if not history:
        print("[action_plots] no action data recorded; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\|a\|_2$")
    ax.set_title("Policy action L2 norm over time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[action_plots] saved action norm plot to {output_path}")
