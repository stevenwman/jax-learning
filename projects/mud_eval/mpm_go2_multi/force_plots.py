"""Plotting helpers for MPM particle -> rigid body contact forces in the Go2 example."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_foot_forces(history: dict[str, dict[str, list]], output_path: str, mode: str = "magnitude") -> None:
    """Save a plot of the MPM particle contact forces applied to the robot feet.

    Parameters
    ----------
    history:
        Maps a body/foot name to ``{"times": [...], "forces": [[fx, fy, fz], ...]}``.
    output_path:
        File to write the PNG to.
    mode:
        ``"magnitude"`` plots ``|F|`` per foot on a single axis. ``"xyz"`` breaks
        the force out into its X/Y/Z components on three stacked axes.
    """
    feet = [name for name, h in history.items() if h["times"]]
    if not feet:
        print("[force_plots] no force data recorded; skipping plot")
        return

    if mode == "xyz":
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        comp_labels = ["Fx", "Fy", "Fz"]
        for name in feet:
            times = np.asarray(history[name]["times"])
            forces = np.asarray(history[name]["forces"])  # (T, 3)
            for c in range(3):
                axes[c].plot(times, forces[:, c], label=name)
        for c, ax in enumerate(axes):
            ax.set_ylabel(f"{comp_labels[c]} (N)")
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("MPM contact force on feet (X/Y/Z components)", fontsize=14)
    else:  # magnitude
        fig, ax = plt.subplots(figsize=(14, 5))
        for name in feet:
            times = np.asarray(history[name]["times"])
            forces = np.asarray(history[name]["forces"])  # (T, 3)
            ax.plot(times, np.linalg.norm(forces, axis=1), label=name)
        ax.set_ylabel("|F| (N)")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()
        fig.suptitle("MPM contact force magnitude on feet", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[force_plots] saved foot force plot to {output_path}")
