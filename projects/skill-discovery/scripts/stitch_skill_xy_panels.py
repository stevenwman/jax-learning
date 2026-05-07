"""scripts/stitch_skill_xy_panels.py — assemble multi-panel skill diversity figure.

Reads existing per-seed PNGs and stitches them into a single panel image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="PNG paths to stitch, in row-major order")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=None,
                        help="Defaults to len(inputs) // rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    cols = args.cols or (len(args.inputs) // args.rows)
    fig, axes = plt.subplots(args.rows, cols,
                             figsize=(cols * 6 * args.scale,
                                      args.rows * 6 * args.scale))
    if args.rows * cols == 1:
        axes = [[axes]]
    elif args.rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[a] for a in axes]

    for idx, p in enumerate(args.inputs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"saved panel to {args.output}")


if __name__ == "__main__":
    main()
