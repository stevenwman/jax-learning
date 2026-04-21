"""Generate CLI flags reference table from argparse definitions.

Run: uv run python docs/scripts/gen_cli_reference.py > docs/reference/cli-flags.md

This script constructs the same argparse parsers used by the training scripts
and renders them as markdown tables. Run it whenever CLI flags change.
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path so we can import jax_rl configs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _type_name(action: argparse.Action) -> str:
    """Human-readable type string for an argparse action."""
    if isinstance(action, argparse._StoreTrueAction):
        return "flag"
    if action.type is None:
        return "str"
    name = getattr(action.type, "__name__", str(action.type))
    if action.nargs == "+":
        return f"{name}+"
    if action.nargs == 2:
        return f"{name} {name}"
    return name


def _default_str(action: argparse.Action) -> str:
    """Human-readable default value."""
    if isinstance(action, argparse._StoreTrueAction):
        return "off"
    if action.default is None:
        return "from preset" if "preset" in (action.help or "").lower() or "config" in (action.help or "").lower() else "-"
    if action.required:
        return "**required**"
    return f"`{action.default}`"


def render_parser(name: str, parser: argparse.ArgumentParser) -> str:
    """Render a parser's arguments as a markdown table."""
    lines = [
        f"## `{name}`",
        "",
        "| Flag | Type | Default | Description |",
        "|------|------|---------|-------------|",
    ]
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        flag = ", ".join(f"`{o}`" for o in action.option_strings) or f"`{action.dest}`"
        lines.append(
            f"| {flag} | {_type_name(action)} | {_default_str(action)} | {action.help or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_sac_parser() -> argparse.ArgumentParser:
    """Mirror of train_sac.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--target-entropy-scale", type=float, default=None,
                        help="target_entropy = -scale * action_dim (default: from algo config)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    return parser


def build_td3_parser() -> argparse.ArgumentParser:
    """Mirror of train_td3.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--exploration-noise", type=float, default=None,
                        help="Exploration noise std for TD3-family")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    return parser


def build_fast_sac_parser() -> argparse.ArgumentParser:
    """Mirror of train_fast_sac.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (UTD ratio, default: from config)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--target-entropy-scale", type=float, default=None,
                        help="target_entropy = -scale * action_dim (default: from algo config)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    return parser


def build_fast_td3_parser() -> argparse.ArgumentParser:
    """Mirror of train_fast_td3.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (UTD ratio, default: from config)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--exploration-noise", type=float, default=None,
                        help="Exploration noise std for TD3-family")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    return parser


def build_ppo_parser() -> argparse.ArgumentParser:
    """Mirror of train_ppo_fast.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartpoleBalance",
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Rollout steps per environment per collect phase (default: from preset)")
    parser.add_argument("--num-updates-per-batch", type=int, default=None,
                        help="Collect-update cycles per iteration (default: from preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (default: from env preset)")
    parser.add_argument("--policy-hidden-dim", type=int, nargs="+", default=None,
                        help="Policy network hidden layer sizes (e.g., 256 128)")
    parser.add_argument("--value-hidden-dim", type=int, nargs="+", default=None,
                        help="Value network hidden layer sizes (e.g., 256 256 256)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="Entropy bonus coefficient (default: from env preset)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 50000 episodes)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: from preset)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print training stats every N iterations")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms")
    return parser


def build_flashsac_parser() -> argparse.ArgumentParser:
    """Mirror of train_flashsac.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartpoleBalance",
                        help="Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (overrides lr_peak in FlashSACConfig)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="End learning rate for cosine decay")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (UTD ratio)")
    parser.add_argument("--no-reward-norm", action="store_true",
                        help="Disable adaptive reward normalization")
    parser.add_argument("--G-max", type=float, default=None,
                        help="Target max magnitude for discounted returns (reward norm)")
    parser.add_argument("--no-weight-norm", action="store_true",
                        help="Disable weight normalization after optimizer steps")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name")
    return parser


def build_pusht_parser() -> argparse.ArgumentParser:
    """Mirror of train_pusht.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", type=str, default="dense",
                        choices=["coverage", "sparse", "shaped", "approach", "dense", "contact_gated"],
                        help="Reward function. contact_gated is the working RL-from-scratch recipe.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total environment steps to train")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for gradient updates")
    parser.add_argument("--grad-updates-per-step", type=int, default=1,
                        help="Gradient updates per env step (UTD ratio)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for actor and critic")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Multiplier on env reward before replay. Use 0.1 for contact_gated.")
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="Global grad norm clip (e.g. 1.0). None = off.")
    parser.add_argument("--target-entropy-scale", type=float, default=1.0,
                        help="SAC target entropy = -scale * action_dim. Bigger = more explore.")
    parser.add_argument("--obs-type", type=str, default="state",
                        choices=["state", "environment_state_agent_pos"],
                        help="state=5d, environment_state_agent_pos=18d (flattened keypoints + agent)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Stack N consecutive obs. Implicit velocity; flattened to obs_dim × N.")
    parser.add_argument("--action-repeat", type=int, default=1,
                        help="Repeat each action K env steps (frame skip). Commits policy to direction.")
    parser.add_argument("--coverage-shape", type=str, default="linear",
                        choices=["linear", "log_barrier"],
                        help="r_coverage shape. 'linear' = raw coverage. 'log_barrier' = -log(1 - cov + eps): unbounded near goal, amplifies final-mile precision.")
    parser.add_argument("--coverage-eps", type=float, default=0.01,
                        help="Epsilon for log_barrier (sets max reward ceiling: ε=0.01 → r_max≈4.6).")
    parser.add_argument("--success-threshold", type=float, default=0.95,
                        help="Coverage threshold for terminated=True. DP paper uses 0.95. Lower to 0.85 for tractable success events.")
    parser.add_argument("--success-bonus", type=float, default=50.0,
                        help="Terminal reward on success (contact_gated mode only). Default 50.")
    parser.add_argument("--block-shape", type=str, default="tee",
                        choices=["tee", "ellipse", "triangle", "s", "dr"],
                        help="Block shape. 'dr' samples uniformly per episode from {tee, ellipse, triangle, s}.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval-every-n-steps", type=int, default=50_000,
                        help="Evaluate every N environment steps")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking")
    return parser


def build_record_parser() -> argparse.ArgumentParser:
    """Mirror of record_video.py's argparse setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name (inferred from checkpoint if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory (random policy if omitted)")
    parser.add_argument("--out", type=str, default="rollout.mp4",
                        help="Output video path")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum rollout steps")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera name override")
    parser.add_argument("--seed", type=int, default=None,
                        help="Environment reset seed")
    parser.add_argument("--kicks", action="store_true",
                        help="Zero velocity command + random velocity kicks every 1.5s")
    return parser


def main():
    header = """# CLI Flags

Auto-generated from argparse definitions. Regenerate with:

```bash
uv run python docs/scripts/gen_cli_reference.py
```

---

"""
    sections = [
        render_parser("train_ppo_fast.py", build_ppo_parser()),
        render_parser("train_sac.py", build_sac_parser()),
        render_parser("train_td3.py", build_td3_parser()),
        render_parser("train_fast_sac.py", build_fast_sac_parser()),
        render_parser("train_fast_td3.py", build_fast_td3_parser()),
        render_parser("train_flashsac.py", build_flashsac_parser()),
        render_parser("train_pusht.py", build_pusht_parser()),
        render_parser("record_video.py", build_record_parser()),
    ]
    output = header + "\n---\n\n".join(sections)

    # Write directly to the docs file
    out_path = Path(__file__).resolve().parents[1] / "reference" / "cli-flags.md"
    out_path.write_text(output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
