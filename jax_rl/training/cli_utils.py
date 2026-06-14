"""CLI override utilities for training scripts."""
import dataclasses


# ── Shared argparse builders (were duplicated verbatim across the off-policy
# train scripts). Each script composes the tiers it actually exposes, then adds
# its algo-specific args inline. Arg names/dests/types are UNCHANGED from the
# originals so apply_cli_overrides (which maps by dest) is unaffected. ──────────
def add_common_train_args(parser):
    """The 12 args every off-policy train script exposes."""
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation.")
    parser.add_argument("--resume-warmup", type=str, default="policy",
                        choices=["policy", "random"],
                        help="On resume, refill buffer using loaded policy actions "
                             "(default, prevents eval drop) or legacy random uniform")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 5000 episodes; "
                             "Go2 OSC/physical presets set 500)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    return parser


def add_env_shaping_args(parser):
    """Obs/reward/latency shaping args (every off-policy script except FlashSAC)."""
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    return parser


def add_replay_args(parser):
    """Replay-buffer / update-ratio args (all off-policy scripts except vanilla TD3)."""
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per environment step (default: from algo config)")
    return parser

# Direct cfg fields: CLI arg name (dashes become underscores via argparse) == config field name
CFG_FIELDS = {
    "num_envs", "total_timesteps", "lr", "reward_scaling",
    "episode_length", "n_frame_stack", "action_delay_ms", "reset_mode",
    "gamma",
}

# Direct algo fields: CLI arg name == config field name
ALGO_FIELDS = {
    "batch_size", "grad_updates_per_step", "buffer_size",
    "target_entropy_scale",
    "v_min", "v_max", "num_atoms",  # C51 distributional critic support
    "tau", "policy_delay",  # SAC hyperparams (FlashSAC-comparison diag)
}

# CLI name -> algo config field name (where they differ)
ALGO_RENAMES = {
    "exploration_noise": "exploration_noise_std",
}

# Bool algo flags: CLI name -> algo config field name
BOOL_ALGO = {
    "obs_norm": "obs_normalization",
}

# Special cases handled explicitly in apply_cli_overrides:
#   frame_stack      -> cfg.n_frame_stack        (CLI rename)
#   eval_every       -> cfg.eval_every_n_episodes (CLI rename)
#   action_delay_range_ms -> cfg.action_delay_range_ms as tuple


def apply_cli_overrides(args, cfg, algo_cfg):
    """Apply argparse Namespace to config dataclasses. Returns (cfg, algo_cfg).

    Only applies non-None / truthy values. Handles field name remapping and
    bool flags. The original configs are returned unchanged if no overrides apply.
    """
    args_dict = vars(args)

    cfg_ov = {}
    algo_ov = {}

    # Direct cfg fields
    for field in CFG_FIELDS:
        val = args_dict.get(field)
        if val is not None:
            cfg_ov[field] = val

    # Direct algo fields
    for field in ALGO_FIELDS:
        val = args_dict.get(field)
        if val is not None:
            algo_ov[field] = val

    # Renamed algo fields
    for cli_name, cfg_name in ALGO_RENAMES.items():
        val = args_dict.get(cli_name)
        if val is not None:
            algo_ov[cfg_name] = val

    # Bool algo flags
    for cli_name, cfg_name in BOOL_ALGO.items():
        if args_dict.get(cli_name):
            algo_ov[cfg_name] = True

    # Special: frame_stack -> n_frame_stack
    if args_dict.get("frame_stack") is not None:
        cfg_ov["n_frame_stack"] = args_dict["frame_stack"]

    # Special: eval_every -> eval_every_n_episodes
    if args_dict.get("eval_every") is not None:
        cfg_ov["eval_every_n_episodes"] = args_dict["eval_every"]

    # Special: action_delay_range_ms -> tuple
    if args_dict.get("action_delay_range_ms") is not None:
        cfg_ov["action_delay_range_ms"] = tuple(args_dict["action_delay_range_ms"])

    if cfg_ov:
        cfg = dataclasses.replace(cfg, **cfg_ov)
    if algo_ov:
        algo_cfg = dataclasses.replace(algo_cfg, **algo_ov)

    return cfg, algo_cfg
