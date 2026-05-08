"""CLI override utilities for training scripts."""
import dataclasses

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
