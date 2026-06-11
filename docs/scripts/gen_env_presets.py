"""Generate environment presets reference from actual config objects.

Run: uv run python docs/scripts/gen_env_presets.py > docs/reference/env-presets.md

Imports the actual preset dictionaries and renders them as markdown tables.
Run whenever presets change in env_presets.py.
"""

import dataclasses
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jax_rl.configs.env_presets import (
    PRESETS, SAC_PRESETS, TD3_PRESETS, FAST_TD3_PRESETS, FAST_SAC_PRESETS,
    FLASH_SAC_PRESETS, TDMPC2_PRESETS,
    get_preset, get_sac_preset, get_td3_preset, get_fast_td3_preset,
    get_fast_sac_preset, get_flash_sac_preset,
)
from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.configs.tdmpc2_config import TDMPC2Config


def _with_go2(presets: dict, getter) -> dict:
    """Append Go2 Warp variant presets resolved through `getter`.

    Go2 Warp (non-splitbelt) names no longer live in the static preset tables —
    they resolve from GO2_WARP_VARIANTS inside each getter — so the docs must
    resolve them the same way to list them.
    """
    return {**presets, **{name: getter(name) for name in sorted(GO2_WARP_VARIANTS)}}


def _fmt(v) -> str:
    """Format a value for display."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 1e6:
            return f"{v:.0e}"
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return f"({', '.join(str(x) for x in v)})"
    if isinstance(v, int) and v >= 1_000_000:
        return f"{v // 1_000_000}M"
    if isinstance(v, int) and v >= 1_000:
        return f"{v:,}"
    return str(v)


def _diff_from_default(obj, default_cls):
    """Return dict of fields that differ from the default dataclass."""
    default = default_cls()
    diff = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        default_val = getattr(default, f.name)
        if val != default_val:
            diff[f.name] = val
    return diff


def render_ppo_presets(presets: dict[str, TrainConfig]) -> str:
    """Render PPO preset table.

    Columns are trimmed for mobile readability: gamma and reward_scaling are
    folded into Notes only when they differ from the TrainConfig default.
    """
    # Columns shown in the table (narrower set)
    table_cols = ["num_envs", "total_timesteps", "lr"]
    ppo_table_cols = ["num_steps", "num_epochs", "entropy_coef"]
    train_default = TrainConfig()

    lines = [
        "## PPO Presets",
        "",
        "Used by `train_ppo_fast.py`. Accessed via `get_preset(env_name)`.",
        "",
        "| Environment | num_envs | timesteps | lr | num_steps | epochs | entropy_coef | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for env_name, cfg in presets.items():
        ppo = cfg.ppo
        notes_parts = []

        # Fold gamma / reward_scaling into Notes if non-default
        if cfg.gamma != train_default.gamma:
            notes_parts.append(f"gamma={_fmt(cfg.gamma)}")
        if cfg.reward_scaling != train_default.reward_scaling:
            notes_parts.append(f"reward_scaling={_fmt(cfg.reward_scaling)}")

        # Check for non-default PPO fields beyond the table columns
        ppo_diff = _diff_from_default(ppo, PPOConfig)
        for k, v in ppo_diff.items():
            if k not in ppo_table_cols and k not in ("num_minibatches", "minibatch_size", "num_envs", "gamma", "encoder"):
                notes_parts.append(f"{k}={_fmt(v)}")

        # Check for non-default TrainConfig fields beyond table columns
        cfg_diff = _diff_from_default(cfg, TrainConfig)
        for k, v in cfg_diff.items():
            if k not in table_cols and k not in ("env_name", "ppo", "episode_length", "gamma", "reward_scaling"):
                notes_parts.append(f"{k}={_fmt(v)}")

        notes = ", ".join(notes_parts) if notes_parts else ""

        lines.append(
            f"| {env_name} | {_fmt(cfg.num_envs)} | {_fmt(cfg.total_timesteps)} | "
            f"{_fmt(cfg.lr)} | "
            f"{_fmt(ppo.num_steps)} | {_fmt(ppo.num_epochs)} | {_fmt(ppo.entropy_coef)} | {notes} |"
        )

    # Add PPO algo defaults
    ppo_default = PPOConfig()
    defaults = {f.name: _fmt(getattr(ppo_default, f.name)) for f in dataclasses.fields(ppo_default)
                if f.name not in ("num_minibatches", "minibatch_size", "num_envs", "gamma", "encoder")}
    default_str = ", ".join(f"`{k}={v}`" for k, v in defaults.items())
    lines.append("")
    lines.append(f"PPO algo defaults: {default_str}.")
    lines.append("")
    return "\n".join(lines)


def render_offpolicy_presets(title: str, getter_name: str,
                              presets: dict[str, tuple], algo_config_cls, script_name: str = "") -> str:
    """Render an off-policy preset table.

    Columns are trimmed for mobile readability: gamma and reward_scaling are
    folded into Notes only when they differ from the TrainConfig default.
    """
    script_ref = f"Used by `{script_name}`." if script_name else "Off-policy algorithm presets."
    train_default = TrainConfig()
    lines = [
        f"## {title}",
        "",
        f"{script_ref} Accessed via `{getter_name}(env_name)`.",
        "",
        "| Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for env_name, (cfg, algo_cfg) in presets.items():
        notes_parts = []

        # Fold gamma / reward_scaling into Notes if non-default
        if cfg.gamma != train_default.gamma:
            notes_parts.append(f"gamma={_fmt(cfg.gamma)}")
        if cfg.reward_scaling != train_default.reward_scaling:
            notes_parts.append(f"reward_scaling={_fmt(cfg.reward_scaling)}")

        # Algo config diffs
        algo_diff = _diff_from_default(algo_cfg, algo_config_cls)
        skip_keys = {"batch_size", "grad_updates_per_step", "lr"}
        for k, v in algo_diff.items():
            if k not in skip_keys:
                notes_parts.append(f"{k}={_fmt(v)}")

        # TrainConfig diffs beyond the table columns (mirrors the PPO renderer).
        # num_eval_episodes / handle_truncation are baked into every off-policy
        # base cfg — skipped to keep Notes signal-only.
        cfg_diff = _diff_from_default(cfg, TrainConfig)
        cfg_skip = {"env_name", "ppo", "episode_length", "gamma", "reward_scaling",
                    "num_eval_episodes", "handle_truncation",
                    "num_envs", "total_timesteps", "lr"}
        for k, v in cfg_diff.items():
            if k not in cfg_skip:
                notes_parts.append(f"{k}={_fmt(v)}")

        notes = ", ".join(notes_parts) if notes_parts else ""
        utd = getattr(algo_cfg, "grad_updates_per_step", "-")

        lines.append(
            f"| {env_name} | {_fmt(cfg.num_envs)} | {_fmt(cfg.total_timesteps)} | "
            f"{_fmt(cfg.lr)} | "
            f"{_fmt(algo_cfg.batch_size)} | {_fmt(utd)} | {notes} |"
        )

    # Add algo defaults
    default = algo_config_cls()
    key_fields = ["tau", "hidden_dim", "activation", "batch_size", "grad_updates_per_step",
                   "buffer_size", "min_buffer_size", "q_layer_norm"]
    defaults = {}
    for fname in key_fields:
        if hasattr(default, fname):
            defaults[fname] = _fmt(getattr(default, fname))
    default_str = ", ".join(f"`{k}={v}`" for k, v in defaults.items())
    lines.append("")
    lines.append(f"{title.split()[0]} algo defaults: {default_str}.")
    lines.append("")
    return "\n".join(lines)


def render_tdmpc2_presets(presets: dict[str, TDMPC2Config]) -> str:
    """Render the TDMPC2 preset table.

    Unlike the other algos, TDMPC2_PRESETS maps env -> a single TDMPC2Config (not a
    (TrainConfig, AlgoConfig) tuple), because TDMPC2Config bundles training-loop
    fields alongside algo hparams. action_dim, discount, and episode length are set
    per-env by make_tdmpc2_config, so they get dedicated columns instead of polluting
    the non-default Notes diff.
    """
    # Set per-env by make_tdmpc2_config (or shown in their own column) — excluded
    # from the non-default Notes diff to keep it clean.
    per_env = {"action_dim", "discount", "total_steps", "num_envs", "horizon",
               "batch_size", "utd", "episode_lengths", "task_names"}
    lines = [
        "## TDMPC2 Presets",
        "",
        "Used by `train_tdmpc2.py`. Accessed via `get_tdmpc2_preset(env_name)` "
        "(raises `KeyError` for unlisted envs — `action_dim` has no safe default).",
        "",
        "| Environment | action_dim | total_steps | num_envs | horizon | batch_size | UTD | discount | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for env_name, cfg in presets.items():
        diff = _diff_from_default(cfg, TDMPC2Config)
        notes = ", ".join(f"{k}={_fmt(v)}" for k, v in diff.items() if k not in per_env)
        lines.append(
            f"| {env_name} | {_fmt(cfg.action_dim)} | {_fmt(cfg.total_steps)} | "
            f"{_fmt(cfg.num_envs)} | {_fmt(cfg.horizon)} | {_fmt(cfg.batch_size)} | "
            f"{_fmt(cfg.utd)} | {_fmt(cfg.discount)} | {notes} |"
        )

    # Algo defaults footer (model-based knobs that don't vary per-env).
    default = TDMPC2Config()
    key_fields = ["latent_dim", "mlp_dim", "num_q", "num_bins", "num_samples",
                  "num_elites", "mppi_iterations", "tau", "lr", "seed_steps"]
    defaults = ", ".join(f"`{k}={_fmt(getattr(default, k))}`" for k in key_fields
                         if hasattr(default, k))
    lines.append("")
    lines.append(f"TDMPC2 algo defaults: {defaults}.")
    lines.append("")
    return "\n".join(lines)


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of text with prefix (for pymdownx.tabbed content)."""
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def _as_tab(label: str, body: str) -> str:
    """Wrap body as a pymdownx.tabbed block.

    The rendered section starts with '## <Algo> Presets'; strip that h2 so
    the tab label isn't duplicated inside the tab content.
    """
    lines = body.split("\n", 2)
    if lines and lines[0].startswith("## "):
        # Drop the h2 and the blank line after it
        body = lines[2] if len(lines) > 2 else ""
    return f'=== "{label}"\n\n{_indent(body)}'


def main():
    header = """# Environment Presets

Auto-generated from `jax_rl/configs/env_presets.py`. Regenerate with:

```bash
uv run python docs/scripts/gen_env_presets.py
```

Presets return fully-configured `(TrainConfig, AlgoConfig)` tuples with tuned hyperparameters per environment. CLI flags override individual fields via `dataclasses.replace()`.

If an environment is not listed, a default config is used with the environment name set. `Go2Warp*` rows (except the splitbelt family) resolve from the variants table in `jax_rl/envs/locomotion/go2_warp_variants.py` rather than static preset entries.

Select an algorithm tab below to see its presets. Defaults (gamma=0.99, reward_scaling=1) are omitted from rows and only appear in Notes when overridden.

"""
    tabs = [
        _as_tab("PPO", render_ppo_presets(_with_go2(PRESETS, get_preset))),
        _as_tab("SAC", render_offpolicy_presets("SAC Presets", "get_sac_preset", _with_go2(SAC_PRESETS, get_sac_preset), SACConfig, "train_sac.py")),
        _as_tab("TD3", render_offpolicy_presets("TD3 Presets", "get_td3_preset", _with_go2(TD3_PRESETS, get_td3_preset), TD3Config, "train_td3.py")),
        _as_tab("FastTD3", render_offpolicy_presets("FastTD3 Presets", "get_fast_td3_preset", _with_go2(FAST_TD3_PRESETS, get_fast_td3_preset), FastTD3Config, "train_fast_td3.py")),
        _as_tab("FastSAC", render_offpolicy_presets("FastSAC Presets", "get_fast_sac_preset", _with_go2(FAST_SAC_PRESETS, get_fast_sac_preset), FastSACConfig, "train_fast_sac.py")),
        _as_tab("FlashSAC", render_offpolicy_presets("FlashSAC Presets", "get_flash_sac_preset", _with_go2(FLASH_SAC_PRESETS, get_flash_sac_preset), FlashSACConfig, "train_flashsac.py")),
        _as_tab("TDMPC2", render_tdmpc2_presets(TDMPC2_PRESETS)),
    ]
    output = header + "\n\n".join(tabs) + "\n"

    out_path = Path(__file__).resolve().parents[1] / "reference" / "env-presets.md"
    out_path.write_text(output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
