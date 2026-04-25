"""Prune training artifacts: checkpoints, wandb local cache, logs.

Retention policy (default):
  - checkpoints/<run>/best/        — KEEP always (eval-best snapshots)
  - checkpoints/<run>/orbax/       — DELETE if run mtime > 30 days
  - checkpoints/<run>/             — KEEP top-level meta.json + actor_params.npy
  - wandb/run-*                    — DELETE if mtime > 14 days (cloud has them)
  - .temp/logs/*.log               — DELETE if mtime > 14 days

Dry-run by default. Pass --apply to actually delete.

Usage:
  uv run python tools/prune_artifacts.py            # dry run, default policy
  uv run python tools/prune_artifacts.py --apply    # actually delete
  uv run python tools/prune_artifacts.py --ckpt-days 60 --wandb-days 30 --apply
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400


def prune_checkpoints(root: Path, days: int, apply: bool) -> tuple[int, int]:
    """Delete old orbax/ subdirs. Keep best/, meta.json, actor_params.npy."""
    if not root.exists():
        return 0, 0
    bytes_deleted = 0
    dirs_deleted = 0
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        orbax = run_dir / "orbax"
        if not orbax.exists():
            continue
        age = _age_days(orbax)
        if age <= days:
            continue
        size = _dir_size(orbax)
        bytes_deleted += size
        dirs_deleted += 1
        action = "DELETE" if apply else "would delete"
        print(f"  {action} {orbax.relative_to(REPO_ROOT)} ({_human_bytes(size)}, {age:.0f}d old)")
        if apply:
            shutil.rmtree(orbax)
    return dirs_deleted, bytes_deleted


def prune_wandb(root: Path, days: int, apply: bool) -> tuple[int, int]:
    """Delete wandb/run-* and wandb/offline-run-* older than N days."""
    if not root.exists():
        return 0, 0
    bytes_deleted = 0
    dirs_deleted = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not (name.startswith("run-") or name.startswith("offline-run-")):
            continue
        age = _age_days(entry)
        if age <= days:
            continue
        size = _dir_size(entry)
        bytes_deleted += size
        dirs_deleted += 1
        action = "DELETE" if apply else "would delete"
        print(f"  {action} {entry.relative_to(REPO_ROOT)} ({_human_bytes(size)}, {age:.0f}d old)")
        if apply:
            shutil.rmtree(entry)
    return dirs_deleted, bytes_deleted


def prune_logs(root: Path, days: int, apply: bool) -> tuple[int, int]:
    """Delete .log files older than N days under root (recursively)."""
    if not root.exists():
        return 0, 0
    bytes_deleted = 0
    files_deleted = 0
    for log in root.rglob("*.log"):
        if log.is_symlink() or not log.is_file():
            continue
        age = _age_days(log)
        if age <= days:
            continue
        try:
            size = log.stat().st_size
        except OSError:
            continue
        bytes_deleted += size
        files_deleted += 1
        action = "DELETE" if apply else "would delete"
        print(f"  {action} {log.relative_to(REPO_ROOT)} ({_human_bytes(size)}, {age:.0f}d old)")
        if apply:
            log.unlink()
    return files_deleted, bytes_deleted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry run)")
    ap.add_argument("--ckpt-days", type=int, default=30, help="Delete checkpoints/*/orbax/ older than N days (default 30)")
    ap.add_argument("--wandb-days", type=int, default=14, help="Delete wandb/run-* older than N days (default 14)")
    ap.add_argument("--log-days", type=int, default=14, help="Delete .temp/logs/*.log older than N days (default 14)")
    ap.add_argument("--skip-ckpt", action="store_true")
    ap.add_argument("--skip-wandb", action="store_true")
    ap.add_argument("--skip-logs", action="store_true")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — pass --apply to actually delete\n")

    total_bytes = 0
    if not args.skip_ckpt:
        print(f"Checkpoints (orbax/ subdirs older than {args.ckpt_days}d):")
        n, b = prune_checkpoints(REPO_ROOT / "checkpoints", args.ckpt_days, args.apply)
        print(f"  -> {n} dirs, {_human_bytes(b)}\n")
        total_bytes += b

    if not args.skip_wandb:
        print(f"wandb (run-* older than {args.wandb_days}d, cloud has them):")
        n, b = prune_wandb(REPO_ROOT / "wandb", args.wandb_days, args.apply)
        print(f"  -> {n} runs, {_human_bytes(b)}\n")
        total_bytes += b

    if not args.skip_logs:
        print(f"Logs (.temp/logs/*.log older than {args.log_days}d):")
        n, b = prune_logs(REPO_ROOT / ".temp" / "logs", args.log_days, args.apply)
        print(f"  -> {n} files, {_human_bytes(b)}\n")
        total_bytes += b

    print(f"Total: {_human_bytes(total_bytes)} {'reclaimed' if args.apply else 'would be reclaimed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
