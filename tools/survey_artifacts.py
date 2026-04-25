"""Survey training artifacts — read-only inventory, no deletion.

Output: markdown digest of checkpoints/, wandb/, .temp/logs/ — sized, aged,
and (for checkpoints) labeled with algo/env/best-eval pulled from meta.json.

Use this to decide what to prune by hand. Companion to manual `rm -rf`.

Usage:
  uv run python tools/survey_artifacts.py                  # default digest
  uv run python tools/survey_artifacts.py --top 50         # show more rows
  uv run python tools/survey_artifacts.py --section ckpts  # one section only
  uv run python tools/survey_artifacts.py --by env         # group ckpts by env
  uv run python tools/survey_artifacts.py --by algo        # group by algo
  uv run python tools/survey_artifacts.py --csv > out.csv  # checkpoint rows as CSV
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} PB"


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


def _read_meta(run_dir: Path) -> dict:
    """Return meta.json contents (algo / env / best eval), or empty dict."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


_ALGO_TOKENS = (
    "flash_sac", "fast_sac", "fast_td3", "ppo_contraction", "ppo_fast",
    "tdmpc2", "ppo", "sac", "td3", "pusht",
)


def _parse_run_name(name: str) -> dict:
    """Best-effort parse of `<YYYYMMDD>_<HHMMSS>_<algo>_<env>_seed<N>` run dirs."""
    out = {"algo": "?", "env": "?", "seed": "?"}
    parts = name.lower()
    # Find an algo token in the name.
    for tok in _ALGO_TOKENS:
        if f"_{tok}_" in parts or parts.startswith(f"{tok}_"):
            out["algo"] = tok
            break
    # Seed.
    import re
    m = re.search(r"_seed(\d+)", parts)
    if m:
        out["seed"] = m.group(1)
    return out


def _best_eval_from_metrics(run_dir: Path) -> float | None:
    """Read metrics.csv (or best/metrics.csv) and return max eval/return seen."""
    candidates = [run_dir / "metrics.csv", run_dir / "best" / "metrics.csv"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open() as f:
                reader = csv.DictReader(f)
                col = None
                rows = []
                for r in reader:
                    if col is None:
                        # Pick the first eval-ish column we can find.
                        for k in ("eval_mean", "eval/mean", "eval/return_mean", "return_eval", "eval"):
                            if k in r:
                                col = k
                                break
                        if col is None:
                            return None
                    try:
                        v = float(r[col])
                        rows.append(v)
                    except (ValueError, TypeError):
                        continue
                if rows:
                    return max(rows)
        except OSError:
            continue
    return None


def _ckpt_row(run_dir: Path) -> dict:
    """Build one row of checkpoint info."""
    meta = _read_meta(run_dir)
    train_cfg = meta.get("train_config", {}) if isinstance(meta, dict) else {}

    # Env from train_config; algo + seed from run name.
    parsed = _parse_run_name(run_dir.name)
    env = train_cfg.get("env_name") or meta.get("env_name") or parsed["env"]
    algo = meta.get("algo") or parsed["algo"]
    seed = train_cfg.get("seed", meta.get("seed", parsed["seed"]))
    total = train_cfg.get("total_timesteps", meta.get("total_timesteps", "?"))

    best = _best_eval_from_metrics(run_dir)
    best_str = f"{best:.1f}" if isinstance(best, (int, float)) else "?"

    has_best = (run_dir / "best").exists()
    has_orbax = (run_dir / "orbax").exists()
    orbax_size = _dir_size(run_dir / "orbax") if has_orbax else 0
    has_video = any(run_dir.rglob("*.mp4"))

    return {
        "run": run_dir.name,
        "size": _dir_size(run_dir),
        "orbax_size": orbax_size,
        "age": _age_days(run_dir),
        "algo": str(algo),
        "env": str(env),
        "seed": str(seed),
        "total": str(total),
        "best": best_str,
        "has_best": has_best,
        "has_orbax": has_orbax,
        "has_video": has_video,
    }


def survey_checkpoints(top: int, group_by: str | None) -> list[dict]:
    root = REPO_ROOT / "checkpoints"
    if not root.exists():
        print("checkpoints/: not present\n")
        return []

    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        rows.append(_ckpt_row(run_dir))

    total_size = sum(r["size"] for r in rows)
    total_orbax = sum(r["orbax_size"] for r in rows)
    print(f"## checkpoints/ — {len(rows)} runs, {_human_bytes(total_size)} total")
    print(f"   orbax/ subdirs (resume state, candidates to prune): {_human_bytes(total_orbax)}")
    print()

    if not rows:
        return rows

    # Top by size
    rows_sorted = sorted(rows, key=lambda r: -r["size"])
    print(f"### Top {min(top, len(rows))} by size")
    print()
    print("| size | orbax | age | algo | env | seed | best | best/ | orbax/ | 🎬 | run |")
    print("|------|-------|-----|------|-----|------|------|-------|--------|----|-----|")
    for r in rows_sorted[:top]:
        print(
            f"| {_human_bytes(r['size'])} | {_human_bytes(r['orbax_size'])} | "
            f"{r['age']:.0f}d | {r['algo']} | {r['env']} | {r['seed']} | "
            f"{r['best']} | {'✓' if r['has_best'] else '·'} | "
            f"{'✓' if r['has_orbax'] else '·'} | "
            f"{'🎬' if r['has_video'] else '·'} | `{r['run']}` |"
        )
    print()

    n_with_video = sum(1 for r in rows if r["has_video"])
    if n_with_video:
        print(f"_{n_with_video} of {len(rows)} runs contain rendered videos (🎬) — typically keep these._")
        print()

    # Group rollups
    if group_by in ("env", "algo"):
        buckets = defaultdict(lambda: {"count": 0, "size": 0})
        for r in rows:
            key = r[group_by]
            buckets[key]["count"] += 1
            buckets[key]["size"] += r["size"]
        print(f"### By {group_by}")
        print()
        print(f"| {group_by} | runs | size |")
        print("|------|------|------|")
        for key in sorted(buckets, key=lambda k: -buckets[k]["size"]):
            b = buckets[key]
            print(f"| {key} | {b['count']} | {_human_bytes(b['size'])} |")
        print()

    # Age buckets
    age_buckets = {
        ">90d": [],
        "30-90d": [],
        "14-30d": [],
        "7-14d": [],
        "<7d": [],
    }
    for r in rows:
        a = r["age"]
        if a > 90:
            age_buckets[">90d"].append(r)
        elif a > 30:
            age_buckets["30-90d"].append(r)
        elif a > 14:
            age_buckets["14-30d"].append(r)
        elif a > 7:
            age_buckets["7-14d"].append(r)
        else:
            age_buckets["<7d"].append(r)

    print("### By age")
    print()
    print("| bucket | runs | size | orbax (prunable) |")
    print("|--------|------|------|-------------------|")
    for label, items in age_buckets.items():
        if not items:
            continue
        sz = sum(r["size"] for r in items)
        ox = sum(r["orbax_size"] for r in items)
        print(f"| {label} | {len(items)} | {_human_bytes(sz)} | {_human_bytes(ox)} |")
    print()

    # Orphans: no best/, no orbax/ — likely failed runs
    orphans = [r for r in rows if not r["has_best"] and not r["has_orbax"]]
    if orphans:
        print(f"### Orphans (no best/, no orbax/ — likely failed/aborted runs): {len(orphans)}")
        print()
        for r in sorted(orphans, key=lambda r: -r["size"])[:top]:
            print(f"- `{r['run']}` — {_human_bytes(r['size'])}, {r['age']:.0f}d old, {r['algo']}/{r['env']}")
        print()

    return rows


def survey_wandb(top: int) -> None:
    root = REPO_ROOT / "wandb"
    if not root.exists():
        print("wandb/: not present\n")
        return
    rows = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not (entry.name.startswith("run-") or entry.name.startswith("offline-run-")):
            continue
        rows.append({
            "name": entry.name,
            "size": _dir_size(entry),
            "age": _age_days(entry),
        })

    total = sum(r["size"] for r in rows)
    print(f"## wandb/ — {len(rows)} run dirs, {_human_bytes(total)} total")
    print(f"   *(local cache; cloud has the source-of-truth at wandb.ai)*")
    print()

    if not rows:
        return

    # Top by size
    rows_sorted = sorted(rows, key=lambda r: -r["size"])
    print(f"### Top {min(top, len(rows))} by size")
    print()
    print("| size | age | run |")
    print("|------|-----|-----|")
    for r in rows_sorted[:top]:
        print(f"| {_human_bytes(r['size'])} | {r['age']:.0f}d | `{r['name']}` |")
    print()

    # Age rollup
    bucks = {">30d": 0, "14-30d": 0, "7-14d": 0, "<7d": 0}
    counts = {">30d": 0, "14-30d": 0, "7-14d": 0, "<7d": 0}
    for r in rows:
        a = r["age"]
        if a > 30:
            k = ">30d"
        elif a > 14:
            k = "14-30d"
        elif a > 7:
            k = "7-14d"
        else:
            k = "<7d"
        bucks[k] += r["size"]
        counts[k] += 1
    print("### By age")
    print()
    print("| bucket | runs | size |")
    print("|--------|------|------|")
    for k, v in bucks.items():
        if counts[k]:
            print(f"| {k} | {counts[k]} | {_human_bytes(v)} |")
    print()


def survey_logs(top: int) -> None:
    root = REPO_ROOT / ".temp" / "logs"
    if not root.exists():
        print(".temp/logs/: not present\n")
        return
    rows = []
    for log in root.rglob("*.log"):
        if log.is_symlink() or not log.is_file():
            continue
        try:
            rows.append({
                "name": str(log.relative_to(root)),
                "size": log.stat().st_size,
                "age": _age_days(log),
            })
        except OSError:
            pass

    total = sum(r["size"] for r in rows)
    print(f"## .temp/logs/ — {len(rows)} files, {_human_bytes(total)} total")
    print()

    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: -r["size"])
    print(f"### Top {min(top, len(rows))} by size")
    print()
    print("| size | age | file |")
    print("|------|-----|------|")
    for r in rows_sorted[:top]:
        print(f"| {_human_bytes(r['size'])} | {r['age']:.0f}d | `{r['name']}` |")
    print()


def emit_csv(rows: list[dict]) -> None:
    """Dump checkpoint rows as CSV for downstream pipelines / sheets."""
    if not rows:
        return
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=20, help="Rows per top-by-size table (default 20)")
    ap.add_argument("--section", choices=["ckpts", "wandb", "logs", "all"], default="all",
                    help="Restrict to one section (default: all)")
    ap.add_argument("--by", choices=["env", "algo"], default=None,
                    help="Add group-by-N rollup to checkpoint section")
    ap.add_argument("--csv", action="store_true",
                    help="Emit checkpoint rows as CSV to stdout (suppresses other output)")
    args = ap.parse_args()

    if args.csv:
        rows = []
        root = REPO_ROOT / "checkpoints"
        if root.exists():
            for run_dir in sorted(root.iterdir()):
                if run_dir.is_dir():
                    rows.append(_ckpt_row(run_dir))
        emit_csv(rows)
        return 0

    print(f"# Artifact Survey — {time.strftime('%Y-%m-%d')}")
    print()

    if args.section in ("ckpts", "all"):
        survey_checkpoints(args.top, args.by)
    if args.section in ("wandb", "all"):
        survey_wandb(args.top)
    if args.section in ("logs", "all"):
        survey_logs(args.top)

    return 0


if __name__ == "__main__":
    sys.exit(main())
