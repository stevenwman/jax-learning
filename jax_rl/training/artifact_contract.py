"""Checkpoint artifact contract — `meta["artifact_kind"]` + version.

Phase A of the artifact-contract rollout (see `.context/lessons/env_backends.md`
followups). Every checkpoint writer declares what kind of artifact it is, so
consumers (record_video, deploy/PolicyRunner, ONNX export) can fail loudly
on shape mismatches instead of silently FileNotFoundError-ing on missing
files that belong to a different algo's contract.

Kind catalog (extend as new artifact shapes appear):

| Kind                  | Files                                          | Writer scripts                                         |
|-----------------------|------------------------------------------------|--------------------------------------------------------|
| `shared_actor_v1`     | meta.json + metrics.csv + actor_params.npy + orbax/ | sac, td3, fast_sac, fast_td3, flash_sac, ppo, ppo_contraction |
| `tdmpc2_v1`           | meta.json + actor_params.npz + world_model_params.npz | train_tdmpc2.py                                        |
| `pusht_legacy_v1`     | actor_params_best.npy + actor_params_final.npy | scripts/train_pusht.py (deprecated; pending deletion)  |

Versioning policy: bump the version suffix when on-disk file shape, file
names, or required meta fields change in a way old consumers can't read.
Adding optional metadata (e.g., `meta["control"]` block in Phase D) is NOT
a version bump — consumers ignore unknown keys.

Phase B (forthcoming): consumers import `KIND_*` constants + check the
artifact_kind against an allowlist before trying to load. Backwards compat:
missing field → warn + assume `shared_actor_v0` (legacy, pre-Phase-A).
"""

# Latest contract version. Writers stamp this; readers compare against
# their supported set.
LATEST_VERSION = 1

# Kind constants. Use these in writers + consumer allowlists.
KIND_SHARED_ACTOR = "shared_actor_v1"
KIND_TDMPC2 = "tdmpc2_v1"
KIND_PUSHT_LEGACY = "pusht_legacy_v1"

# Backwards-compat sentinel for pre-Phase-A checkpoints (no artifact_kind
# field in meta.json). Consumers fall back to this when the field is
# missing, with a one-line warning.
KIND_LEGACY_SHARED_ACTOR = "shared_actor_v0"


def stamp_meta(meta: dict, kind: str, version: int = LATEST_VERSION) -> dict:
    """Stamp a meta dict with artifact_kind + artifact_version.

    Returns the same dict for chaining. Idempotent — calling twice with
    the same kind is a no-op (overwrites with the same value).
    """
    meta["artifact_kind"] = kind
    meta["artifact_version"] = int(version)
    return meta


# ── Phase B: consumer validation ─────────────────────────────────────────

def read_artifact_kind(meta: dict, *, ckpt_path: str | None = None) -> str:
    """Return `meta["artifact_kind"]`, defaulting to legacy with a warning.

    Pre-Phase-A checkpoints don't have the field. Treat them as
    `KIND_LEGACY_SHARED_ACTOR` (i.e. assume they're the shared off-policy
    shape) and emit a single-line warning naming the ckpt path so the
    operator can identify which old artifact triggered the fallback.
    """
    kind = meta.get("artifact_kind")
    if kind is None:
        ckpt_str = f" ({ckpt_path})" if ckpt_path else ""
        print(
            f"  [artifact_contract] WARN: meta missing artifact_kind"
            f"{ckpt_str} — assuming {KIND_LEGACY_SHARED_ACTOR!r}. "
            f"Re-train to get a self-describing checkpoint."
        )
        return KIND_LEGACY_SHARED_ACTOR
    return str(kind)


def assert_artifact_kind(
    meta: dict,
    allowed: list[str],
    tool_name: str,
    *,
    ckpt_path: str | None = None,
    redirect: str | None = None,
) -> str:
    """Verify `meta["artifact_kind"]` is in the allowlist; raise on mismatch.

    Args:
        meta: parsed meta.json dict.
        allowed: list of supported `KIND_*` constants for this tool.
        tool_name: shown in the error message (e.g., "record_video.py").
        ckpt_path: shown in the error message for operator context.
        redirect: optional redirect text appended to the error
                  ("use scripts/record_video_tdmpc2.py instead").

    Returns the resolved kind on success.
    """
    kind = read_artifact_kind(meta, ckpt_path=ckpt_path)
    if kind in allowed:
        return kind
    msg = (
        f"{tool_name} supports artifact_kind={allowed!r} but got "
        f"{kind!r}{f' from {ckpt_path}' if ckpt_path else ''}."
    )
    if redirect:
        msg += f"\n\n{redirect}"
    raise ValueError(msg)


# ── Phase C: per-kind on-disk file validators ────────────────────────────

def validate_shared_actor_files(ckpt_dir: str) -> None:
    """Verify a `shared_actor_v1` ckpt directory has the required files.

    Required: meta.json + actor_params.npy + orbax/.
    Raises FileNotFoundError naming the missing piece.
    """
    import os
    required = ("meta.json", "actor_params.npy", "orbax")
    missing = [f for f in required if not os.path.exists(os.path.join(ckpt_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"shared_actor_v1 ckpt at {ckpt_dir!r} is missing required files: "
            f"{missing}. Expected: {list(required)}."
        )


def validate_tdmpc2_files(ckpt_dir: str) -> None:
    """Verify a `tdmpc2_v1` ckpt directory has the required files.

    Required: meta.json + actor_params.npz + world_model_params.npz.
    Raises FileNotFoundError naming the missing piece.
    """
    import os
    required = ("meta.json", "actor_params.npz", "world_model_params.npz")
    missing = [f for f in required if not os.path.exists(os.path.join(ckpt_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"tdmpc2_v1 ckpt at {ckpt_dir!r} is missing required files: "
            f"{missing}. Expected: {list(required)}."
        )
