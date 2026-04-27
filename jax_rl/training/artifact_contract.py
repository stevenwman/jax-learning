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
