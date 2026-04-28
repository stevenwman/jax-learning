"""Export a trained FastSAC actor to ONNX for deploy-time inference.

Deploy target: Jetson running onnxruntime, no JAX dependency.

The FastSAC actor is a plain MLP:

    obs (48,) → Dense(512) → swish
              → Dense(256) → swish
              → Dense(128) → swish
              → Dense_mean(12)
              → tanh → action (12,)

No obs normalization (this checkpoint was trained with obs_normalization=False).
No BatchNorm, no LayerNorm. The stochastic log_std head exists in the checkpoint
but is dropped here — deterministic inference only.

We hand-build the ONNX graph via onnx.helper (4 Gemm nodes, 3 swish = Sigmoid+Mul,
1 Tanh). No tensorflow / jax2tf dependency at export time.

─────────────────────────────────────────────────────────────────────────────
Observation layout (48-dim, Go2WarpJoystickFlat) — embed in deploy code:

    Idx      Name              Dim  Description
    [0:3]    gyro              3    IMU angular velocity (rad/s)
    [3:6]    accelerometer     3    IMU linear acceleration (m/s²)
    [6:9]    gravity           3    Projected gravity in base frame (state estimator)
    [9:21]   joint_pos_offset  12   qpos - default_pose, policy order
    [21:33]  joint_vel         12   Joint velocities, policy order
    [33:45]  last_act          12   Previous raw policy action (post-tanh, pre-scale)
    [45:48]  command           3    [vx, vy, yaw_rate] joystick

Joint order (policy):
    [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]

SDK order differs — see deploy/go2_constants.py::SDK_TO_POLICY.

Output: action ∈ [-1, 1]^12 (tanh'd). Deploy pipeline applies:
    joint_target = default_pose + 0.5 * action
then PD with Kp=20, Kd=0.5 at 500 Hz inner loop, 50 Hz policy.
─────────────────────────────────────────────────────────────────────────────

Usage::

    uv run python -m jax_rl.utils.export \\
        --checkpoint checkpoints/<run>/best \\
        --out checkpoints/<run>/best/actor.onnx

Dependencies (already installed in the main training venv; onnxruntime only
adds ~40MB)::

    uv pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from typing import Any

import numpy as np


def export_actor_to_onnx(checkpoint_dir: str, output_path: str) -> dict[str, Any]:
    """Export a FastSAC actor checkpoint to an ONNX file.

    Args:
        checkpoint_dir: path containing meta.json + actor_params.npy
        output_path: destination .onnx file

    Returns:
        dict with obs_dim, action_dim, hidden_dim, onnx_path, file_size_bytes.

    Raises:
        ValueError: if algo != fast_sac, activation != swish, obs_normalization=True,
            or any weight shape does not match meta.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from jax_rl.training.checkpointing import load_actor_for_inference

    meta, actor_params, norm_state, actor_batch_stats = load_actor_for_inference(
        checkpoint_dir
    )

    # ── Validate checkpoint shape matches our assumptions ────────────────────
    algo = meta.get("algo")
    if algo != "fast_sac":
        raise ValueError(
            f"export_actor_to_onnx only supports fast_sac checkpoints; "
            f"got algo={algo!r}"
        )
    sc = meta.get("fast_sac_config", {})
    obs_norm_baked = bool(sc.get("obs_normalization", False)) and int(norm_state.count) > 0
    activation = sc.get("activation", "swish")
    if activation != "swish":
        raise ValueError(
            f"Only 'swish' activation is supported; got {activation!r}. "
            f"Add the appropriate ONNX op if you need another activation."
        )
    if actor_batch_stats:
        raise ValueError(
            "Checkpoint contains actor_batch_stats (BatchNorm). "
            "FastSAC actor should have none — aborting."
        )

    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    hidden_dim = tuple(sc.get("hidden_dim", (512, 256, 128)))

    # ── Extract weights in numpy form ────────────────────────────────────────
    params = actor_params["params"]
    enc = params["MlpEncoder_0"]
    head = params["GaussianHead_0"]

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    prev = obs_dim
    for i, d_out in enumerate(hidden_dim):
        key = f"Dense_{i}"
        k = np.asarray(enc[key]["kernel"], dtype=np.float32)
        b = np.asarray(enc[key]["bias"], dtype=np.float32)
        if k.shape != (prev, d_out):
            raise ValueError(
                f"encoder {key} kernel shape {k.shape} != expected ({prev}, {d_out})"
            )
        if b.shape != (d_out,):
            raise ValueError(
                f"encoder {key} bias shape {b.shape} != expected ({d_out},)"
            )
        weights.append(k)
        biases.append(b)
        prev = d_out

    # Mean head: GaussianHead_0.Dense_0. Dense_1 is log_std and is discarded.
    w_mean = np.asarray(head["Dense_0"]["kernel"], dtype=np.float32)
    b_mean = np.asarray(head["Dense_0"]["bias"], dtype=np.float32)
    if w_mean.shape != (prev, action_dim):
        raise ValueError(
            f"mean head kernel shape {w_mean.shape} != ({prev}, {action_dim})"
        )
    if b_mean.shape != (action_dim,):
        raise ValueError(
            f"mean head bias shape {b_mean.shape} != ({action_dim},)"
        )

    # ── Build initializers ───────────────────────────────────────────────────
    initializers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        initializers.append(numpy_helper.from_array(w, name=f"W{i}"))
        initializers.append(numpy_helper.from_array(b, name=f"B{i}"))
    initializers.append(numpy_helper.from_array(w_mean, name="W_mean"))
    initializers.append(numpy_helper.from_array(b_mean, name="B_mean"))

    # Bake observation normalization (obs - mean) * inv_std into the graph as
    # constants if the ckpt trained with --obs-norm. This makes the ONNX a
    # self-contained artifact: deploy can feed raw obs without a Python-side
    # normalize step. Mirrors deploy/policy_runner.py:normalize_obs.
    nodes: list = []
    x = "obs"
    if obs_norm_baked:
        norm_mean = np.asarray(norm_state.mean, dtype=np.float32)
        norm_mos = np.asarray(norm_state.mean_of_squares, dtype=np.float32)
        variance = np.maximum(norm_mos - norm_mean ** 2, 0.0)
        inv_std = (1.0 / (np.sqrt(variance) + 1e-8)).astype(np.float32)
        if norm_mean.shape != (obs_dim,) or inv_std.shape != (obs_dim,):
            raise ValueError(
                f"norm stats shape {norm_mean.shape}/{inv_std.shape} != ({obs_dim},)"
            )
        initializers.append(numpy_helper.from_array(norm_mean, name="norm_mean"))
        initializers.append(numpy_helper.from_array(inv_std, name="norm_inv_std"))
        nodes.append(
            helper.make_node("Sub", inputs=["obs", "norm_mean"], outputs=["obs_centered"],
                             name="norm_sub")
        )
        nodes.append(
            helper.make_node("Mul", inputs=["obs_centered", "norm_inv_std"],
                             outputs=["obs_norm"], name="norm_mul")
        )
        x = "obs_norm"

    # ── Build nodes: Gemm → (Sigmoid + Mul for swish) ×3, then Gemm + Tanh ──
    # Gemm(A, B, C) = A @ B + C  with alpha=beta=1, transA=transB=0.
    # Flax kernels are stored [in, out], which matches Gemm's default B layout.
    for i in range(len(hidden_dim)):
        pre = f"h{i}_pre"
        sig = f"h{i}_sig"
        act = f"h{i}"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[x, f"W{i}", f"B{i}"],
                outputs=[pre],
                name=f"dense_{i}",
                alpha=1.0, beta=1.0, transA=0, transB=0,
            )
        )
        nodes.append(
            helper.make_node("Sigmoid", inputs=[pre], outputs=[sig],
                             name=f"swish_sig_{i}")
        )
        nodes.append(
            helper.make_node("Mul", inputs=[pre, sig], outputs=[act],
                             name=f"swish_mul_{i}")
        )
        x = act

    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[x, "W_mean", "B_mean"],
            outputs=["mean"],
            name="dense_mean",
            alpha=1.0, beta=1.0, transA=0, transB=0,
        )
    )
    nodes.append(
        helper.make_node("Tanh", inputs=["mean"], outputs=["action"], name="tanh_out")
    )

    # ── Graph + model ────────────────────────────────────────────────────────
    obs_info = helper.make_tensor_value_info(
        "obs", TensorProto.FLOAT, ["batch", obs_dim]
    )
    act_info = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, ["batch", action_dim]
    )

    graph = helper.make_graph(
        nodes,
        name="fast_sac_actor",
        inputs=[obs_info],
        outputs=[act_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="jax-rl.export",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    onnx.save(model, output_path)

    return {
        "onnx_path": output_path,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "file_size_bytes": os.path.getsize(output_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation: compare JAX vs ONNX outputs
# ─────────────────────────────────────────────────────────────────────────────

def _build_fast_sac_for_inference(meta: dict, obs_dim: int, action_dim: int):
    """Recreate a FastSAC algo object matching the checkpoint config (dummy opt)."""
    import optax

    from jax_rl.algos.fast_sac import FastSAC
    from jax_rl.configs.fast_sac_config import FastSACConfig

    sc = meta["fast_sac_config"]
    cfg = FastSACConfig(
        hidden_dim=tuple(sc.get("hidden_dim", (512, 256, 128))),
        activation=sc.get("activation", "swish"),
        q_layer_norm=sc.get("q_layer_norm", True),
        target_entropy_scale=sc.get("target_entropy_scale", 0.0),
        num_atoms=sc.get("num_atoms", 101),
        v_min=sc.get("v_min", -20.0),
        v_max=sc.get("v_max", 20.0),
        q_aggregation=sc.get("q_aggregation", "avg"),
        critic_hidden_dim=tuple(sc["critic_hidden_dim"])
        if sc.get("critic_hidden_dim") else None,
        max_std=sc.get("max_std", 1.0),
    )
    dummy_opt = optax.adam(1e-3)
    return FastSAC(cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)


def validate_export(
    checkpoint_dir: str,
    onnx_path: str,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Compare JAX and ONNX outputs on random, zero, and default-pose inputs.

    Raises AssertionError if any max abs diff exceeds ``atol``.
    """
    import jax
    import jax.numpy as jnp
    import onnxruntime as ort

    # Force JAX to use strict fp32 matmul (not TF32, which has only ~1e-3
    # precision on Ampere+ GPUs). Without this, random-input diffs against
    # ONNX Runtime's fp32 CPU math are dominated by TF32 truncation (~5e-3),
    # not real bugs. Must be set before any jit trace happens inside FastSAC.
    jax.config.update("jax_default_matmul_precision", "highest")

    from jax_rl.training.checkpointing import load_actor_for_inference

    meta, actor_params, norm_state, _ = load_actor_for_inference(checkpoint_dir)
    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    algo = _build_fast_sac_for_inference(meta, obs_dim, action_dim)

    # If the ONNX has obs-norm baked in, the JAX side must normalize before
    # select_action so the comparison is apples-to-apples (ONNX feeds raw obs
    # and applies Sub+Mul internally; JAX receives raw obs and applies the
    # same formula in numpy before calling the actor).
    sc = meta.get("fast_sac_config", {})
    obs_norm_baked = bool(sc.get("obs_normalization", False)) and int(norm_state.count) > 0
    if obs_norm_baked:
        nm = np.asarray(norm_state.mean, dtype=np.float32)
        mos = np.asarray(norm_state.mean_of_squares, dtype=np.float32)
        inv_std_np = (1.0 / (np.sqrt(np.maximum(mos - nm ** 2, 0.0)) + 1e-8)).astype(np.float32)
        normalize_for_jax = lambda o: (o - nm) * inv_std_np
    else:
        normalize_for_jax = lambda o: o

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    assert input_name == "obs", f"expected input 'obs', got {input_name!r}"

    key = jax.random.PRNGKey(0)

    def jax_action(obs_np: np.ndarray) -> np.ndarray:
        out = algo.select_action(
            actor_params, jnp.asarray(normalize_for_jax(obs_np)), key, deterministic=True
        )
        return np.asarray(out)

    def onnx_action(obs_np: np.ndarray) -> np.ndarray:
        return sess.run(None, {input_name: obs_np})[0]

    # 1) Random batch of 16
    rng = np.random.default_rng(0)
    rand_obs = rng.standard_normal((16, obs_dim)).astype(np.float32)
    rand_diff = float(np.max(np.abs(jax_action(rand_obs) - onnx_action(rand_obs))))

    # 2) Zero obs
    zero_obs = np.zeros((1, obs_dim), dtype=np.float32)
    zero_diff = float(np.max(np.abs(jax_action(zero_obs) - onnx_action(zero_obs))))

    # 3) Default-pose obs: robot at home pose, zero velocities, zero command.
    #    With obs_normalization=False, every slice of the obs vector is zero:
    #      linvel=0, gyro=0, gravity=0 (the +z in base frame would be [0,0,-1]
    #      only if the base is upright — but we cannot bake physics in here,
    #      and the policy is trained on raw (not-normalized) obs, so the
    #      "home + quiet" input is the zero vector by construction of
    #      joint_pos_offset = qpos - default_pose).
    #    Gravity projection in practice sits near (0,0,-1); we test that too.
    gravity_obs = np.zeros((1, obs_dim), dtype=np.float32)
    gravity_obs[0, 6:9] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    dp_diff = float(
        np.max(np.abs(jax_action(gravity_obs) - onnx_action(gravity_obs)))
    )

    results = {
        "random_max_diff": rand_diff,
        "zero_max_diff": zero_diff,
        "default_pose_max_diff": dp_diff,
    }
    worst = max(results.values())
    if worst > atol:
        raise AssertionError(
            f"ONNX export mismatch: worst max-abs-diff={worst:.3e} > atol={atol:.1e}\n"
            f"  details: {results}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _peak_rss_mb() -> float:
    # Linux: ru_maxrss is in KB. macOS: bytes. We assume Linux (Jetson / dev box).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a FastSAC actor checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint directory (e.g. .../best)")
    parser.add_argument("--out", required=True,
                        help="Output .onnx path")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Max allowed abs diff between JAX and ONNX")
    args = parser.parse_args()

    t0 = time.time()
    info = export_actor_to_onnx(args.checkpoint, args.out)
    t_export = time.time() - t0

    print(f"Exported: {info['onnx_path']}")
    print(f"  obs_dim = {info['obs_dim']}, action_dim = {info['action_dim']}")
    print(f"  hidden_dim = {info['hidden_dim']}")
    print(f"  file size = {info['file_size_bytes'] / 1024:.1f} KB")
    print(f"  export time = {t_export:.2f}s")

    t0 = time.time()
    diffs = validate_export(args.checkpoint, args.out, atol=args.atol)
    t_validate = time.time() - t0

    print("Validation (JAX vs ONNX max-abs-diff):")
    for k, v in diffs.items():
        print(f"  {k:>24s} = {v:.3e}")
    print(f"  validate time = {t_validate:.2f}s")
    print(f"Peak RSS during export+validate: {_peak_rss_mb():.1f} MB")
    print("OK: export matches JAX within tolerance.")


if __name__ == "__main__":
    main()
