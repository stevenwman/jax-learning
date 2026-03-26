"""Load a JAX RL checkpoint and run inference with pure numpy (no JAX dependency)."""
import json
import os
import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _swish(x: np.ndarray) -> np.ndarray:
    # Numerically stable: use expit
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return x * sig


ACTIVATIONS = {"relu": _relu, "swish": _swish, "silu": _swish}


class PolicyRunner:
    """Pure numpy policy inference from a JAX RL checkpoint.

    Supports PPO, SAC, and FastSAC checkpoints.
    For deterministic deployment: obs -> MLP encoder -> tanh(mean) -> action in [-1, 1].
    """

    def __init__(self, ckpt_dir: str):
        meta_path = os.path.join(ckpt_dir, "meta.json")
        params_path = os.path.join(ckpt_dir, "actor_params.npy")

        with open(meta_path) as f:
            self.meta = json.load(f)

        saved = np.load(params_path, allow_pickle=True).item()
        self.actor_params = saved["actor_params"]

        self.obs_dim = self.meta["obs_dim"]
        self.action_dim = self.meta["action_dim"]
        self.algo = self.meta.get("algo", "unknown")

        # Extract normalization state
        self.norm_mean = np.array(saved.get("norm_mean", np.zeros(self.obs_dim)), dtype=np.float32)
        self.norm_mos = np.array(saved.get("norm_mean_of_squares", np.ones(self.obs_dim)), dtype=np.float32)
        self.norm_count = int(saved.get("norm_count", 0))
        self.use_obs_norm = self.norm_count > 0

        # Determine network config
        self._resolve_network_config()

        # Extract weight matrices
        self._extract_weights()

    def _resolve_network_config(self):
        """Determine hidden dims and activation from meta.json."""
        if self.algo == "ppo":
            cfg = self.meta.get("ppo_config", {})
            self.hidden_dim = tuple(cfg.get("policy_hidden_dim", None) or (32, 32, 32, 32))
            self.activation = cfg.get("activation", "swish")
            self.squash = cfg.get("squash", True)
        elif self.algo in ("sac", "fast_sac"):
            cfg_key = "fast_sac_config" if self.algo == "fast_sac" else "sac_config"
            cfg = self.meta.get(cfg_key, {})
            self.hidden_dim = tuple(cfg.get("hidden_dim", (256, 256)))
            self.activation = cfg.get("activation", "relu")
            self.squash = True  # SAC always squashes
        else:
            raise ValueError(f"Unsupported algo: {self.algo}")

        self.act_fn = ACTIVATIONS.get(self.activation, _relu)
        self.has_layer_norm = False

    def _extract_weights(self):
        """Extract encoder Dense layers + policy head mean layer from Flax params."""
        self.encoder_layers = []  # list of (weight, bias, ln_scale, ln_bias)

        params = self.actor_params
        if "params" in params:
            params = params["params"]

        # Encoder: MlpEncoder_0
        encoder_params = params.get("MlpEncoder_0", {})
        i = 0
        while f"Dense_{i}" in encoder_params:
            layer = encoder_params[f"Dense_{i}"]
            w = np.array(layer["kernel"], dtype=np.float32)
            b = np.array(layer["bias"], dtype=np.float32)

            # Check for LayerNorm
            ln_key = f"LayerNorm_{i}"
            ln_scale, ln_bias = None, None
            if ln_key in encoder_params:
                self.has_layer_norm = True
                ln_scale = np.array(encoder_params[ln_key]["scale"], dtype=np.float32)
                ln_bias = np.array(encoder_params[ln_key]["bias"], dtype=np.float32)

            self.encoder_layers.append((w, b, ln_scale, ln_bias))
            i += 1

        # Policy head: GaussianHead_0 -> Dense_0 is the mean projection
        head_params = params.get("GaussianHead_0", {})
        if "Dense_0" not in head_params:
            raise ValueError(f"Could not find mean layer in GaussianHead_0: {list(head_params.keys())}")

        self.mean_w = np.array(head_params["Dense_0"]["kernel"], dtype=np.float32)
        self.mean_b = np.array(head_params["Dense_0"]["bias"], dtype=np.float32)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply obs normalization using saved running statistics."""
        if not self.use_obs_norm:
            return obs
        variance = np.maximum(self.norm_mos - self.norm_mean ** 2, 0.0)
        std = np.sqrt(variance) + 1e-8
        return (obs - self.norm_mean) / std

    @staticmethod
    def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numpy LayerNorm matching flax.linen.LayerNorm."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * scale + bias

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Run deterministic policy inference: obs -> action in [-1, 1]."""
        x = self.normalize_obs(obs.astype(np.float32))

        # Encoder forward pass
        for w, b, ln_scale, ln_bias in self.encoder_layers:
            x = x @ w + b
            if ln_scale is not None:
                x = self._layer_norm(x, ln_scale, ln_bias)
            x = self.act_fn(x)

        # Mean head -> tanh squash
        mean = x @ self.mean_w + self.mean_b

        if self.squash:
            action = np.tanh(mean)
        else:
            action = np.clip(mean, -1.0, 1.0)

        return action.astype(np.float32)
