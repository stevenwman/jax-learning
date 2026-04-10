"""Stateless obs extraction + normalization pipeline for off-policy training.

Built once from config. Eliminates per-step conditionals in training loops.
norm_state is passed in/out — the pipeline holds no mutable state.

Usage::

    pipe = ObsPipeline(dict_obs=True, has_privileged=True,
                       use_obs_norm=True, n_frame_stack=3)

    # In training loop:
    raw_obs = pipe.get_obs(env_state.obs)
    norm_state = pipe.update_stats(raw_obs, norm_state)
    obs_for_action = pipe.normalize_for_action(raw_obs, norm_state)

    # After sampling from buffer:
    batch = pipe.normalize_batch(batch, norm_state)
"""

from jax_rl.buffers.jax_replay_buffer import FrameStackConfig, JaxReplayBuffer
from jax_rl.utils.normalization import (
    init as norm_init,
    normalize,
    normalize_stacked,
    update as norm_update,
)


class ObsPipeline:
    """Stateless obs extraction + normalization pipeline.

    Built once from config. Eliminates per-step conditionals in training loops.
    norm_state is passed in/out — the pipeline holds no mutable state.
    """

    def __init__(
        self,
        dict_obs: bool,
        has_privileged: bool,
        use_obs_norm: bool,
        n_frame_stack: int = 1,
        obs_norm_eps: float = 1e-8,
    ):
        self.dict_obs = dict_obs
        self.has_privileged = has_privileged
        self.use_obs_norm = use_obs_norm
        self.n_frame_stack = n_frame_stack
        self.obs_norm_eps = obs_norm_eps

    # ── Obs extraction ────────────────────────────────────────────────────

    def get_obs(self, env_obs):
        """Extract actor obs from env_state.obs (dict['state'] or passthrough)."""
        return env_obs["state"] if self.dict_obs else env_obs

    def get_critic_obs(self, env_obs):
        """Extract critic obs (privileged_state if available, else actor obs)."""
        if self.has_privileged:
            return env_obs["privileged_state"]
        return self.get_obs(env_obs)

    # ── Normalization ─────────────────────────────────────────────────────

    def update_stats(self, obs, norm_state):
        """Update running normalization stats. Returns new norm_state.

        No-op if obs_norm disabled. When frame stacking, updates stats on the
        raw (newest) frame only — first raw_dim columns.
        """
        if not self.use_obs_norm:
            return norm_state
        if self.n_frame_stack > 1:
            raw_dim = obs.shape[-1] // self.n_frame_stack
            return norm_update(norm_state, obs[:, :raw_dim])
        return norm_update(norm_state, obs)

    def normalize_for_action(self, obs, norm_state):
        """Normalize obs for actor forward pass. Passthrough if disabled."""
        if not self.use_obs_norm:
            return obs
        if self.n_frame_stack > 1:
            return normalize_stacked(
                norm_state, obs, self.n_frame_stack, eps=self.obs_norm_eps
            )
        return normalize(norm_state, obs, eps=self.obs_norm_eps)

    def normalize_batch(self, batch, norm_state):
        """Normalize obs+next_obs in replay batch. Sets critic_obs keys if not privileged.

        When has_privileged is False, aliases critic_obs/critic_next_obs to the
        (possibly normalized) obs/next_obs so algos always read batch["critic_obs"].
        """
        if self.use_obs_norm:
            if self.n_frame_stack > 1:
                batch["obs"] = normalize_stacked(
                    norm_state, batch["obs"], self.n_frame_stack, eps=self.obs_norm_eps
                )
                batch["next_obs"] = normalize_stacked(
                    norm_state, batch["next_obs"], self.n_frame_stack, eps=self.obs_norm_eps
                )
            else:
                batch["obs"] = normalize(norm_state, batch["obs"], eps=self.obs_norm_eps)
                batch["next_obs"] = normalize(norm_state, batch["next_obs"], eps=self.obs_norm_eps)
        if not self.has_privileged:
            batch["critic_obs"] = batch["obs"]
            batch["critic_next_obs"] = batch["next_obs"]
        return batch

    # ── Buffer factory ────────────────────────────────────────────────────

    def make_buffer(self, obs_dim, action_dim, buffer_size,
                    critic_obs_dim=None, num_envs=None):
        """Create JaxReplayBuffer with correct frame_stack + extra_obs_dims.

        Args:
            obs_dim: Actor observation dim (stacked if frame stacking,
                i.e. raw_dim * n_frames).
            action_dim: Action dimensionality.
            buffer_size: Maximum number of transitions.
            critic_obs_dim: Privileged critic obs dim. Required when
                has_privileged is True. Ignored otherwise.
            num_envs: Number of parallel envs (required when n_frame_stack > 1).

        Returns:
            JaxReplayBuffer configured for this pipeline.
        """
        extra_obs_dims = None
        if self.has_privileged:
            if critic_obs_dim is None:
                raise ValueError("critic_obs_dim required when has_privileged=True")
            extra_obs_dims = {"critic_obs": critic_obs_dim}

        frame_stack_config = None
        if self.n_frame_stack > 1:
            if num_envs is None:
                raise ValueError("num_envs required when n_frame_stack > 1")
            raw_dim = obs_dim // self.n_frame_stack
            frame_stack_config = FrameStackConfig(
                n_frames=self.n_frame_stack, raw_dim=raw_dim, num_envs=num_envs
            )
            return JaxReplayBuffer(
                raw_dim, action_dim, max_size=buffer_size,
                frame_stack_config=frame_stack_config,
                extra_obs_dims=extra_obs_dims,
            )

        return JaxReplayBuffer(
            obs_dim, action_dim, max_size=buffer_size,
            extra_obs_dims=extra_obs_dims,
        )

    def make_buffer_with_critic(self, obs_dim, action_dim, buffer_size,
                                critic_obs_dim, num_envs=None):
        """Create JaxReplayBuffer with privileged critic obs support.

        Like make_buffer, but also allocates extra buffers for critic_obs when
        has_privileged is True.

        Args:
            obs_dim: Actor observation dim (stacked if frame stacking).
            action_dim: Action dimensionality.
            buffer_size: Maximum number of transitions.
            critic_obs_dim: Privileged critic obs dim (used only if has_privileged).
            num_envs: Number of parallel envs (required when n_frame_stack > 1).

        Returns:
            JaxReplayBuffer configured for this pipeline.
        """
        extra_obs_dims = {"critic_obs": critic_obs_dim} if self.has_privileged else None
        frame_stack_config = None

        if self.n_frame_stack > 1:
            if num_envs is None:
                raise ValueError("num_envs required when n_frame_stack > 1")
            raw_dim = obs_dim // self.n_frame_stack
            frame_stack_config = FrameStackConfig(
                n_frames=self.n_frame_stack, raw_dim=raw_dim, num_envs=num_envs
            )
            return JaxReplayBuffer(
                raw_dim, action_dim, max_size=buffer_size,
                frame_stack_config=frame_stack_config,
                extra_obs_dims=extra_obs_dims,
            )

        return JaxReplayBuffer(
            obs_dim, action_dim, max_size=buffer_size,
            extra_obs_dims=extra_obs_dims,
        )

    # ── Eval helper ───────────────────────────────────────────────────────

    def make_obs_norm_fn(self, norm_state):
        """Return obs normalization function for eval_runner. None if not needed.

        The returned function accepts raw env obs (dict or flat) and returns
        normalized flat obs suitable for the actor.

        Returns None only when obs_norm is disabled AND obs is flat (no
        extraction needed).
        """
        if not self.use_obs_norm and not self.dict_obs:
            return None

        # Capture pipeline config in closure — norm_state is read at call time
        # from the variable in the enclosing scope (caller rebinds it each step).
        dict_obs = self.dict_obs
        use_obs_norm = self.use_obs_norm
        n_frame_stack = self.n_frame_stack
        obs_norm_eps = self.obs_norm_eps

        def _obs_norm_fn(obs):
            flat = obs["state"] if dict_obs else obs
            if not use_obs_norm:
                return flat
            if n_frame_stack > 1:
                return normalize_stacked(norm_state, flat, n_frame_stack, eps=obs_norm_eps)
            return normalize(norm_state, flat, eps=obs_norm_eps)

        return _obs_norm_fn

    # ── Norm state factory ────────────────────────────────────────────────

    def init_norm_state(self, obs_dim):
        """Create initial norm_state appropriate for this pipeline's config.

        When obs_norm is enabled, returns a fresh norm state (count=0).
        When disabled, returns an identity norm state (mean=0, var=1, count=1)
        for checkpoint compatibility.

        Args:
            obs_dim: Actor observation dim (stacked if frame stacking).

        Returns:
            NormalizationState ready for use.
        """
        from jax_rl.training.env_setup import make_identity_norm_state

        if self.use_obs_norm:
            if self.n_frame_stack > 1:
                raw_dim = obs_dim // self.n_frame_stack
                return norm_init(raw_dim)
            return norm_init(obs_dim)

        return make_identity_norm_state(obs_dim)
