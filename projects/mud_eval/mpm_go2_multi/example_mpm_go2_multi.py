'''
Example MPM Go2
Load in a policy, and can use it on the GO2

May need to run export path first for the example_robot_go2 module to work
  export PYTHONPATH=/home/rml/Documents/george/mud_dynamics_2/Newton:$PYTHONPATH

  Old working policy: uv run python newton/examples/mpm/mpm_go2_multi/example_mpm_go2_multi.py --viewer gl  --policy-path policies/RSL_RL-2026-02-18_11-30-14/model_500.pt

Currently testing these two policies:
  uv run python newton/examples/mpm/mpm_go2_multi/example_mpm_go2_multi.py \
  --viewer gl   --policy-path policies/Baseline/model_4999.pt \
  --viewer gl   \
  --tolerance 1e-6 \
  --max-iterations 500 \
  --plot-actions newton/examples/mpm/mpm_go2_multi/plots/l2_norm_ppo.png




  Plot the MPM contact forces applied to a foot (magnitude or X/Y/Z components):
  uv run python newton/examples/mpm/mpm_go2_multi/example_mpm_go2_multi.py \
      --viewer gl \
      --policy-path policies/Baseline/model_4999.pt \
      --plot-forces newton/examples/mpm/mpm_go2_multi/plots/fl_foot_forces.png \
      --plot-forces-foot FL_calf \
      --plot-forces-mode xyz

      
      uv run python newton/examples/mpm/mpm_go2_multi/example_mpm_go2_multi.py \
      --viewer gl \
      --policy-path policies/Curriculum/model_4999.pt \
      --video recordings/curriculum.mp4 \
      --plot-forces newton/examples/mpm/mpm_go2_multi/plots/fl_foot_forces.png \
      --plot-forces-foot FL_calf \
      --plot-forces-mode xyz


'''
import sys

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from mpm_go2_multi.bounding_walls import BoundingWalls
from mpm_go2_multi.load_go2_policy import Go2Policy, INITIAL_Q, PD_GAINS_KE, PD_GAINS_KD, DEFAULT_ACTION_SCALE
from mpm_go2_multi.twoway_coupling_go2 import compute_body_forces, subtract_body_force
from newton.solvers import SolverImplicitMPM

from pathlib import Path


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

class Example:
    def __init__(self, viewer, options):
        # ------------------------------------------------------------------
        # Load YAML config. All tunable parameters live there; CLI flags
        # below override individual entries.
        # ------------------------------------------------------------------
        here = Path(__file__).resolve().parent
        config_path = Path(getattr(options, "config", None) or (here / "config.yaml"))
        if not config_path.is_absolute():
            config_path = (here / config_path).resolve()
        self.cfg = _load_config(config_path)
        cfg = self.cfg
        print(f"Loaded config from {config_path}")

        # setup simulation parameters first
        self.fps = cfg["simulation"]["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = cfg["simulation"]["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps


        self.viewer = viewer
        self._debug_forces = bool(getattr(options, "debug_forces", False))

        self.device = wp.get_device()

        # import the robot model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        bd = cfg["builder_defaults"]
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=bd["joint"]["armature"],
            limit_ke=bd["joint"]["limit_ke"],
            limit_kd=bd["joint"]["limit_kd"],
        )
        builder.default_shape_cfg.ke = bd["shape"]["ke"]
        builder.default_shape_cfg.kd = bd["shape"]["kd"]
        builder.default_shape_cfg.kf = bd["shape"]["kf"]
        builder.default_shape_cfg.mu = bd["shape"]["mu"]

        self.body_keys = builder.body_key

        robot_cfg = cfg["robot"]
        stage_path = str(here / robot_cfg["urdf_relative_path"])

        initial_pos = wp.vec3(*robot_cfg["initial_position"])
        yaw_axis = wp.vec3(*robot_cfg["initial_yaw_axis"])
        yaw_angle = wp.pi * float(robot_cfg["initial_yaw_angle_pi_mult"])

        builder.add_urdf(
            stage_path,
            xform=wp.transform(initial_pos, wp.quat_from_axis_angle(yaw_axis, yaw_angle)),
            floating=bool(robot_cfg["floating"]),
            enable_self_collisions=bool(robot_cfg["enable_self_collisions"]),
            collapse_fixed_joints=bool(robot_cfg["collapse_fixed_joints"]),
            ignore_inertial_definitions=bool(robot_cfg["ignore_inertial_definitions"]),
        )


        # set initial joint positions (config overrides load_go2_policy defaults)
        initial_q = cfg["policy"].get("initial_joint_q") or INITIAL_Q
        for key, value in initial_q.items():
            builder.joint_q[builder.joint_key.index(key) + 6] = value

        # Set PD control gains from config (overrides load_go2_policy defaults)
        pd_ke = float(cfg["policy"]["pd_gains_ke"])
        pd_kd = float(cfg["policy"]["pd_gains_kd"])
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = pd_ke
            builder.joint_target_kd[i] = pd_kd
        self.action_scale = float(cfg["policy"]["action_scale"])

        ### ------------------------------------------------------------------------------------------------------ ###
        ### --------------------------------------------- SAND SETUP --------------------------------------------- ###
        ### ------------------------------------------------------------------------------------------------------ ###

        # Resolve voxel size: CLI overrides config when explicitly set.
        voxel_size = options.voxel_size if options.voxel_size is not None else float(cfg["mpm"]["voxel_size"])
        tolerance = options.tolerance if options.tolerance is not None else float(cfg["mpm"]["tolerance"])
        max_iterations = options.max_iterations if options.max_iterations is not None else int(cfg["mpm"]["max_iterations"])

        # spawn two material regions (config-driven)
        thin_mud_particles, medium_mud_particles, thick_mud_particles = self.spawn_multiple_materials(builder, voxel_size=voxel_size)

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        thin_mud_particles = wp.array(thin_mud_particles, dtype=int, device=self.model.device)
        medium_mud_particles = wp.array(medium_mud_particles, dtype=int, device=self.model.device)
        thick_mud_particles = wp.array(thick_mud_particles, dtype=int, device=self.model.device)

        pc = cfg["particle_contact"]
        self.model.particle_ke = float(pc["ke"])
        self.model.particle_kd = float(pc["kd"])
        self.model.particle_mu = float(pc["mu"])

        # Setup MPM Solver — only solver numerics here, material props go through material_parameters
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = cfg["mpm"]["transfer_scheme"]
        mpm_options.strain_basis = cfg["mpm"]["strain_basis"]
        mpm_options.max_iterations = max_iterations
        mpm_options.grid_type = cfg["mpm"]["grid_type"]
        mpm_options.air_drag = float(cfg["mpm"]["air_drag"])

        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        # Static MPM-only containment walls. Not added to the Newton/MuJoCo
        # model — they are MPM-only static colliders, so the rigid solver
        # never sees them.
        bw_cfg = cfg["bounding_walls"]
        if bw_cfg.get("enabled", True):
            mode = bw_cfg.get("mode", "auto")
            thickness = float(bw_cfg["thickness"])
            if mode == "auto":
                height = bw_cfg.get("height")
                self.bounding_walls = BoundingWalls.around_particles(
                    self.model.particle_q,
                    thickness=thickness,
                    padding=float(bw_cfg["padding"]),
                    height=float(height) if height is not None else None,
                    device=self.model.device,
                )
            elif mode == "explicit":
                self.bounding_walls = BoundingWalls.around_bed(
                    x_range=tuple(bw_cfg["x_range"]),
                    y_range=tuple(bw_cfg["y_range"]),
                    height=float(bw_cfg["explicit_height"]),
                    thickness=thickness,
                    device=self.model.device,
                )
            else:
                raise ValueError(f"bounding_walls.mode must be 'auto' or 'explicit', got {mode!r}")
        else:
            self.bounding_walls = BoundingWalls([], device=self.model.device)

        shape_flags_np = self.model.shape_flags.numpy()
        def _has_particle_shapes(body_id):
            shapes = np.array(self.model.body_shapes[body_id], dtype=int)
            if len(shapes) == 0:
                return False
            return bool(np.any((shape_flags_np[shapes] & newton.ShapeFlags.COLLIDE_PARTICLES) > 0))

        robot_body_ids = [bi for bi in range(-1, self.model.body_count) if _has_particle_shapes(bi)]
        n_bodies = len(robot_body_ids)
        collider_body_ids = robot_body_ids  + self.bounding_walls.collider_body_ids()
        collider_meshes = [None] * n_bodies  + self.bounding_walls.collider_meshes()
        mpm_model.setup_collider(
            collider_meshes=collider_meshes,
            collider_body_ids=collider_body_ids,
        )

        # Apply material plastic params per config (apply_material toggle per material).
        materials = cfg["materials"]
        particle_ids_by_name = {
            "thick_mud": thick_mud_particles,
            "medium_mud": medium_mud_particles,
            "thin_mud":  thin_mud_particles,
        }
        for name, mat in materials.items():
            if not mat.get("apply_material", False):
                continue
            pids = particle_ids_by_name.get(name)
            if pids is None:
                continue
            Example._apply_material(
                mpm_model, pids,
                yield_stress=float(mat["yield_stress"]),
                tensile_yield_ratio=float(mat["tensile_yield_ratio"]),
                yield_pressure=float(mat["yield_pressure"]),
                hardening=float(mat["hardening"]),
            )

        mpm_model.notify_particle_material_changed()

        # setup solvers
        ms = cfg["mujoco_solver"]
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_parallel=bool(ms["ls_parallel"]),
            njmax=int(ms["njmax"]),
        )
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)


        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)


        default_color = wp.vec3(*cfg["render"]["default_particle_color"])
        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=default_color, device=self.model.device
        )
        self.particle_colors[thin_mud_particles].fill_(wp.vec3(*materials["thin_mud"]["spawn"]["color"]))
        self.particle_colors[medium_mud_particles].fill_(wp.vec3(*materials["medium_mud"]["spawn"]["color"]))
        self.particle_colors[thick_mud_particles].fill_(wp.vec3(*materials["thick_mud"]["spawn"]["color"]))


        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)


        max_nodes = int(cfg["buffers"]["max_collider_nodes"])
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collider_body_id = self.mpm_solver.mpm_model.collider.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        # collect initial (empty) impulses
        self.mpm_solver.collect_collider_impulses(self.state_0)

        ### ------------------------------------------------------------------------------------------------------ ###
        ### ---------------------------------- SETUP THE CONTROL FOR THE POLICY ---------------------------------- ###
        ### ------------------------------------------------------------------------------------------------------ ###
        self.control = self.model.control()
        torch_device = wp.to_torch(self.state_0.joint_q).device

        # Newton joint order (skip floating_base): FL_hip, FL_thigh, FL_calf, FR_hip, ...
        # The policy's joint_pos_initial is the action offset / obs reference and
        # must equal the training default pose, so build it from the same source
        # (config initial_joint_q) used to set the spawn pose above.
        newton_joint_names = [k for k in builder.joint_key if k != 'floating_base']
        joint_pos_initial = torch.tensor(
            [initial_q[name] for name in newton_joint_names],
            device=torch_device, dtype=torch.float32,
        ).unsqueeze(0)

        self.policy = Go2Policy(
            options.policy_path,
            device=torch_device,
            joint_pos_initial=joint_pos_initial,
            action_scale=self.action_scale,
            search_relative_to=here,
        )

        self.command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
        self._auto_forward = bool(cfg["control"]["auto_forward"])

        # Optional action-norm logging — populated in apply_control(), plotted at exit.
        self._action_norm_plot_path = getattr(options, "plot_actions", None)
        self._action_norm_history: list[float] | None = [] if self._action_norm_plot_path else None
        if self._action_norm_history is not None:
            import atexit  # noqa: PLC0415
            from mpm_go2_multi.action_plots import plot_action_norm  # noqa: PLC0415
            atexit.register(
                lambda: plot_action_norm(self._action_norm_history, self._action_norm_plot_path)
            )

        # Optional MPM->foot contact-force logging — populated in step(), plotted at exit.
        self._force_plot_path = getattr(options, "plot_forces", None)
        self._force_plot_mode = getattr(options, "plot_forces_mode", "magnitude")
        self._force_history: dict[str, dict[str, list]] | None = None
        if self._force_plot_path:
            foot_arg = getattr(options, "plot_forces_foot", "FL_calf")
            self._force_bodies = self._resolve_foot_bodies(foot_arg)
            self._force_history = {name: {"times": [], "forces": []} for name in self._force_bodies}
            import atexit  # noqa: PLC0415
            from mpm_go2_multi.force_plots import plot_foot_forces  # noqa: PLC0415
            atexit.register(
                lambda: plot_foot_forces(self._force_history, self._force_plot_path, self._force_plot_mode)
            )
            print(
                f"Recording MPM contact forces on {self._force_bodies} "
                f"-> {self._force_plot_path} (mode={self._force_plot_mode})"
            )

        # Set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # ---- Video writer (.mp4 of the GL framebuffer) ----
        self._video_writer = None
        video_path = getattr(options, "video", None)
        if video_path:
            try:
                import imageio.v2 as imageio  # noqa: PLC0415
            except ImportError:
                import imageio  # noqa: PLC0415
            video_fps = int(getattr(options, "video_fps", 50))
            self._video_writer = imageio.get_writer(
                video_path, fps=video_fps, codec="libx264", quality=8, macro_block_size=2
            )
            print(f"Recording video to {video_path} at {video_fps} fps")

            # Make sure the mp4 gets finalized even on Ctrl-C / unclean exit.
            import atexit  # noqa: PLC0415
            def _flush_video():
                if self._video_writer is not None:
                    try:
                        self._video_writer.close()
                    finally:
                        self._video_writer = None
            atexit.register(_flush_video)

        # Close hook for video writer.
        if self._video_writer is not None:
            _orig_close = self.viewer.close
            def _close_all(*a, **kw):
                try:
                    if self._video_writer is not None:
                        self._video_writer.close()
                        self._video_writer = None
                finally:
                    _orig_close(*a, **kw)
            self.viewer.close = _close_all

        self.capture()

        # Precomputed-frame playback buffers. When populated by precompute(),
        # step() stops advancing the live simulation and instead copies cached
        # body_q / particle_q snapshots back into state_0 so render() draws
        # them. This trades startup time for smoother on-screen playback.
        self._precomputed_body_q = []
        self._precomputed_particle_q = []
        self._precomputed_sim_time = []
        self._playback_idx = 0
        ### ---------------------------------- End of setup for the control for the policy  ---------------------------------- ###

    def _collect_collider_impulses(self):
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(self.state_0)

        self.collider_impulse_ids.fill_(-1)
        n = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(collider_impulses[:n])
        self.collider_impulse_pos[:n].assign(collider_impulse_pos[:n])
        self.collider_impulse_ids[:n].assign(collider_impulse_ids[:n])

    def _resolve_foot_bodies(self, foot_arg):
        """Map a user foot spec to actual body-key names for force logging.

        Accepts a comma-separated list. Each entry may be a full body key
        (e.g. ``"FL_calf"``), a leg prefix (``"FL"``), or an ``"_foot"`` link
        name (``"FL_foot"``); the latter two resolve to the corresponding
        ``"_calf"`` body that survives ``collapse_fixed_joints``.
        """
        keys = list(self.body_keys)
        resolved = []
        for raw in str(foot_arg).split(","):
            token = raw.strip()
            if not token:
                continue
            name = self._match_foot(token, keys)
            if name not in resolved:
                resolved.append(name)
        if not resolved:
            raise ValueError(f"--plot-forces-foot did not resolve to any body: {foot_arg!r}")
        return resolved

    @staticmethod
    def _match_foot(token, keys):
        if token in keys:
            return token
        # "_foot" links are collapsed into their parent "_calf" body.
        candidate = token.replace("_foot", "_calf")
        if candidate in keys:
            return candidate
        candidate = f"{token}_calf"
        if candidate in keys:
            return candidate
        matches = [k for k in keys if token.lower() in k.lower()]
        calf_matches = [k for k in matches if "calf" in k.lower()]
        if len(calf_matches) == 1:
            return calf_matches[0]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"could not resolve foot {token!r} to a unique body. "
            f"Matches: {matches or 'none'}. Available bodies: {keys}"
        )

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

        self.sand_graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_sand()
            self.sand_graph = capture.graph

    def apply_control(self):
        target = self.policy.compute_joint_targets(self.state_0, self.command)
        wp.copy(self.control.joint_target_pos, target)
        if self._action_norm_history is not None:
            with torch.no_grad():
                self._action_norm_history.append(self.policy.last_action.norm().item())

    def simulate_robot(self):
            """
            Advance the rigid-body simulation one frame with two-way sand coupling.

            Each substep:
            1. Apply the sand to body forces accumulated at the end of the last frame.
            2. Save those forces so we can subtract them before the next MPM step.
            3. Step the rigid-body solver (MuJoCo).
            """
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()

                # Apply forces to rigid bodies
                wp.launch(
                    compute_body_forces, 
                    dim=self.collider_impulse_ids.shape[0],
                    inputs=[
                        self.frame_dt, 
                        self.collider_impulse_ids, 
                        self.collider_impulses, 
                        self.collider_impulse_pos, 
                        self.collider_body_id, 
                        self.state_0.body_q, 
                        self.model.body_com, 
                        self.state_0.body_f
                        ])

                # Save forces for subtraction in the next MPM step to avoid double counting
                self.body_sand_forces.assign(self.state_0.body_f)

                # self.viewer.apply_forces(self.state_0) # Empty function call, commented out since forces are applied directly onto the body state in the compute_body_forces kernel
                self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)

                self.state_0, self.state_1 = self.state_1, self.state_0
            

    def simulate_sand(self):
        """
        Advance the MPM sand simulation one frame with two-way coupling.

        Before stepping MPM we subtract the previously applied sand forces from
        the body velocities that are passed to the MPM collider.  This ensures
        the MPM solver sees the *pre-force* velocity when computing the
        complementarity-based frictional contact impulses, avoiding
        double-counting of the reaction forces.
        """
        
        # Remove the previously applied sand forces from body velocities before the MPM step to avoid double-counting

        if self.state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.state_0.body_q.shape[0],
                inputs=[
                    self.frame_dt,
                    self.state_0.body_q, 
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.state_0.body_q,   # body_q_res  (in-place, positions unchanged)
                    self.state_0.body_qd,  # body_qd_res (in-place update)
                ],
            )

        # ---- (4) Step MPM ----
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

        # ---- (5) Collect new impulses for next robot step ----
        self._collect_collider_impulses()

    def precompute(self, num_frames):
        """
        Run the simulation forward `num_frames` frames and cache a snapshot of
        the per-frame rendering state (body_q + particle_q). After this call,
        step() will play back the cached frames in order rather than advancing
        the live simulation, which keeps the on-screen frame rate smooth even
        when the underlying MPM step is slow.
        """
        if num_frames <= 0:
            return

        print(f"Precomputing {num_frames} frames before viewer starts...")
        self._precomputing = True
        try:
            for i in range(num_frames):
                self.step()  # advances live sim and updates state_0

                # Snapshot only what render() actually consumes.
                self._precomputed_body_q.append(
                    wp.clone(self.state_0.body_q) if self.state_0.body_q is not None else None
                )
                self._precomputed_particle_q.append(
                    wp.clone(self.state_0.particle_q) if self.state_0.particle_q is not None else None
                )
                self._precomputed_sim_time.append(float(self.sim_time))

                if (i + 1) % 25 == 0 or (i + 1) == num_frames:
                    print(f"  precomputed {i + 1}/{num_frames} frames")
        finally:
            self._precomputing = False

        # Reset sim_time so playback runs from t=0 in the viewer.
        self.sim_time = 0.0
        self._playback_idx = 0
        print("Precomputation done — switching to playback for the viewer loop.")

    def step(self):
        # If we have a precomputed playback buffer and we're not currently
        # filling it, just copy the next cached snapshot into state_0 and
        # bail out of live simulation entirely.
        if (
            not getattr(self, "_precomputing", False)
            and self._precomputed_body_q
            and self._playback_idx < len(self._precomputed_body_q)
        ):
            idx = self._playback_idx
            cached_body_q = self._precomputed_body_q[idx]
            cached_particle_q = self._precomputed_particle_q[idx]
            if cached_body_q is not None and self.state_0.body_q is not None:
                wp.copy(self.state_0.body_q, cached_body_q)
            if cached_particle_q is not None and self.state_0.particle_q is not None:
                wp.copy(self.state_0.particle_q, cached_particle_q)
            self.sim_time = self._precomputed_sim_time[idx]
            self._playback_idx += 1
            return

        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                # disable forward motion
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1.0

        # compute control before graph/step
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()


        if self._debug_forces:
            body_f_np = self.state_0.body_f.numpy()
            for i, (name, f) in enumerate(zip(self.body_keys, body_f_np)):
                torque = f[:3]
                force = f[3:]
                print(f"  [{i:2d}] {name:<10s}  force=({force[0]:+.3f}, {force[1]:+.3f}, {force[2]:+.3f})  torque=({torque[0]:+.3f}, {torque[1]:+.3f}, {torque[2]:+.3f})")

        # Record MPM->body contact forces on the selected feet for plotting at exit.
        # body_f is the spatial force [torque(0:3), linear force(3:6)]; we keep the
        # linear part. Runs only on live sim steps (playback returns earlier above).
        if self._force_history is not None:
            body_f_np = self.state_0.body_f.numpy()
            for name, f in zip(self.body_keys, body_f_np):
                hist = self._force_history.get(name)
                if hist is None:
                    continue
                force = f[3:]
                hist["times"].append(self.sim_time)
                hist["forces"].append([float(force[0]), float(force[1]), float(force[2])])


        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()
        self.sim_time += self.frame_dt

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 0.9,  # This threshold assumes 100 frames
        )

        forward_vel_min = wp.spatial_vector(-0.2, 0.9, -0.2, -0.8, -0.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.2, 1.1, 0.2, 0.8, 0.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.utils.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )
        voxel_size = self.mpm_solver.mpm_model.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def test_final(self):
        pass

    @staticmethod
    def _apply_material(mpm_model, particles, *, yield_stress, tensile_yield_ratio, yield_pressure, hardening):
        params = mpm_model.material_parameters
        params.yield_stress[particles].fill_(yield_stress)
        params.tensile_yield_ratio[particles].fill_(tensile_yield_ratio)
        params.yield_pressure[particles].fill_(yield_pressure)
        params.hardening[particles].fill_(hardening)

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, voxel_size, bounds_lo, bounds_hi, density, flags,
                         particles_per_cell=3, jitter_multiplier=2.0):
        res = np.array(
            np.ceil(particles_per_cell * (bounds_hi - bounds_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        begin_id = len(builder.particle_q)
        builder.add_particle_grid(
            pos=wp.vec3(bounds_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res[0] + 1,
            dim_y=res[1] + 1,
            dim_z=res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=jitter_multiplier * radius,
            radius_mean=radius,
            flags=flags,
        )

        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=int)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)  # keeps robot rendering
        self.viewer.log_points(
            name="/model/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=False,
        )

        self.viewer.end_frame()

        # Pull the rendered framebuffer and append to the .mp4. Must run AFTER
        # end_frame() — that's what triggers the GL render into the FBO that
        # get_frame() reads from.
        if self._video_writer is not None and hasattr(self.viewer, "get_frame"):
            try:
                frame = self.viewer.get_frame()
                self._video_writer.append_data(frame.numpy())
            except Exception as e:
                print(f"video frame capture failed: {e}")

    def spawn_multiple_materials(self, builder: newton.ModelBuilder, voxel_size):
        materials = self.cfg["materials"]
        spawn_cfg = self.cfg["particle_spawn"]
        ppc = int(spawn_cfg["particles_per_cell"])
        jitter_mul = float(spawn_cfg["jitter_multiplier"])

        def _spawn(material_key):
            mat = materials[material_key]["spawn"]
            return Example._spawn_particles(
                builder,
                voxel_size,
                bounds_lo=np.array(mat["bounds_lo"]),
                bounds_hi=np.array(mat["bounds_hi"]),
                density=float(mat["density"]),
                flags=newton.ParticleFlags.ACTIVE,
                particles_per_cell=ppc,
                jitter_multiplier=jitter_mul,
            )

        thin_mud_particles = _spawn("thin_mud")
        medium_mud_particles = _spawn("medium_mud")
        thick_mud_particles = _spawn("thick_mud")
        return thin_mud_particles, medium_mud_particles, thick_mud_particles


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to hyperparameter YAML (default: config.yaml next to this script)")
    parser.add_argument("--voxel-size", "-dx", type=float, default=None, help="MPM voxel size (overrides mpm.voxel_size in YAML)")
    parser.add_argument("--max-iterations", "-it", type=int, default=None, help="MPM solver max iterations (overrides mpm.max_iterations in YAML)")
    parser.add_argument("--tolerance", "-tol", type=float, default=None, help="MPM solver tolerance (overrides mpm.tolerance in YAML)")
    parser.add_argument("--policy-path", "-cp", type=str, default=None, help="Path to the model checkpoint (.pt or .onnx)")
    parser.add_argument("--precompute-frames", type=int, default=0, help="Run n default sim frames before viewer starts, cache in memory. Viewer plays back cached frames rather than stepping the live sim, eliminating per-frame solver lag.")
    parser.add_argument("--video", type=str, default=None, help="If set, write a real-time .mp4 of the GL framebuffer to this path. Combine with --precompute-frames so playback runs at full fps.")
    parser.add_argument("--video-fps", type=int, default=50, help="Frames per second for the output .mp4 (default 50, matches sim fps).")
    parser.add_argument("--debug-forces", action="store_true", help="Print per-body force/torque every frame (slow; off by default).")
    parser.add_argument("--plot-actions", type=str, default=None, help="If set, save a plot of the policy action L2 norm per step to this path.")
    parser.add_argument("--plot-forces", type=str, default=None, help="If set, save a plot of the MPM particle contact forces applied to the robot feet to this path.")
    parser.add_argument("--plot-forces-foot", type=str, default="FL_calf", help="Which foot/body to record contact forces for (e.g. 'FL', 'FL_calf', 'FL_foot'). Comma-separate for multiple feet. Default: FL_calf.")
    parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude", help="Plot the force magnitude |F| (default) or break it out into X/Y/Z components.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # This example requires a GPU device
    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # Create example and load policy
    example = Example(viewer, args)

    # Optionally precompute frames so the viewer plays them back smoothly.
    if getattr(args, "precompute_frames", 0) > 0:
        example.precompute(args.precompute_frames)

    # Run via unified example runner
    newton.examples.run(example, args)