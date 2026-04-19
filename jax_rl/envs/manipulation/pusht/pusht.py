import collections
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    """
    ## Description

    PushT environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices.
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

    If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

    ## Rewards

    The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The agent starts at a random position and the block starts at a random position and angle.

    ## Episode Termination

    The episode terminates when the block is at least 95% in the goal zone.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state`, `keypoints`, `pixels` or `pixels_agent_pos`.
      Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `observation_width`: (int) The width of the observed image. Default is `96`.

    * `observation_height`: (int) The height of the observed image. Default is `96`.

    * `visualization_width`: (int) The width of the visualized image. Default is `680`.

    * `visualization_height`: (int) The height of the visualized image. Default is `680`.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
        reward_mode="coverage",
    ):
        """
        Args:
            reward_mode: reward function to return from step().
                "coverage" — DP default. `clip(coverage / 0.95, 0, 1)`.
                "sparse"   — 1.0 iff coverage > 0.95 else 0.0.
                "shaped"   — coverage + 0.01 * contact_count - 0.001 * pusher_to_block.
                "approach" — coverage + 0.05 * (1 - pusher_to_block / max_dist).
                "dense"    — coverage + pose-error shaping + block velocity toward goal
                             + contact + approach. Recommended for RL from scratch.
                Pick to bootstrap RL (coverage alone is too sparse for from-scratch RL).
        """
        super().__init__()
        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = 0.95  # 95% coverage
        self.reward_mode = reward_mode
        if reward_mode not in ("coverage", "sparse", "shaped", "approach", "dense", "contact_gated"):
            raise ValueError(
                f"Unknown reward_mode {reward_mode!r}. Must be one of "
                "[coverage, sparse, shaped, approach, dense, contact_gated]."
            )

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([512, 512, 512, 512, 2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "environment_state": spaces.Box(
                        low=np.zeros(16),
                        high=np.full((16,), 512),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, environment_state_agent_pos, "
                "pixels_agent_pos]"
            )

    def _get_coverage(self):
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

        # Coverage always computed (ground-truth objective + used by info/success).
        coverage = self._get_coverage()
        is_success = coverage > self.success_threshold
        coverage_clip = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))

        # Geometry signals for shaping.
        block_pos = np.array(self.block.position)
        agent_pos = np.array(self.agent.position)
        goal_xy = self.goal_pose[:2]
        goal_yaw = self.goal_pose[2]
        pusher_to_block = float(np.linalg.norm(agent_pos - block_pos))
        block_to_goal = float(np.linalg.norm(block_pos - goal_xy))
        # Shortest angular distance. Block angles aren't wrapped by pymunk,
        # so use arctan2(sin(Δ), cos(Δ)) for continuity.
        block_angle = float(self.block.angle)
        angle_err = abs(np.arctan2(
            np.sin(block_angle - goal_yaw), np.cos(block_angle - goal_yaw)
        ))

        # Block velocity toward goal (positive = approaching).
        block_vel = np.array(self.block.velocity)
        to_goal = goal_xy - block_pos
        to_goal_norm = float(np.linalg.norm(to_goal))
        if to_goal_norm > 1e-6:
            block_vel_toward = float(np.dot(block_vel, to_goal / to_goal_norm))
        else:
            block_vel_toward = 0.0

        # Scale constants. Arena is 512 px; normalize so shaping terms
        # stay roughly in [0, 1] scale and never dominate coverage.
        max_dist = 512 * np.sqrt(2)
        max_pos_err = 512.0  # rough upper bound

        r_coverage    = coverage_clip                             # [0, 1]
        r_pos         = 1.0 - min(block_to_goal / max_pos_err, 1.0)   # [0, 1]
        r_angle       = 1.0 - min(angle_err / np.pi, 1.0)             # [0, 1]
        r_approach    = 1.0 - min(pusher_to_block / max_dist, 1.0)    # [0, 1]
        # Block velocity reward: small constant scale, positive when progressing.
        # pymunk velocities in px/s; 50 px/s ≈ "decent progress". Clip to ±1.
        r_block_vel   = float(np.clip(block_vel_toward / 50.0, -1.0, 1.0))
        r_contact     = 0.01 * float(self.n_contact_points)
        r_success     = 5.0 if is_success else 0.0

        if self.reward_mode == "coverage":
            # DP default — scaled IoU only.
            reward = r_coverage
        elif self.reward_mode == "sparse":
            reward = float(is_success)
        elif self.reward_mode == "shaped":
            # Coverage + small contact bonus + tiny approach penalty.
            reward = (
                r_coverage
                + 0.01 * float(self.n_contact_points)
                - 0.001 * pusher_to_block
            )
        elif self.reward_mode == "approach":
            # Coverage + smooth proximity bonus.
            reward = r_coverage + 0.05 * r_approach
        elif self.reward_mode == "contact_gated":
            # Published recipe: Cong et al. 2023 (Frontiers Neurorobotics,
            # contact-based pushing) + Ferrari et al. 2025 (directional
            # unit-vector progress reward). Key idea: shape on CONTACT STATE.
            #
            # New: add "pusher-on-correct-side" — structural prior that for
            # pushing block toward goal, pusher must be on the far side of
            # block from goal. Standard in robotic pushing literature.
            in_contact = self.n_contact_points > 0

            # σ=50px: exp(-d/50) ≈ 0.37 at 50px, ≈ 0.05 at 150px.
            approach_bonus = float(np.exp(-pusher_to_block / 50.0))
            r_approach_gated = (not in_contact) * approach_bonus

            # Pusher-on-correct-side (NEW). Unit vector from pusher→block
            # dotted with block→goal direction. +1 when pusher is behind
            # block relative to goal, -1 when in front.
            pb_vec = block_pos - agent_pos
            pb_norm = float(np.linalg.norm(pb_vec)) + 1e-6
            r_correct_side = float(np.dot(
                pb_vec / pb_norm, (goal_xy - block_pos) / (to_goal_norm + 1e-6)
            ))

            # Unit-vector dot product of block velocity and to-goal vector.
            # Magnitude-free by construction (Ferrari 2025). Range [-1, 1].
            vel_norm = float(np.linalg.norm(block_vel)) + 1e-6
            v_dot = float(np.dot(block_vel, goal_xy - block_pos) / (vel_norm * (to_goal_norm + 1e-6)))
            r_dir_gated = (1.0 if in_contact else 0.0) * v_dot

            # Angle-delta — rewards rotating block toward goal yaw. Gated on
            # contact because otherwise block doesn't rotate anyway.
            prev_angle = getattr(self, '_prev_angle_err', angle_err)
            r_angle_delta_gated = (1.0 if in_contact else 0.0) * (prev_angle - angle_err)

            reward = (
                r_coverage                        # [0, 1]      — task objective
                + 0.2 * r_approach_gated          # [0, 0.2]    — NOT in contact
                + 0.15 * r_correct_side           # [-0.15, 0.15] — always on
                + 0.3 * r_dir_gated               # [-0.3, 0.3] — IN contact
                + 2.0 * r_angle_delta_gated       # telescopes, bounded
                + (50.0 if is_success else 0.0)   # big success salience
            )
            self._prev_angle_err = angle_err

        elif self.reward_mode == "dense":
            # Delta shaping — only rewards *progress*, not static state.
            #
            # Absolute-bonus shaping (e.g. "0.3 * (1 - pos_err/max)") gives the
            # policy free reward for block simply being near goal, even with
            # the policy doing nothing. That farming was what killed the
            # previous dense-mode run (eval 115, coverage 0%). With deltas,
            # stationary block → zero shaping reward → policy forced to
            # actually solve. Telescopes: total delta reward = total progress
            # made, naturally bounded by initial_err.
            #
            # Init guards: `_prev_*_err` initialized in reset().
            r_pos_delta   = getattr(self, '_prev_pos_err', block_to_goal) - block_to_goal
            r_angle_delta = getattr(self, '_prev_angle_err', angle_err) - angle_err
            reward = (
                r_coverage
                + 0.01 * r_pos_delta     # ~2 total over episode if solved (512 → 0)
                + 0.50 * r_angle_delta   # ~1.5 total over episode if solved (π → 0)
                + r_contact              # 0.01 per contact point — encourages touching
                + r_success              # 5 terminal bonus
            )
            self._prev_pos_err = block_to_goal
            self._prev_angle_err = angle_err

        # Terminate on success per DP convention (can be disabled later).
        terminated = bool(is_success)

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage"] = coverage
        info["pusher_to_block"] = pusher_to_block
        info["block_to_goal"] = block_to_goal
        info["angle_err"] = angle_err
        info["block_vel_toward"] = block_vel_toward
        info["n_contact_points"] = int(self.n_contact_points)
        # Per-component reward split — useful diagnostic regardless of mode.
        info["r_coverage"] = r_coverage
        info["r_pos"] = r_pos
        info["r_angle"] = r_angle
        info["r_approach"] = r_approach
        info["r_block_vel"] = r_block_vel
        info["r_contact"] = r_contact
        info["r_success"] = r_success

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            # we use the env's RNG instead of global np.random
            state = np.array(
                [
                    self.np_random.integers(50, 450),
                    self.np_random.integers(50, 450),
                    self.np_random.integers(100, 400),
                    self.np_random.integers(100, 400),
                    self.np_random.uniform(-np.pi, np.pi),
                ]
            )
        self._set_state(state)

        # Init delta-shaping state for dense reward mode.
        block_pos = np.array(self.block.position)
        goal_xy = self.goal_pose[:2]
        goal_yaw = self.goal_pose[2]
        self._prev_pos_err = float(np.linalg.norm(block_pos - goal_xy))
        self._prev_angle_err = float(abs(np.arctan2(
            np.sin(float(self.block.angle) - goal_yaw),
            np.cos(float(self.block.angle) - goal_yaw),
        )))

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # Draw goal pose
        goal_body = self.get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / 512 * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(screen, width=width, height=height, render_action=visualize)
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz)))
            pygame.display.update()
        else:
            raise ValueError(self.render_mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float64)

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints(self._block_shapes).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        # Add walls
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone
        self.agent = self.add_circle(self.space, (256, 400), 15)
        self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.position = list(state[2:4])
        self.block.angle = state[4]

        # Run physics to take effect
        self.space.step(self.dt)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    @staticmethod
    def get_keypoints(block_shapes):
        """Get a (8, 2) numpy array with the T keypoints.

        The T is composed of two rectangles each with 4 keypoints.

        0───────────1
        │           │
        3───4───5───2
            │   │
            │   │
            │   │
            │   │
            7───6
        """
        keypoints = []
        for shape in block_shapes:
            for v in shape.get_vertices():
                v = v.rotated(shape.body.angle)
                v = v + shape.body.position
                keypoints.append(np.array(v))
        return np.row_stack(keypoints)
