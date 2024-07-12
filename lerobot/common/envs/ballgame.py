from typing import List, Tuple

import cv2
import gymnasium as gym
import numpy as np


class BallgameEnv(gym.Env):
    """
    The maze-ball game environment.
    """

    DEFAULT_MAZE = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    DEFAULT_KEY_LOCATIONS = [(1.5, 6.5), (6.5, 1.5), (6.5, 6.5)]
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 800
    DEFAULT_CELL_SIZE = DEFAULT_WIDTH // 8
    DEFAULT_BALL_RADIUS = DEFAULT_CELL_SIZE / 6
    DEFAULT_FPS = 60
    DEFAULT_FORCE_MAGNITUDE = 0.1
    DEFAULT_MASS = 1
    DEFAULT_DAMPING = 0.99
    DEFAULT_TIMEOUT = 1000
    SQRT_2 = 2**0.5
    RENDER_MODES = ["human", "rgb_array"]

    metadata = {"render_modes": RENDER_MODES, "render_fps": DEFAULT_FPS}

    def __init__(
        self,
        obs_type: str = "state",
        maze: np.ndarray = DEFAULT_MAZE,
        key_locations: List[Tuple[int, int]] = DEFAULT_KEY_LOCATIONS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        cell_size: int = DEFAULT_CELL_SIZE,
        ball_radius: int = DEFAULT_BALL_RADIUS,
        force_magnitude: float = DEFAULT_FORCE_MAGNITUDE,
        mass: float = DEFAULT_MASS,
        damping: float = DEFAULT_DAMPING,
        fps: float = DEFAULT_FPS,
        timeout: int = DEFAULT_TIMEOUT,
        reset_location_noise_scale: float = 0.1,
    ):
        self.maze = maze
        self.key_locations = key_locations
        self.reset_location_noise_scale = reset_location_noise_scale
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.ball_radius = ball_radius
        self.force_magnitude = force_magnitude
        self.mass = mass
        self.damping = damping
        self.fps = fps
        # Determine the action and the observation spaces.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._setup_observation_space(obs_type)
        self.obs_type = obs_type
        self.episode_length = timeout
        self.reset()

    def _setup_observation_space(self, obs_type: str):
        if obs_type == "state":
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif obs_type == "image":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        elif obs_type == "pixels_agent_pos":
            self.observation_space = gym.spaces.Dict(
                {
                    "pixels": gym.spaces.Box(
                        low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
                    ),
                    "agent_pos": gym.spaces.Box(
                        low=np.array([0.0, 0.0, -np.inf, -np.inf]),
                        high=np.array([self.height, self.width, np.inf, np.inf]),
                        shape=(4,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

    def reset(self, seed=None):
        self.ball = np.array([1.5, 1.5]) * self.cell_size
        if seed is not None:
            np.random.seed(seed)
        self.ball += self.cell_size * np.random.normal(scale=self.reset_location_noise_scale, size=2)
        self.ball_velocity = np.array([0, 0], dtype=np.float32)
        self.trail = []
        obs = self._get_observation()
        return obs, {"trail": np.zeros((0, 2)), "is_success": False}

    def _get_observation(self, obs_type: str = None):
        obs_type = obs_type or self.obs_type
        if obs_type == "state":
            return np.concatenate([self.ball, self.ball_velocity])
        elif obs_type == "image":
            return self.render(render_mode="rgb_array")
        elif obs_type == "pixels_agent_pos":
            return {
                "pixels": self.render(render_mode="rgb_array"),
                "agent_pos": np.concatenate([self.ball, self.ball_velocity]),
            }

    def step(self, action: np.ndarray):
        reached_goal, reached_timeout = False, False
        reward = 0
        # First, ensure the action has the right shape.
        assert action.shape == (2,), f"Action shape should be (2,) but got {action.shape}"

        # Apply the forces.
        self.ball_velocity += action * self.force_magnitude / self.mass

        # Then update the physics.
        self.ball_velocity *= self.damping
        next_ball = self.ball + self.ball_velocity

        col = int(next_ball[0] // self.cell_size)
        row = int(next_ball[1] // self.cell_size)

        if self.maze[row, col] == 1:
            if self.maze[int(self.ball[1] // self.cell_size), col] == 1:
                self.ball_velocity[0] *= -1
            if self.maze[row, int(self.ball[0] // self.cell_size)] == 1:
                self.ball_velocity[1] *= -1
        else:
            self.ball = next_ball

        # check for reaching keys.
        for key in self.key_locations:
            dist = np.linalg.norm(self.ball - np.array(key) * self.cell_size)
            if dist < self.SQRT_2 * self.ball_radius:
                reached_goal = True
                reward = 1
                break

        self.trail.append(self.ball.copy())
        if len(self.trail) > self.episode_length:
            reached_timeout = True
            reward = -1

        return (
            self._get_observation(),
            reward,
            reached_goal,
            reached_timeout,
            {"trail": np.stack(self.trail), "is_success": reached_goal},
        )

    def render(self, render_mode="rgb_array"):
        # Render the environment with CV2.
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        for row in range(len(self.maze)):
            for col in range(len(self.maze[row])):
                color = (0, 0, 0) if self.maze[row, col] == 1 else (255, 255, 255)
                img[
                    row * self.cell_size : (row + 1) * self.cell_size,
                    col * self.cell_size : (col + 1) * self.cell_size,
                ] = color

        for key in self.key_locations:
            cv2.circle(
                img,
                tuple(int(x) for x in np.array(key) * self.cell_size),
                int(self.ball_radius),
                (0, 0, 255),
                -1,
            )

        for i in range(len(self.trail) - 1):
            cv2.line(
                img,
                tuple(int(x) for x in self.trail[i]),
                tuple(int(x) for x in self.trail[i + 1]),
                (255, 0, 0),
                2,
            )

        cv2.circle(
            img,
            tuple(int(x) for x in self.ball),
            int(self.ball_radius),
            (255, 0, 0),
            -1,
        )

        if render_mode == "human":
            cv2.imshow("Ballgame", img)
            cv2.waitKey(1)
        elif render_mode == "rgb_array":
            # convert image to RGB np array.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    env = BallgameEnv()
    obs = env.reset()
    trunc, timeout = False, False
    while not (trunc or timeout):
        action = env.action_space.sample()
        obs, reward, trunc, timeout, _ = env.step(action)
        env.render()
    cv2.destroyAllWindows()
