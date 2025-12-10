import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridWorldAlternate(gym.Env):
    """
    DynaMITE-RL GridWorld (Appendix A.4.1 스타일)
    - 5x5 grid
    - 두 개의 goal, 그 중 하나만 +1, 나머지 0
    - 비-goal 칸은 -0.1
    - latent goal index는 Bernoulli(p)로 세션 종료, 전이행렬 [[0.2, 0.8], [0.8, 0.2]]
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps=60, switch_prob=0.07):
        super().__init__()
        self.grid_size = 5
        self.max_steps = max_steps
        self.switch_prob = switch_prob

        # [ax, ay, g1x, g1y, g2x, g2y] / 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # 논문과 동일하게 4방향
        self.action_space = spaces.Discrete(4)  # 0:U, 1:D, 2:L, 3:R

        self.goals = None
        self.agent_pos = None
        self.active_goal_idx = 0
        self.steps = 0

    def _get_obs(self):
        obs = np.concatenate([self.agent_pos, self.goals[0], self.goals[1]]) / (
            self.grid_size - 1
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        rng = self.np_random

        # agent + 2 goals, 겹치지 않게
        positions = rng.choice(self.grid_size * self.grid_size, 3, replace=False)
        self.agent_pos = np.array(
            [positions[0] // self.grid_size, positions[0] % self.grid_size], dtype=int
        )
        self.goals = [
            np.array([positions[1] // self.grid_size, positions[1] % self.grid_size], dtype=int),
            np.array([positions[2] // self.grid_size, positions[2] % self.grid_size], dtype=int),
        ]

        # 현재 세션의 rewarding goal index
        self.active_goal_idx = rng.integers(0, 2)

        obs = self._get_obs()
        info = {"task_index": self.active_goal_idx, "session_changed": False}
        return obs, info

    def step(self, action):
        # 4방향 이동
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
        dy, dx = moves[int(action)]
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dy, 0, self.grid_size - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dx, 0, self.grid_size - 1)

        self.steps += 1

        reward = -0.1
        active_goal_pos = self.goals[self.active_goal_idx]

        if np.array_equal(self.agent_pos, active_goal_pos):
            reward = 1.0
        elif np.array_equal(self.agent_pos, self.goals[1 - self.active_goal_idx]):
            reward = 0.0  # 비보상 goal은 0

        # latent switch (세션 종료)
        session_changed = False
        if self.np_random.random() < self.switch_prob:
            session_changed = True
            if self.active_goal_idx == 0:
                self.active_goal_idx = self.np_random.choice([0, 1], p=[0.2, 0.8])
            else:
                self.active_goal_idx = self.np_random.choice([0, 1], p=[0.8, 0.2])

        terminated = False  # goal 도달해도 episode 계속
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "task_index": self.active_goal_idx,
            "session_changed": session_changed,
        }
        return obs, reward, terminated, truncated, info
