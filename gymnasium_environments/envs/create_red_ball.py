import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps":4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Discrete(10)  # arbitrary for now
        self.action_space = spaces.Discrete(4)        # arbitrary for now
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        info = {}
        return observation, info

    def step(self, action):
        observation = self.observation_space.sample()
        reward = 0.0  # arbitrary reward
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        pass 
        
    def close(self):
        pass
