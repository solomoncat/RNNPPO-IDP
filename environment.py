import numpy as np
import gym
from gym import spaces

class PrisonersDilemmaEnv(gym.Env):
    def __init__(self, max_steps=100, history_length=10, temptation_range=(3.0, 10.0), seed=None):
        self.max_steps = max_steps
        self.history_length = history_length
        self.t_low, self.t_high = temptation_range
        self.rng = np.random.RandomState(seed)
        
        self.R = 3.0
        self.P = 1.0
        self.S = 0.0
        
        self.action_space = spaces.Discrete(2)
        obs_size = history_length * 3 + 1  # (action_A, action_B, temptation) * history_length + current_temptation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.history = []
        self.steps = 0
        self.current_temptation = float(self.rng.uniform(self.t_low, self.t_high))
        return self._get_obs()
    
    def step(self, actions):
        if isinstance(actions, dict):
            aA = int(actions['A'])
            aB = int(actions['B'])
        else:
            aA, aB = int(actions[0]), int(actions[1])
        
        rewards = self._calculate_rewards(aA, aB)
        
        self.history.append((aA, aB, self.current_temptation))
        self.steps += 1
        
        done = self.steps >= self.max_steps
        if not done:
            self.current_temptation = float(self.rng.uniform(self.t_low, self.t_high))
        
        obs = self._get_obs()
        return obs, rewards, done, {}
    
    def _calculate_rewards(self, aA, aB):
        T = self.current_temptation
        if aA == 0 and aB == 0:
            rA, rB = self.R, self.R
        elif aA == 1 and aB == 0:
            rA, rB = T, self.S
        elif aA == 0 and aB == 1:
            rA, rB = self.S, T
        else:
            rA, rB = self.P, self.P
        return {'A': float(rA), 'B': float(rB)}
    
    def _get_obs(self):
        h = self.history[-self.history_length:]
        pad = self.history_length - len(h)
        if pad > 0:
            h = [(0, 0, 0.0)] * pad + h
        
        obs_A = []
        obs_B = []
        
        for aA, aB, temptation in h:
            obs_A.extend([aA, aB, temptation])
            obs_B.extend([aB, aA, temptation])
        
        obs_A.append(self.current_temptation)
        obs_B.append(self.current_temptation)
        
        return {
            'A': np.array(obs_A, dtype=np.float32),
            'B': np.array(obs_B, dtype=np.float32)
        }
