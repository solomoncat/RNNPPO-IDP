import numpy as np
import gym
from gym import spaces

class PrisonersDilemmaEnv(gym.Env):
    """
    Iterated Prisoner's Dilemma environment with varying temptation values.
    
    The game matrix is:
    - Both cooperate (0,0): (R, R) = (3, 3)  
    - Agent A defects, B cooperates (1,0): (T, S) where T is variable temptation, S = 0
    - Agent A cooperates, B defects (0,1): (S, T) = (0, T)
    - Both defect (1,1): (P, P) = (1, 1)
    
    Actions:
    - 0: Cooperate
    - 1: Defect
    
    Observations include:
    - History of previous actions for both agents
    - History of temptation values
    - Current temptation value
    """
    
    def __init__(self, max_steps=100, history_length=10, temptation_range=(3.0, 10.0), seed=None):
        super().__init__()
        
        self.max_steps = max_steps
        self.history_length = history_length
        self.t_low, self.t_high = temptation_range
        self.rng = np.random.RandomState(seed)
        
        # Payoff matrix values (fixed except for temptation T)
        self.R = 3.0  # Reward for mutual cooperation
        self.P = 1.0  # Punishment for mutual defection  
        self.S = 0.0  # Sucker's payoff
        # T = temptation (variable)
        
        # Action space: 0 = cooperate, 1 = defect
        self.action_space = spaces.Discrete(2)
        
        # Observation space: (action_A, action_B, temptation) * history_length + current_temptation
        obs_size = history_length * 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment for new episode"""
        self.history = []
        self.steps = 0
        self.current_temptation = float(self.rng.uniform(self.t_low, self.t_high))
        return self._get_obs()
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: dict with keys 'A' and 'B' containing integer actions (0 or 1)
                    or tuple/list [action_A, action_B]
        
        Returns:
            observations: dict with keys 'A' and 'B' containing observation arrays
            rewards: dict with keys 'A' and 'B' containing reward values
            done: boolean indicating if episode is finished
            info: dict with additional information
        """
        # Handle different action input formats
        if isinstance(actions, dict):
            aA = int(actions['A'])
            aB = int(actions['B'])
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            aA, aB = int(actions[0]), int(actions[1])
        else:
            raise ValueError(f"Actions must be dict with 'A','B' keys or length-2 sequence, got {actions}")
        
        # Calculate rewards based on current temptation
        rewards = self._calculate_rewards(aA, aB)
        
        # Update history
        self.history.append((aA, aB, self.current_temptation))
        self.steps += 1
        
        # Check if episode is done
        done = self.steps >= self.max_steps
        
        # Generate new temptation value for next step (if not done)
        if not done:
            self.current_temptation = float(self.rng.uniform(self.t_low, self.t_high))
        
        # Get new observations
        obs = self._get_obs()
        
        # Additional info
        info = {
            'steps': self.steps,
            'temptation': self.current_temptation,
            'last_actions': (aA, aB),
            'cooperation_rate': self._get_cooperation_rate()
        }
        
        return obs, rewards, done, info
    
    def _calculate_rewards(self, aA, aB):
        """Calculate rewards for both agents given their actions"""
        T = self.current_temptation
        
        if aA == 0 and aB == 0:
            # Both cooperate
            rA, rB = self.R, self.R
        elif aA == 1 and aB == 0:
            # A defects, B cooperates
            rA, rB = T, self.S
        elif aA == 0 and aB == 1:
            # A cooperates, B defects
            rA, rB = self.S, T
        else:
            # Both defect
            rA, rB = self.P, self.P
        
        return {'A': float(rA), 'B': float(rB)}
    
    def _get_obs(self):
        """Generate observations for both agents based on history"""
        # Get recent history (pad with zeros if necessary)
        h = self.history[-self.history_length:]
        pad_length = self.history_length - len(h)
        if pad_length > 0:
            # Pad with (0, 0, 0.0) - no actions, no temptation
            padding = [(0, 0, 0.0)] * pad_length
            h = padding + h
        
        # Build observations
        obs_A = []
        obs_B = []
        
        # Add history: each agent sees (own_action, opponent_action, temptation)
        for aA, aB, temptation in h:
            # Agent A's perspective: (my_action, opponent_action, temptation)
            obs_A.extend([aA, aB, temptation])
            # Agent B's perspective: (my_action, opponent_action, temptation)  
            obs_B.extend([aB, aA, temptation])
        
        # Add current temptation value
        obs_A.append(self.current_temptation)
        obs_B.append(self.current_temptation)
        
        return {
            'A': np.array(obs_A, dtype=np.float32),
            'B': np.array(obs_B, dtype=np.float32)
        }
    
    def _get_cooperation_rate(self):
        """Calculate cooperation rate from current history"""
        if not self.history:
            return 0.0
        
        cooperations = sum(1 for aA, aB, _ in self.history if aA == 0 and aB == 0)
        return cooperations / len(self.history)
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            if self.history:
                last_aA, last_aB, last_T = self.history[-1]
                action_names = {0: 'Cooperate', 1: 'Defect'}
                print(f"Step {self.steps}: A={action_names[last_aA]}, B={action_names[last_aB]}, T={last_T:.2f}")
                print(f"Cooperation rate: {self._get_cooperation_rate():.3f}")
            else:
                print("Episode not started")
    
    def close(self):
        """Clean up environment"""
        pass
    
    def seed(self, seed=None):
        """Set random seed"""
        self.rng = np.random.RandomState(seed)
        return [seed]

    def get_history_summary(self):
        """Get summary statistics of the current episode"""
        if not self.history:
            return {}
        
        actions_A = [h[0] for h in self.history]
        actions_B = [h[1] for h in self.history]
        temptations = [h[2] for h in self.history]
        
        return {
            'episode_length': len(self.history),
            'cooperation_rate_A': (np.array(actions_A) == 0).mean(),
            'cooperation_rate_B': (np.array(actions_B) == 0).mean(),
            'mutual_cooperation_rate': self._get_cooperation_rate(),
            'avg_temptation': np.mean(temptations),
            'min_temptation': np.min(temptations),
            'max_temptation': np.max(temptations)
        }


# Utility function for creating multiple environments
def make_env(env_id=0, **kwargs):
    """Factory function for creating environments with different seeds"""
    seed = kwargs.get('seed', None)
    if seed is not None:
        seed = seed + env_id
    env_kwargs = {**kwargs, 'seed': seed}
    return PrisonersDilemmaEnv(**env_kwargs)


# Example usage and testing
def test_environment():
    """Test the environment functionality"""
    print("Testing Prisoner's Dilemma Environment...")
    
    env = PrisonersDilemmaEnv(max_steps=10, history_length=3, seed=42)
    obs = env.reset()
    
    print(f"Initial observation shapes: A={obs['A'].shape}, B={obs['B'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test different action combinations
    test_actions = [
        ({'A': 0, 'B': 0}, "Both cooperate"),
        ({'A': 1, 'B': 0}, "A defects, B cooperates"), 
        ({'A': 0, 'B': 1}, "A cooperates, B defects"),
        ({'A': 1, 'B': 1}, "Both defect")
    ]
    
    for actions, description in test_actions:
        obs, rewards, done, info = env.step(actions)
        print(f"{description}: Rewards A={rewards['A']:.1f}, B={rewards['B']:.1f}, T={info['temptation']:.2f}")
        
        if done:
            print("Episode finished!")
            break
    
    # Test episode summary
    summary = env.get_history_summary()
    print(f"\nEpisode summary: {summary}")
    
    print("Environment test completed successfully!")


if __name__ == "__main__":
    test_environment()
