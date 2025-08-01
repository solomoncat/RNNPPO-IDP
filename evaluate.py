import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable

from environment import PrisonersDilemmaEnv
from agent import PPOAgent
from utils import compute_cooperation_rate, episode_summary


# Baseline policies
class AlwaysCooperate:
    def reset(self):
        pass

    def act(self, obs, deterministic=False):
        # action 0 = cooperate
        return {'action': np.array(0), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}


class AlwaysDefect:
    def reset(self):
        pass

    def act(self, obs, deterministic=False):
        return {'action': np.array(1), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}


class TitForTat:
    def __init__(self):
        self.last_opponent = 0  # start cooperating

    def reset(self):
        self.last_opponent = 0

    def act(self, obs, deterministic=False):
        return {'action': np.array(self.last_opponent), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}

    def update_opponent(self, opponent_action):
        self.last_opponent = int(opponent_action)


class GrimTrigger:
    def __init__(self):
        self.betrayed = False

    def reset(self):
        self.betrayed = False

    def act(self, obs, deterministic=False):
        if self.betrayed:
            return {'action': np.array(1), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}
        return {'action': np.array(0), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}

    def update_opponent(self, opponent_action):
        if int(opponent_action) == 1:
            self.betrayed = True


def play_episode(env: PrisonersDilemmaEnv, agent_A, agent_B, max_steps: int):
    obs = env.reset()
    obs_A = obs['A']
    obs_B = obs['B']
    # reset hidden if exists
    if hasattr(agent_A, 'reset_hidden'):
        agent_A.reset_hidden(batch_size=1)
    if hasattr(agent_B, 'reset_hidden'):
        agent_B.reset_hidden(batch_size=1)

    history = []
    rewards_A = []
    rewards_B = []
    for t in range(max_steps):
        outA = agent_A.act(obs_A)
        outB = agent_B.act(obs_B)
        aA = int(outA['action'])
        aB = int(outB['action'])
        next_obs, rewards, done, _ = env.step({'A': aA, 'B': aB})
        obs_A = next_obs['A']
        obs_B = next_obs['B']
        history.append((aA, aB, env.current_temptation))
        rewards_A.append(rewards['A'])
        rewards_B.append(rewards['B'])

        # update baseline if needed
        if hasattr(agent_A, 'update_opponent'):
            agent_A.update_opponent(aB)
        if hasattr(agent_B, 'update_opponent'):
            agent_B.update_opponent(aA)

        if done:
            break

    return {
        'history': history,
        'rewards_A': rewards_A,
        'rewards_B': rewards_B
    }


def evaluate_pair(agent, opponent_factory: Callable[[], object], env_kwargs: dict, episodes: int = 100, max_steps=100):
    stats = []
    for ep in range(episodes):
        env = PrisonersDilemmaEnv(**env_kwargs)
        opp = opponent_factory()
        if hasattr(opp, 'reset'):
            opp.reset()
        # determine who is policy vs baseline depending if shared/self-play
        result = play_episode(env, agent, opp, max_steps)
        coop_rate = compute_cooperation_rate(result['history'])
        summary = episode_summary(result['rewards_A'], result['rewards_B'])
        stats.append({
            'coop_rate': coop_rate,
            **summary
        })
    # aggregate
    coop = np.mean([s['coop_rate'] for s in stats])
    sumA = np.mean([s['sum_A'] for s in stats])
    sumB = np.mean([s['sum_B'] for s in stats])
    meanA = np.mean([s['mean_A'] for s in stats])
    meanB = np.mean([s['mean_B'] for s in stats])
    return {
        'cooperation_rate': coop,
        'sum_A': sumA,
        'sum_B': sumB,
        'mean_A': meanA,
        'mean_B': meanB
    }


def plot_episode(result: dict, title: str = None):
    history = result['history']
    rewards_A = result['rewards_A']
    rewar
