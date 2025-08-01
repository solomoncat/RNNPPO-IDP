import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable

from environment import PrisonersDilemmaEnv
from agent import PPOAgent
from utils import compute_cooperation_rate, episode_summary


# Baseline policies
class AlwaysCooperate:
    def __init__(self):
        pass
        
    def reset(self):
        pass

    def act(self, obs, deterministic=False):
        # action 0 = cooperate
        return {'action': np.array(0), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}


class AlwaysDefect:
    def __init__(self):
        pass
        
    def reset(self):
        pass

    def act(self, obs, deterministic=False):
        # action 1 = defect
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


class RandomPolicy:
    def __init__(self, coop_prob=0.5):
        self.coop_prob = coop_prob
        self.rng = np.random.RandomState()
    
    def reset(self):
        pass
    
    def act(self, obs, deterministic=False):
        action = 0 if self.rng.random() < self.coop_prob else 1
        return {'action': np.array(action), 'log_prob': 0.0, 'value': 0.0, 'entropy': 0.0}


def play_episode(env: PrisonersDilemmaEnv, agent_A, agent_B, max_steps: int):
    """Play a single episode between two agents"""
    obs = env.reset()
    obs_A = obs['A']
    obs_B = obs['B']
    
    # Reset agents if they have reset methods
    if hasattr(agent_A, 'reset'):
        agent_A.reset()
    if hasattr(agent_B, 'reset'):
        agent_B.reset()
    if hasattr(agent_A, 'reset_hidden'):
        agent_A.reset_hidden(batch_size=1)
    if hasattr(agent_B, 'reset_hidden'):
        agent_B.reset_hidden(batch_size=1)

    history = []
    rewards_A = []
    rewards_B = []
    
    for t in range(max_steps):
        # Get actions
        outA = agent_A.act(obs_A, deterministic=True)
        outB = agent_B.act(obs_B, deterministic=True)
        
        aA = int(outA['action'])
        aB = int(outB['action'])
        
        # Step environment
        next_obs, rewards, done, _ = env.step({'A': aA, 'B': aB})
        obs_A = next_obs['A']
        obs_B = next_obs['B']
        
        # Record
        history.append((aA, aB, env.current_temptation))
        rewards_A.append(rewards['A'])
        rewards_B.append(rewards['B'])

        # Update baseline agents if needed
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
    """Evaluate agent against a specific opponent type"""
    stats = []
    
    for ep in range(episodes):
        env = PrisonersDilemmaEnv(**env_kwargs)
        opp = opponent_factory()
        
        # Play episode
        result = play_episode(env, agent, opp, max_steps)
        
        # Compute statistics
        coop_rate = compute_cooperation_rate(result['history'])
        summary = episode_summary(result['rewards_A'], result['rewards_B'])
        
        stats.append({
            'coop_rate': coop_rate,
            **summary
        })
    
    # Aggregate statistics
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
        'mean_B': meanB,
        'episodes': episodes
    }


def strategy_analysis(agent, env_kwargs: dict, baselines: dict, episodes: int = 100, max_steps: int = 100):
    """Complete strategy analysis against baseline opponents"""
    results = {}
    
    print("Running strategy analysis...")
    
    for name, baseline_class in baselines.items():
        print(f"  Evaluating against {name}...")
        
        result = evaluate_pair(
            agent=agent,
            opponent_factory=baseline_class,
            env_kwargs=env_kwargs,
            episodes=episodes,
            max_steps=max_steps
        )
        
        results[name] = result
        
        print(f"    Cooperation rate: {result['cooperation_rate']:.3f}")
        print(f"    Agent reward: {result['mean_A']:.3f} (avg), {result['sum_A']:.1f} (total)")
        print(f"    Opponent reward: {result['mean_B']:.3f} (avg), {result['sum_B']:.1f} (total)")
    
    # Self-play evaluation
    print("  Evaluating self-play...")
    self_play_result = evaluate_self_play(agent, env_kwargs, episodes, max_steps)
    results['SelfPlay'] = self_play_result
    
    print(f"    Self-play cooperation rate: {self_play_result['cooperation_rate']:.3f}")
    print(f"    Self-play reward: {self_play_result['mean_A']:.3f} (avg)")
    
    return results


def evaluate_self_play(agent, env_kwargs: dict, episodes: int = 100, max_steps: int = 100):
    """Evaluate agent in self-play"""
    stats = []
    
    for ep in range(episodes):
        env = PrisonersDilemmaEnv(**env_kwargs)
        
        # Both agents are the same (but with independent hidden states)
        agent.reset_hidden(batch_size=1)  # Reset for agent A
        # We'll need to track separate hidden states manually
        
        obs = env.reset()
        obs_A = obs['A']
        obs_B = obs['B']
        
        history = []
        rewards_A = []
        rewards_B = []
        
        # Store initial hidden state for agent B
        agent_B_hidden = agent.net.initial_state(1, device=agent.device)
        
        for t in range(max_steps):
            # Agent A acts (uses stored hidden state)
            outA = agent.act(obs_A, deterministic=True)
            aA = int(outA['action'])
            
            # Agent B acts (uses separate hidden state)
            current_hidden = agent.hidden_state
            agent.hidden_state = agent_B_hidden
            outB = agent.act(obs_B, deterministic=True)
            agent_B_hidden = agent.hidden_state
            agent.hidden_state = current_hidden
            aB = int(outB['action'])
            
            # Step environment
            next_obs, rewards, done, _ = env.step({'A': aA, 'B': aB})
            obs_A = next_obs['A']
            obs_B = next_obs['B']
            
            # Record
            history.append((aA, aB, env.current_temptation))
            rewards_A.append(rewards['A'])
            rewards_B.append(rewards['B'])
            
            if done:
                break
        
        # Compute statistics
        coop_rate = compute_cooperation_rate(history)
        summary = episode_summary(rewards_A, rewards_B)
        
        stats.append({
            'coop_rate': coop_rate,
            **summary
        })
    
    # Aggregate
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
        'mean_B': meanB,
        'episodes': episodes
    }


def plot_episode(result: dict, title: str = None):
    """Plot the results of a single episode"""
    history = result['history']
    rewards_A = result['rewards_A']
    rewards_B = result['rewards_B']
    
    if not history:
        print("No history to plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data
    actions_A = [h[0] for h in history]
    actions_B = [h[1] for h in history]
    temptations = [h[2] for h in history]
    steps = list(range(len(history)))
    
    # Plot actions
    ax1.plot(steps, actions_A, 'bo-', label='Agent A', markersize=4)
    ax1.plot(steps, actions_B, 'ro-', label='Agent B', markersize=4)
    ax1.set_ylabel('Action (0=Coop, 1=Defect)')
    ax1.set_title('Actions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot temptation values
    ax2.plot(steps, temptations, 'g-', linewidth=2)
    ax2.set_ylabel('Temptation Value')
    ax2.set_title('Temptation Values Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    cumulative_A = np.cumsum(rewards_A)
    cumulative_B = np.cumsum(rewards_B)
    ax3.plot(steps, cumulative_A, 'b-', label='Agent A', linewidth=2)
    ax3.plot(steps, cumulative_B, 'r-', label='Agent B', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Rewards Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_strategy_analysis(results: dict, title: str = None):
    """Plot results of strategy analysis"""
    strategies = list(results.keys())
    cooperation_rates = [results[s]['cooperation_rate'] for s in strategies]
    mean_rewards = [results[s]['mean_A'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cooperation rates
    bars1 = ax1.bar(strategies, cooperation_rates, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation Rate vs Different Opponents')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, cooperation_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Mean rewards
    bars2 = ax2.bar(strategies, mean_rewards, alpha=0.7, color='lightcoral')
    ax2.set_ylabel('Mean Reward per Step')
    ax2.set_title('Mean Reward vs Different Opponents')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars2, mean_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{reward:.2f}', ha='center', va='bottom')
    
    # Rotate x-axis labels if needed
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def print_strategy_summary(results: dict):
    """Print a summary table of strategy analysis results"""
    print("\n" + "="*80)
    print("STRATEGY ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Opponent':<15} {'Coop Rate':<12} {'Mean Reward':<12} {'Total Reward':<12} {'Episodes':<10}")
    print("-"*80)
    
    for strategy, result in results.items():
        print(f"{strategy:<15} {result['cooperation_rate']:<12.3f} "
              f"{result['mean_A']:<12.2f} {result['sum_A']:<12.1f} {result['episodes']:<10}")
    
    print("="*80)
    
    # Find best performance
    best_coop = max(results.items(), key=lambda x: x[1]['cooperation_rate'])
    best_reward = max(results.items(), key=lambda x: x[1]['mean_A'])
    
    print(f"\nHighest cooperation rate: {best_coop[0]} ({best_coop[1]['cooperation_rate']:.3f})")
    print(f"Highest mean reward: {best_reward[0]} ({best_reward[1]['mean_A']:.2f})")
    print("="*80)
