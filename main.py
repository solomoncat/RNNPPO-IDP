import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from environment import PrisonersDilemmaEnv
from agent import PPOAgent
from ppo import PPO
from buffer import RolloutBuffer
from utils import SimpleLogger, save_model, load_model
from evaluate import strategy_analysis, AlwaysCooperate, AlwaysDefect, TitForTat, GrimTrigger, RandomPolicy

def load_config(path):
    """Load configuration from YAML file"""
    if path and Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def make_trainer(cfg):
    """Create training components"""
    device = torch.device(cfg.get("device", "cpu"))
    
    # Create environment to get observation dimension
    env = PrisonersDilemmaEnv(
        max_steps=cfg.get("env_max_steps", 100),
        history_length=cfg.get("history_length", 10),
        temptation_range=(cfg.get("temptation_low", 3.0), cfg.get("temptation_high", 10.0)),
        seed=cfg.get("seed", None)
    )
    sample_obs = env.reset()["A"]
    obs_dim = sample_obs.shape[0]

    # Create agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 1),
        rnn_type=cfg.get("rnn_type", "lstm"),
        device=device
    )
    
    # Create PPO trainer
    ppo = PPO(
        agent,  # Pass the full agent
        lr=cfg.get("lr", 3e-4),
        clip_epsilon=cfg.get("clip_epsilon", 0.2),
        epochs=cfg.get("ppo_epochs", 10),
        batch_size=cfg.get("batch_size", 64),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        value_coef=cfg.get("value_coef", 0.5),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        clip_value_loss=cfg.get("clip_value_loss", True),
        recurrent=True,
        device=device
    )
    
    return env, agent, ppo, device

def train(cfg):
    """Training loop"""
    env, agent, ppo, device = make_trainer(cfg)
    logger = SimpleLogger(cfg.get("log_dir", "runs"))
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    rollout_length = cfg.get("rollout_length", 128)
    num_iterations = cfg.get("num_iterations", 1000)
    shared = cfg.get("shared_policy", True)
    
    print(f"Starting training on {device}")
    print(f"Rollout length: {rollout_length}, Iterations: {num_iterations}")
    print(f"Shared policy: {shared}")
    
    for iteration in range(1, num_iterations + 1):
        # Create buffer
        buffer = RolloutBuffer(gamma=cfg.get("gamma", 0.99), lam=cfg.get("gae_lambda", 0.95))
        
        # Reset environment and agent
        obs_dict = env.reset()
        obs_A = torch.tensor(obs_dict["A"], dtype=torch.float32, device=device)
        obs_B = torch.tensor(obs_dict["B"], dtype=torch.float32, device=device)
        
        agent.reset_hidden(batch_size=1)
        
        # Store initial hidden state for agent B if doing self-play
        if shared:
            h0_A = agent.hidden_state
            # For self-play, we need separate hidden states
            h0_B = agent.net.initial_state(1, device=device)
            buffer.set_initial_hidden(h0_A)
        
        episode_rewards_A = 0.0
        episode_rewards_B = 0.0
        episode_length = 0
        
        # Collect rollout
        for t in range(rollout_length):
            # Agent A acts
            outA = agent.act(obs_A)
            aA = int(outA["action"])
            logpA = outA["log_prob"]
            vA = outA["value"]
            
            if shared:
                # Agent B acts (self-play with separate hidden state)
                current_hidden = agent.hidden_state
                agent.hidden_state = h0_B
                outB = agent.act(obs_B)
                h0_B = agent.hidden_state
                agent.hidden_state = current_hidden
            else:
                # Would need separate agent for Agent B
                outB = agent.act(obs_B)  # Simplified for now
                
            aB = int(outB["action"])
            logpB = outB["log_prob"]
            vB = outB["value"]

            # Step environment
            next_obs, rewards, done, _ = env.step({"A": aA, "B": aB})
            rA = rewards["A"]
            rB = rewards["B"]
            
            episode_rewards_A += rA
            episode_rewards_B += rB
            episode_length += 1

            # Store transitions for both agents (shared policy training)
            mask = 0.0 if done else 1.0
            
            # Add Agent A's experience
            buffer.add_step(
                obs=obs_A,
                action=torch.tensor(aA, dtype=torch.long),
                logprob=logpA,
                value=vA,
                reward=torch.tensor(rA, dtype=torch.float32),
                done=torch.tensor(done, dtype=torch.float32),
                mask=torch.tensor(mask, dtype=torch.float32)
            )
            
            # Add Agent B's experience (for shared policy)
            buffer.add_step(
                obs=obs_B,
                action=torch.tensor(aB, dtype=torch.long),
                logprob=logpB,
                value=vB,
                reward=torch.tensor(rB, dtype=torch.float32),
                done=torch.tensor(done, dtype=torch.float32),
                mask=torch.tensor(mask, dtype=torch.float32)
            )
            
            # Update observations
            obs_A = torch.tensor(next_obs["A"], dtype=torch.float32, device=device)
            obs_B = torch.tensor(next_obs["B"], dtype=torch.float32, device=device)
            
            if done:
                # Log episode stats
                logger.log({
                    "episode_reward_A": episode_rewards_A,
                    "episode_reward_B": episode_rewards_B,
                    "episode_length": episode_length,
                }, step=iteration)
                
                # Reset for next episode
                obs_dict = env.reset()
                obs_A = torch.tensor(obs_dict["A"], dtype=torch.float32, device=device)
                obs_B = torch.tensor(obs_dict["B"], dtype=torch.float32, device=device)
                agent.reset_hidden(batch_size=1)
                if shared:
                    h0_B = agent.net.initial_state(1, device=device)
                episode_rewards_A = 0.0
                episode_rewards_B = 0.0
                episode_length = 0
        
        # Bootstrap last value
        last_value = agent.get_value(obs_A)
        buffer.set_last_value(last_value)
        
        # Compute GAE and update
        batch = buffer.compute_gae()
        stats = ppo.update(batch)
        
        # Log training stats
        logger.log({
            "iteration": iteration,
            "loss_pi": stats.get("loss_pi", 0.0),
            "loss_v": stats.get("loss_v", 0.0),
            "entropy": stats.get("entropy", 0.0),
            "kl": stats.get("kl", 0.0),
            "clipfrac": stats.get("clipfrac", 0.0)
        }, step=iteration)
        
        # Print progress
        if iteration % cfg.get("log_interval", 100) == 0:
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"  Loss (pi/v): {stats.get('loss_pi', 0.0):.4f} / {stats.get('loss_v', 0.0):.4f}")
            print(f"  KL/Entropy: {stats.get('kl', 0.0):.4f} / {stats.get('entropy', 0.0):.4f}")
        
        # Save checkpoint
        if iteration % cfg.get("save_interval", 100) == 0:
            path = save_dir / f"checkpoint_{iteration}.pt"
            save_model(str(path), agent.net, optimizer=ppo.optimizer, 
                      extra={"iteration": iteration, "config": cfg})
            print(f"Saved checkpoint to {path}")

    logger.close()
    print("Training completed!")

def evaluate(cfg):
    """Evaluation against baseline strategies"""
    # Environment parameters
    env_kwargs = {
        "max_steps": cfg.get("env_max_steps", 100),
        "history_length": cfg.get("history_length", 10),
        "temptation_range": (cfg.get("temptation_low", 3.0), cfg.get("temptation_high", 10.0)),
        "seed": cfg.get("seed", None)
    }
    
    device = torch.device(cfg.get("device", "cpu"))
    
    # Create environment to get observation dimension
    env = PrisonersDilemmaEnv(**env_kwargs)
    sample_obs = env.reset()["A"]
    obs_dim = sample_obs.shape[0]
    
    # Create agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 1),
        rnn_type=cfg.get("rnn_type", "lstm"),
        device=device
    )
    
    # Load checkpoint if specified
    if cfg.get("load_checkpoint"):
        checkpoint_path = cfg["load_checkpoint"]
        print(f"Loading checkpoint from {checkpoint_path}")
        extra = load_model(checkpoint_path, agent.net)
        print(f"Loaded checkpoint from iteration {extra.get('iteration', 'unknown')}")
    else:
        print("No checkpoint specified, using randomly initialized agent")
    
    # Define baseline strategies
    baselines = {
        "AlwaysCooperate": AlwaysCooperate,
        "AlwaysDefect": AlwaysDefect,
        "TitForTat": TitForTat,
        "GrimTrigger": GrimTrigger,
        "Random50": lambda: RandomPolicy(0.5),
        "Random25": lambda: RandomPolicy(0.25),
        "Random75": lambda: RandomPolicy(0.75),
    }
    
    # Run evaluation
    results = strategy_analysis(
        agent, 
        env_kwargs, 
        baselines, 
        episodes=cfg.get("eval_episodes", 100),
        max_steps=cfg.get("env_max_steps", 100)
    )
    
    # Print results
    from evaluate import print_strategy_summary
    print_strategy_summary(results)
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PPO Prisoner's Dilemma Training")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to load')
    
    # Training parameters
    parser.add_argument('--num-iterations', type=int, default=1000)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    
    # Environment parameters
    parser.add_argument('--env-max-steps', type=int, default=100)
    parser.add_argument('--history-length', type=int, default=10)
    parser.add_argument('--temptation-low', type=float, default=3.0)
    parser.add_argument('--temptation-high', type=float, default=10.0)
    
    # Other
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--shared-policy', action='store_true', default=True)
    parser.add_argument('--eval-episodes', type=int, default=100)
    
    args = parser.parse_args()
    
    # Load config file if provided
    cfg = load_config(args.config) if args.config else {}
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            cfg[key.replace('_', '_')] = value  # Keep underscores for consistency
    
    # Auto-detect device
    if cfg.get('device') == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    
    # Set random seeds for reproducibility
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        print(f"Set random seed to {cfg['seed']}")
    
    print(f"Using device: {cfg.get('device', 'cpu')}")
    
    if cfg.get("mode") == "train":
        train(cfg)
    else:
        evaluate(cfg)
