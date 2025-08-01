"""
Simplified training script that uses the main.py functionality.
This version removes redundancy and focuses on the core training loop with better logging.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time
import os
import json
from datetime import datetime
import argparse
from typing import Dict, Tuple, Optional

# Import your modules
from agent import PPOAgent
from environment import PrisonersDilemmaEnv
from ppo import PPO
from buffer import RolloutBuffer

# Optional: Import logging libraries if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, using console logging only")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available")


class SelfPlayTrainer:
    """Improved self-play trainer with better environment handling"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Environment
        self.env = PrisonersDilemmaEnv(
            max_steps=config.env_max_steps,
            history_length=config.history_length,
            temptation_range=(config.temptation_low, config.temptation_high),
            seed=config.seed
        )
        
        # Get observation dimension
        sample_obs = self.env.reset()['A']
        obs_dim = sample_obs.shape[0]
        
        # Policy network (shared for self-play)
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            hidden_size=config.hidden_size,
            rnn_type=config.rnn_type,
            num_layers=config.num_layers,
            device=self.device
        )
        
        # PPO trainer
        self.ppo = PPO(
            self.agent,
            lr=config.lr,
            clip_epsilon=config.clip_epsilon,
            epochs=config.ppo_epochs,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            clip_value_loss=config.clip_value_loss,
            recurrent=True,
            device=self.device
        )
        
        # Logging
        self.setup_logging()
        
        # Stats
        self.global_step = 0
        self.episode_count = 0
    
    def setup_logging(self):
        """Setup logging (wandb, tensorboard, file logging)"""
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.config.log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        # Setup wandb
        if WANDB_AVAILABLE and self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"ppo_selfplay_{timestamp}",
                config=vars(self.config)
            )
        
        # Setup tensorboard
        if TENSORBOARD_AVAILABLE and self.config.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))
    
    def collect_rollout(self, num_steps: int) -> RolloutBuffer:
        """Collect a single environment rollout for self-play"""
        buffer = RolloutBuffer(gamma=self.config.gamma, lam=self.config.gae_lambda)
        
        # Reset environment and agent
        obs_dict = self.env.reset()
        obs_A = torch.tensor(obs_dict['A'], dtype=torch.float32, device=self.device)
        obs_B = torch.tensor(obs_dict['B'], dtype=torch.float32, device=self.device)
        
        # Reset hidden states for both agents
        self.agent.reset_hidden(batch_size=1)
        h0_A = self.agent.hidden_state
        h0_B = self.agent.net.initial_state(1, device=self.device)
        
        buffer.set_initial_hidden(h0_A)
        
        episode_reward_A = 0.0
        episode_reward_B = 0.0
        episode_length = 0
        cooperation_count = 0
        
        for step in range(num_steps):
            # Agent A acts
            self.agent.hidden_state = h0_A
            out_A = self.agent.act(obs_A)
            h0_A = self.agent.hidden_state
            action_A = int(out_A['action'])
            
            # Agent B acts (same policy, different hidden state)
            self.agent.hidden_state = h0_B
            out_B = self.agent.act(obs_B)
            h0_B = self.agent.hidden_state
            action_B = int(out_B['action'])
            
            # Step environment
            next_obs_dict, rewards, done, _ = self.env.step({
                'A': action_A, 
                'B': action_B
            })
            
            reward_A = rewards['A']
            reward_B = rewards['B']
            episode_reward_A += reward_A
            episode_reward_B += reward_B
            episode_length += 1
            
            # Track cooperation
            if action_A == 0 and action_B == 0:
                cooperation_count += 1
            
            # Store transitions for both agents
            mask = 0.0 if done else 1.0
            
            # Agent A's experience
            buffer.add_step(
                obs=obs_A.unsqueeze(0),  # Add batch dimension
                action=torch.tensor([action_A], dtype=torch.long),
                logprob=out_A['log_prob'].unsqueeze(0),
                value=out_A['value'].unsqueeze(0),
                reward=torch.tensor([reward_A], dtype=torch.float32),
                done=torch.tensor([done], dtype=torch.float32),
                mask=torch.tensor([mask], dtype=torch.float32)
            )
            
            # Agent B's experience
            buffer.add_step(
                obs=obs_B.unsqueeze(0),  # Add batch dimension
                action=torch.tensor([action_B], dtype=torch.long),
                logprob=out_B['log_prob'].unsqueeze(0),
                value=out_B['value'].unsqueeze(0),
                reward=torch.tensor([reward_B], dtype=torch.float32),
                done=torch.tensor([done], dtype=torch.float32),
                mask=torch.tensor([mask], dtype=torch.float32)
            )
            
            # Update observations
            obs_A = torch.tensor(next_obs_dict['A'], dtype=torch.float32, device=self.device)
            obs_B = torch.tensor(next_obs_dict['B'], dtype=torch.float32, device=self.device)
            
            self.global_step += 1
            
            if done:
                # Log episode stats
                cooperation_rate = cooperation_count / episode_length if episode_length > 0 else 0.0
                self.log_episode_stats({
                    'episode_reward_A': episode_reward_A,
                    'episode_reward_B': episode_reward_B,
                    'episode_length': episode_length,
                    'cooperation_rate': cooperation_rate,
                    'mutual_cooperation_count': cooperation_count
                })
                
                # Reset for next episode within the rollout
                obs_dict = self.env.reset()
                obs_A = torch.tensor(obs_dict['A'], dtype=torch.float32, device=self.device)
                obs_B = torch.tensor(obs_dict['B'], dtype=torch.float32, device=self.device)
                
                # Reset hidden states
                self.agent.reset_hidden(batch_size=1)
                h0_A = self.agent.hidden_state
                h0_B = self.agent.net.initial_state(1, device=self.device)
                
                episode_reward_A = 0.0
                episode_reward_B = 0.0
                episode_length = 0
                cooperation_count = 0
                self.episode_count += 1
        
        # Bootstrap last values
        self.agent.hidden_state = h0_A
        last_value_A = self.agent.get_value(obs_A)
        buffer.set_last_value(last_value_A.unsqueeze(0))
        
        return buffer
    
    def log_episode_stats(self, stats: dict):
        """Log episode statistics"""
        if WANDB_AVAILABLE and self.config.use_wandb:
            wandb.log({
                **stats,
                'episode': self.episode_count,
                'global_step': self.global_step
            })
        
        if TENSORBOARD_AVAILABLE and self.config.use_tensorboard:
            for key, value in stats.items():
                self.writer.add_scalar(f'episode/{key}', value, self.episode_count)
    
    def log_training_stats(self, stats: dict):
        """Log training statistics"""
        log_stats = {**stats, 'global_step': self.global_step}
        
        if WANDB_AVAILABLE and self.config.use_wandb:
            wandb.log(log_stats)
        
        if TENSORBOARD_AVAILABLE and self.config.use_tensorboard:
            for key, value in stats.items():
                self.writer.add_scalar(f'train/{key}', value, self.global_step)
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'config': vars(self.config)
        }
        
        path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pt')
        torch.save(checkpoint, path)
        
        # Save latest link
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(path), latest_path)
        
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        
        print(f"Loaded checkpoint from {path}")
        return checkpoint['iteration']
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Config: {vars(self.config)}")
        
        start_time = time.time()
        
        for iteration in range(1, self.config.num_iterations + 1):
            iter_start_time = time.time()
            
            # Collect rollout
            buffer = self.collect_rollout(self.config.rollout_length)
            
            # Get batch for training
            batch = buffer.get_batch(self.device)
            
            # PPO update
            stats = self.ppo.update(batch)
            
            # Log stats
            self.log_training_stats(stats)
            
            # Print progress
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                iter_time = time.time() - iter_start_time
                fps = self.config.rollout_length / iter_time
                
                print(f"\nIteration {iteration}/{self.config.num_iterations}")
                print(f"  Global Step: {self.global_step}")
                print(f"  Episodes: {self.episode_count}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Elapsed: {elapsed/60:.1f} min")
                print(f"  Loss (pi/v): {stats['loss_pi']:.4f} / {stats['loss_v']:.4f}")
                print(f"  KL/Entropy: {stats['kl']:.4f} / {stats['entropy']:.4f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        # Final checkpoint
        self.save_checkpoint(self.config.num_iterations)
        print("\nTraining completed!")


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PPO Self-Play Training')
    
    # Environment
    parser.add_argument('--env-max-steps', type=int, default=100)
    parser.add_argument('--history-length', type=int, default=10)
    parser.add_argument('--temptation-low', type=float, default=3.0)
    parser.add_argument('--temptation-high', type=float, default=10.0)
    
    # Model
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--rnn-type', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--num-layers', type=int, default=1)
    
    # Training
    parser.add_argument('--num-iterations', type=int, default=1000)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--ppo-epochs', type=int, default=10)
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--clip-value-loss', action='store_true')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='ppo-selfplay')
    parser.add_argument('--use-tensorboard', action='store_true')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--load-checkpoint', type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed}")
    
    # Create trainer
    trainer = SelfPlayTrainer(args)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train
    trainer.train()
