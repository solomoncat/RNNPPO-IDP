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
from models import PPOAgent
from environment import PrisonersDilemmaEnv
from ppo import PPO

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


class RolloutBuffer:
    """Buffer for storing self-play rollouts with RNN support"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = defaultdict(list)
        self.episode_starts = []
    
    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)
    
    def get_batch(self, device='cpu') -> Dict[str, torch.Tensor]:
        """Convert lists to tensors and prepare batch for PPO update"""
        batch = {}
        
        # Stack temporal data
        for k in ['obs', 'actions', 'logprobs', 'values', 'rewards', 'dones', 'masks']:
            if k in self.data:
                batch[k] = torch.stack(self.data[k], dim=0).to(device)
        
        # Handle last values (no temporal dimension)
        if 'last_values' in self.data:
            batch['last_values'] = self.data['last_values'][0].to(device)
        
        # Handle initial hidden states
        if 'h0' in self.data and self.data['h0']:
            h0 = self.data['h0'][0]
            if isinstance(h0, tuple):
                batch['h0'] = tuple(h.to(device) for h in h0)
            else:
                batch['h0'] = h0.to(device)
        
        return batch


class SelfPlayTrainer:
    """Main trainer for self-play PPO"""
    
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
        
        # Policy networks (shared for self-play or separate)
        obs_dim = self.env.observation_space.shape[0]
        if config.shared_policy:
            self.agent = PPOAgent(
                obs_dim=obs_dim,
                hidden_size=config.hidden_size,
                rnn_type=config.rnn_type,
                num_layers=config.num_layers
            ).to(self.device)
            self.agent_A = self.agent_B = self.agent
        else:
            self.agent_A = PPOAgent(
                obs_dim=obs_dim,
                hidden_size=config.hidden_size,
                rnn_type=config.rnn_type,
                num_layers=config.num_layers
            ).to(self.device)
            self.agent_B = PPOAgent(
                obs_dim=obs_dim,
                hidden_size=config.hidden_size,
                rnn_type=config.rnn_type,
                num_layers=config.num_layers
            ).to(self.device)
        
        # PPO trainers
        self.ppo_A = PPO(
            self.agent_A,
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
        
        if not config.shared_policy:
            self.ppo_B = PPO(
                self.agent_B,
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
    
    def collect_rollouts(self, num_envs: int, num_steps: int) -> Tuple[RolloutBuffer, RolloutBuffer]:
        """Collect self-play rollouts for both agents"""
        buffer_A = RolloutBuffer()
        buffer_B = RolloutBuffer()
        
        # Initialize environments and hidden states
        envs = [PrisonersDilemmaEnv(
            max_steps=self.config.env_max_steps,
            history_length=self.config.history_length,
            temptation_range=(self.config.temptation_low, self.config.temptation_high),
            seed=self.config.seed + i if self.config.seed else None
        ) for i in range(num_envs)]
        
        obs_dict = [env.reset() for env in envs]
        obs_A = torch.stack([torch.tensor(o['A'], dtype=torch.float32) for o in obs_dict]).to(self.device)
        obs_B = torch.stack([torch.tensor(o['B'], dtype=torch.float32) for o in obs_dict]).to(self.device)
        
        h_A = self.agent_A.initial_state(num_envs, device=self.device)
        h_B = self.agent_B.initial_state(num_envs, device=self.device)
        
        # Store initial hidden states
        buffer_A.data['h0'].append(h_A)
        buffer_B.data['h0'].append(h_B)
        
        episode_rewards_A = np.zeros(num_envs)
        episode_rewards_B = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=int)
        masks = torch.ones(num_envs, device=self.device)
        
        # Collect rollout
        for step in range(num_steps):
            # Get actions from both agents
            with torch.no_grad():
                actions_A, logp_A, values_A, h_A = self.agent_A.act(obs_A, h0=h_A)
                actions_B, logp_B, values_B, h_B = self.agent_B.act(obs_B, h0=h_B)
            
            # Execute actions in environments
            rewards_A = torch.zeros(num_envs, device=self.device)
            rewards_B = torch.zeros(num_envs, device=self.device)
            dones = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            
            next_obs_A = []
            next_obs_B = []
            
            for i, env in enumerate(envs):
                actions = (actions_A[i].item(), actions_B[i].item())
                obs_dict, reward_dict, done, _ = env.step(actions)
                
                rewards_A[i] = reward_dict['A']
                rewards_B[i] = reward_dict['B']
                dones[i] = done
                
                if done:
                    # Log episode stats
                    self.log_episode_stats({
                        'episode_reward_A': episode_rewards_A[i] + rewards_A[i].item(),
                        'episode_reward_B': episode_rewards_B[i] + rewards_B[i].item(),
                        'episode_length': episode_lengths[i] + 1,
                        'cooperation_rate': self.calculate_cooperation_rate(env)
                    })
                    
                    # Reset environment
                    obs_dict = env.reset()
                    episode_rewards_A[i] = 0
                    episode_rewards_B[i] = 0
                    episode_lengths[i] = 0
                    self.episode_count += 1
                else:
                    episode_rewards_A[i] += rewards_A[i].item()
                    episode_rewards_B[i] += rewards_B[i].item()
                    episode_lengths[i] += 1
                
                next_obs_A.append(torch.tensor(obs_dict['A'], dtype=torch.float32))
                next_obs_B.append(torch.tensor(obs_dict['B'], dtype=torch.float32))
            
            next_obs_A = torch.stack(next_obs_A).to(self.device)
            next_obs_B = torch.stack(next_obs_B).to(self.device)
            
            # Store transitions
            buffer_A.add(
                obs=obs_A,
                actions=actions_A,
                logprobs=logp_A,
                values=values_A,
                rewards=rewards_A,
                dones=dones,
                masks=masks.clone()
            )
            
            buffer_B.add(
                obs=obs_B,
                actions=actions_B,
                logprobs=logp_B,
                values=values_B,
                rewards=rewards_B,
                dones=dones,
                masks=masks.clone()
            )
            
            # Update observations and masks
            obs_A = next_obs_A
            obs_B = next_obs_B
            masks = (~dones).float()
            
            # Reset hidden states for done episodes
            if self.config.rnn_type == "lstm":
                h_A = (h_A[0] * masks.unsqueeze(0).unsqueeze(-1),
                       h_A[1] * masks.unsqueeze(0).unsqueeze(-1))
                h_B = (h_B[0] * masks.unsqueeze(0).unsqueeze(-1),
                       h_B[1] * masks.unsqueeze(0).unsqueeze(-1))
            else:
                h_A = h_A * masks.unsqueeze(0).unsqueeze(-1)
                h_B = h_B * masks.unsqueeze(0).unsqueeze(-1)
            
            self.global_step += num_envs
        
        # Compute last values for GAE
        with torch.no_grad():
            _, _, last_values_A, _ = self.agent_A.act(obs_A, h0=h_A)
            _, _, last_values_B, _ = self.agent_B.act(obs_B, h0=h_B)
        
        buffer_A.data['last_values'].append(last_values_A)
        buffer_B.data['last_values'].append(last_values_B)
        
        return buffer_A, buffer_B
    
    def calculate_cooperation_rate(self, env) -> float:
        """Calculate cooperation rate from environment history"""
        if not env.history:
            return 0.0
        cooperations = sum(1 for a_A, a_B, _ in env.history if a_A == 0 and a_B == 0)
        return cooperations / len(env.history)
    
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
    
    def log_training_stats(self, stats_A: dict, stats_B: dict):
        """Log training statistics"""
        stats = {
            **{f'agent_A/{k}': v for k, v in stats_A.items()},
            **{f'agent_B/{k}': v for k, v in stats_B.items()},
            'global_step': self.global_step
        }
        
        if WANDB_AVAILABLE and self.config.use_wandb:
            wandb.log(stats)
        
        if TENSORBOARD_AVAILABLE and self.config.use_tensorboard:
            for key, value in stats.items():
                if key != 'global_step':
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'config': vars(self.config)
        }
        
        if self.config.shared_policy:
            checkpoint['agent_state_dict'] = self.agent.state_dict()
            checkpoint['optimizer_state_dict'] = self.ppo_A.optimizer.state_dict()
        else:
            checkpoint['agent_A_state_dict'] = self.agent_A.state_dict()
            checkpoint['agent_B_state_dict'] = self.agent_B.state_dict()
            checkpoint['optimizer_A_state_dict'] = self.ppo_A.optimizer.state_dict()
            checkpoint['optimizer_B_state_dict'] = self.ppo_B.optimizer.state_dict()
        
        path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pt')
        torch.save(checkpoint, path)
        
        # Save best/latest links
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(path), latest_path)
        
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.config.shared_policy:
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.ppo_A.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.agent_A.load_state_dict(checkpoint['agent_A_state_dict'])
            self.agent_B.load_state_dict(checkpoint['agent_B_state_dict'])
            self.ppo_A.optimizer.load_state_dict(checkpoint['optimizer_A_state_dict'])
            self.ppo_B.optimizer.load_state_dict(checkpoint['optimizer_B_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        
        print(f"Loaded checkpoint from {path}")
        return checkpoint['iteration']
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Config: {vars(self.config)}")
        
        start_time = time.time()
        
        for iteration in range(self.config.num_iterations):
            iter_start_time = time.time()
            
            # Collect rollouts
            buffer_A, buffer_B = self.collect_rollouts(
                num_envs=self.config.num_envs,
                num_steps=self.config.rollout_length
            )
            
            # Get batches
            batch_A = buffer_A.get_batch(self.device)
            batch_B = buffer_B.get_batch(self.device)
            
            # PPO updates
            stats_A = self.ppo_A.update(batch_A)
            if self.config.shared_policy:
                stats_B = stats_A  # Same stats for shared policy
            else:
                stats_B = self.ppo_B.update(batch_B)
            
            # Log stats
            self.log_training_stats(stats_A, stats_B)
            
            # Print progress
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                iter_time = time.time() - iter_start_time
                fps = (self.config.num_envs * self.config.rollout_length) / iter_time
                
                print(f"\nIteration {iteration}/{self.config.num_iterations}")
                print(f"  Global Step: {self.global_step}")
                print(f"  Episodes: {self.episode_count}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Elapsed: {elapsed/60:.1f} min")
                print(f"  Agent A - Loss: {stats_A['loss_pi']:.4f} (pi) {stats_A['loss_v']:.4f} (v)")
                print(f"  Agent A - KL: {stats_A['kl']:.4f}, Entropy: {stats_A['entropy']:.4f}")
                if not self.config.shared_policy:
                    print(f"  Agent B - Loss: {stats_B['loss_pi']:.4f} (pi) {stats_B['loss_v']:.4f} (v)")
                    print(f"  Agent B - KL: {stats_B['kl']:.4f}, Entropy: {stats_B['entropy']:.4f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        # Final checkpoint
        self.save_checkpoint(self.config.num_iterations)
        print("\nTraining completed!")


def get_args():
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
    parser.add_argument('--shared-policy', action='store_true', 
                        help='Use shared policy for both agents')
    
    # Training
    parser.add_argument('--num-iterations', type=int, default=1000)
    parser.add_argument('--num-envs', type=int, default=32)
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
    
    # Create trainer
    trainer = SelfPlayTrainer(args)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train
    trainer.train()
