import torch
import numpy as np
from collections import defaultdict


class RolloutBuffer:
    """Unified rollout buffer that handles both single and multi-env rollouts"""
    
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        """Reset buffer for new rollout collection"""
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []
        self.h0 = None  # initial hidden state for sequence
        self.last_value = None  # bootstrap value

    def add_step(self, obs, action, logprob, value, reward, done, mask):
        """Add a single timestep for all environments
        
        Args:
            obs: (B, obs_dim) observations
            action: (B,) actions  
            logprob: (B,) log probabilities
            value: (B,) value estimates
            reward: (B,) rewards
            done: (B,) done flags
            mask: (B,) masks (1.0 if not done, 0.0 if done)
        """
        # Convert all inputs to tensors if they aren't already
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.long)
        if not torch.is_tensor(logprob):
            logprob = torch.as_tensor(logprob, dtype=torch.float32)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        if not torch.is_tensor(reward):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        if not torch.is_tensor(done):
            done = torch.as_tensor(done, dtype=torch.float32)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.float32)
        
        # Ensure proper shapes
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if logprob.dim() == 0:
            logprob = logprob.unsqueeze(0)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)
        if mask.dim() == 0:
            mask = mask.unsqueeze(0)
            
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def add(self, **kwargs):
        """Alternative interface for adding data - converts to add_step format"""
        self.add_step(
            obs=kwargs['obs'],
            action=kwargs['actions'],
            logprob=kwargs['logprobs'], 
            value=kwargs['values'],
            reward=kwargs['rewards'],
            done=kwargs['dones'],
            mask=kwargs['masks']
        )

    def set_initial_hidden(self, h0):
        """Set initial hidden state for RNN"""
        self.h0 = h0

    def set_last_value(self, last_value):
        """Set bootstrap values for GAE computation"""
        if not torch.is_tensor(last_value):
            last_value = torch.as_tensor(last_value, dtype=torch.float32)
        if last_value.dim() == 0:
            last_value = last_value.unsqueeze(0)
        self.last_value = last_value

    def compute_gae(self):
        """Compute Generalized Advantage Estimation"""
        if len(self.obs) == 0:
            raise ValueError("Buffer is empty, cannot compute GAE")
            
        # Convert lists to tensors: shape (T, B, ...)
        obs = torch.stack(self.obs)           # (T, B, obs_dim)
        actions = torch.stack(self.actions)   # (T, B)
        logprobs = torch.stack(self.logprobs) # (T, B)
        values = torch.stack(self.values)     # (T, B)
        rewards = torch.stack(self.rewards)   # (T, B)
        dones = torch.stack(self.dones)       # (T, B)
        masks = torch.stack(self.masks)       # (T, B) 1.0 if not done

        T, B = rewards.shape
        device = rewards.device

        advantages = torch.zeros_like(rewards, device=device)
        last_adv = torch.zeros(B, device=device)
        next_value = self.last_value if self.last_value is not None else torch.zeros(B, device=device)

        for t in reversed(range(T)):
            mask = masks[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv
            next_value = values[t]
        
        returns = advantages + values

        # Normalize advantages across all timesteps and batch
        flat_adv = advantages.reshape(-1)
        if flat_adv.numel() > 1:
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std(unbiased=False) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        self.batch = {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            "h0": self.h0,
            "last_values": self.last_value,
        }
        return self.batch

    def get_batch(self, device='cpu'):
        """Get the computed batch, converting to specified device"""
        if not hasattr(self, 'batch') or self.batch is None:
            self.compute_gae()
        
        # Move batch to device
        batch = {}
        for k, v in self.batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                batch[k] = tuple(x.to(device) for x in v)
            else:
                batch[k] = v
        return batch

    def get_minibatches(self, minibatch_size):
        """
        Yield minibatches for PPO update.
        Assumes compute_gae() has been called.
        Keeps sequence length intact: batch is over environments (B); T is full rollout length.
        """
        if not hasattr(self, 'batch') or self.batch is None:
            raise ValueError("Must call compute_gae() before getting minibatches")
            
        b = self.batch
        T, B = b["obs"].shape[0], b["obs"].shape[1]
        indices = torch.randperm(B)
        
        for start in range(0, B, minibatch_size):
            end = min(start + minibatch_size, B)
            idx = indices[start:end]
            
            mb = {
                "obs": b["obs"][:, idx],
                "actions": b["actions"][:, idx],
                "logprobs": b["logprobs"][:, idx],
                "values": b["values"][:, idx],
                "rewards": b["rewards"][:, idx],
                "dones": b["dones"][:, idx],
                "masks": b["masks"][:, idx],
                "advantages": b["advantages"][:, idx],
                "returns": b["returns"][:, idx],
                "h0": None,
                "last_values": b["last_values"][idx] if b["last_values"] is not None else None,
            }
            
            if b["h0"] is not None:
                h0 = b["h0"]
                if isinstance(h0, tuple):
                    mb["h0"] = (h0[0][:, idx], h0[1][:, idx])
                else:
                    mb["h0"] = h0[:, idx]
                    
            yield mb
