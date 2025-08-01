import torch
import torch.nn as nn
from torch.distributions import Categorical

class RNNPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, rnn_type="lstm", num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=False)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=False)
        self.pi = nn.Linear(hidden_size, 2)  # Always 2 actions for Prisoner's Dilemma
        self.v = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size, device=None):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "gru":
            return h
        c = torch.zeros_like(h)
        return (h, c)

    def _apply_mask(self, h, mask):
        if mask is None:
            return h
        m = mask.view(1, -1, 1)
        if self.rnn_type == "gru":
            return h * m
        h0, c0 = h
        return (h0 * m, c0 * m)

    def forward(self, obs, h0=None, masks=None):
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(0)
        T, B = obs.shape[0], obs.shape[1]
        x = self.enc(obs.reshape(T * B, -1)).reshape(T, B, -1)
        if h0 is None:
            h0 = self.initial_state(B, device=obs.device)
        if masks is None:
            out, hN = self.rnn(x, h0)
        else:
            hs = []
            h = h0
            for t in range(T):
                h = self._apply_mask(h, masks[t])
                out_t, h = self.rnn(x[t:t+1], h)
                hs.append(out_t)
            out = torch.cat(hs, dim=0)
            hN = h
        logits = self.pi(out)
        values = self.v(out).squeeze(-1)
        if single_step:
            logits = logits.squeeze(0)
            values = values.squeeze(0)
        return logits, values, hN

    def evaluate_actions(self, obs, actions, h0=None, masks=None):
        logits, values, _ = self.forward(obs, h0=h0, masks=masks)
        if logits.dim() == 3:
            T, B = logits.shape[:2]
            dist = Categorical(logits=logits.reshape(T * B, -1))
            logp = dist.log_prob(actions.reshape(T * B)).reshape(T, B)
            ent = dist.entropy().reshape(T, B)
        else:
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ent = dist.entropy()
        return logp, ent, values

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, rnn_type="lstm", num_layers=1, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.net = RNNPolicyNetwork(obs_dim, hidden_size, rnn_type, num_layers)
        self.hidden_state = None
        self.to(self.device)

    def parameters(self, recurse: bool = True):
        return self.net.parameters(recurse=recurse)
    
    def reset_hidden(self, batch_size=1):
        """Reset hidden state for new episodes"""
        self.hidden_state = self.net.initial_state(batch_size, device=self.device)
    
    def initial_state(self, batch_size, device=None):
        """Get initial hidden state"""
        return self.net.initial_state(batch_size, device=device or self.device)

    @torch.no_grad()
    def act(self, obs, h0=None, deterministic=False):
        """
        Act given observation. Handles both dict and tensor inputs.
        Returns dict with action, log_prob, value, entropy for compatibility.
        """
        # Handle dict observations from environment
        if isinstance(obs, dict):
            # Assume we want one of the agent observations
            obs = obs.get('A') or obs.get('B') or list(obs.values())[0]
        
        # Convert to tensor if needed
        if isinstance(obs, (list, tuple, np.ndarray)):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # Ensure proper batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Use provided hidden state or stored one
        if h0 is None:
            h0 = self.hidden_state
        
        logits, values, hN = self.net(obs, h0=h0, masks=None)
        self.hidden_state = hN  # Update stored hidden state
        
        dist = Categorical(logits=logits)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return {
            'action': actions,
            'log_prob': logp,
            'value': values.squeeze(-1),
            'entropy': entropy
        }

    @torch.no_grad()
    def get_value(self, obs, h0=None):
        """Get value estimate for given observation"""
        # Handle dict observations from environment
        if isinstance(obs, dict):
            obs = obs.get('A') or obs.get('B') or list(obs.values())[0]
        
        # Convert to tensor if needed
        if isinstance(obs, (list, tuple, np.ndarray)):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        if h0 is None:
            h0 = self.hidden_state
            
        _, values, _ = self.net(obs, h0=h0, masks=None)
        return values.squeeze(-1)

    def evaluate_actions(self, obs, actions, h0=None, masks=None):
        return self.net.evaluate_actions(obs, actions, h0=h0, masks=masks)

# Import numpy for compatibility
import numpy as np
