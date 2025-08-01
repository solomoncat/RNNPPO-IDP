import torch
import torch.nn as nn
import torch.optim as optim

def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    T, B = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    next_value = last_values
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns

class PPO:
    def __init__(
        self,
        policy_or_agent,  # Can accept either policy network or full agent
        lr=3e-4,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        clip_value_loss=True,
        recurrent=True,
        device=None,
    ):
        # Handle both agent and policy network inputs
        if hasattr(policy_or_agent, 'net'):
            # It's an agent with .net attribute
            self.policy = policy_or_agent.net
            self.agent = policy_or_agent
        elif hasattr(policy_or_agent, 'evaluate_actions'):
            # It's a policy network directly
            self.policy = policy_or_agent
            self.agent = None
        else:
            # Assume it's a policy network
            self.policy = policy_or_agent
            self.agent = None
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.recurrent = recurrent
        self.device = device or next(self.policy.parameters()).device

    @torch.no_grad()
    def _norm_adv(self, adv):
        """Normalize advantages"""
        flat_adv = adv.reshape(-1)
        if flat_adv.numel() <= 1:
            return adv
        m = flat_adv.mean()
        s = flat_adv.std(unbiased=False)
        return (adv - m) / (s + 1e-8)

    def update(self, batch):
        """Update policy using PPO algorithm"""
        # Move batch to device
        obs = batch["obs"].to(self.device)                # (T,B,...) tensor
        actions = batch["actions"].to(self.device)        # (T,B,...) tensor or long
        old_logp = batch["logprobs"].to(self.device)      # (T,B)
        values = batch["values"].to(self.device)          # (T,B)
        rewards = batch["rewards"].to(self.device)        # (T,B)
        dones = batch["dones"].to(self.device)            # (T,B)
        last_values = batch["last_values"]
        if last_values is not None:
            last_values = last_values.to(self.device)    # (B,)
        else:
            last_values = torch.zeros(rewards.shape[1], device=self.device)
            
        masks = batch.get("masks", None)
        if masks is not None:
            masks = masks.to(self.device)
        h0 = batch.get("h0", None)
        if isinstance(h0, tuple):
            h0 = tuple(x.to(self.device) for x in h0)
        elif h0 is not None:
            h0 = h0.to(self.device)

        # Compute advantages if not already done
        if "advantages" in batch and "returns" in batch:
            adv = batch["advantages"].to(self.device)
            returns = batch["returns"].to(self.device)
        else:
            adv, returns = compute_gae(rewards, values, dones, last_values, self.gamma, self.gae_lambda)
        
        # Normalize advantages
        adv = self._norm_adv(adv)

        T, B = rewards.shape
        metrics = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
        num_updates = 0

        if self.recurrent:
            # For recurrent policies, batch over environments but keep time dimension
            env_indices = torch.arange(B, device=self.device)
            mb_size = max(1, min(self.batch_size, B))
            
            for epoch in range(self.epochs):
                perm = env_indices[torch.randperm(B)]
                for start in range(0, B, mb_size):
                    end = min(start + mb_size, B)
                    idx = perm[start:end]
                    
                    # Extract minibatch
                    o = obs[:, idx]
                    a = actions[:, idx]
                    lp_old = old_logp[:, idx]
                    v_old = values[:, idx]
                    adv_mb = adv[:, idx]
                    ret_mb = returns[:, idx]
                    
                    h0_mb = None
                    if h0 is not None:
                        if isinstance(h0, tuple):
                            h0_mb = (h0[0][:, idx], h0[1][:, idx])
                        else:
                            h0_mb = h0[:, idx]
                    
                    m_mb = masks[:, idx] if masks is not None else None

                    # Forward pass
                    new_logp, entropy, v = self.policy.evaluate_actions(o, a, h0=h0_mb, masks=m_mb)
                    
                    # PPO loss computation
                    ratio = (new_logp - lp_old).exp()
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                    loss_pi = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    if self.clip_value_loss:
                        v_clipped = v_old + (v - v_old).clamp(-self.clip_epsilon, self.clip_epsilon)
                        loss_v = 0.5 * torch.max((v - ret_mb).pow(2), (v_clipped - ret_mb).pow(2)).mean()
                    else:
                        loss_v = 0.5 * (v - ret_mb).pow(2).mean()
                    
                    ent = entropy.mean()
                    loss = loss_pi + self.value_coef * loss_v - self.entropy_coef * ent

                    # Backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Metrics
                    with torch.no_grad():
                        kl = (lp_old - new_logp).mean().abs()
                        clipfrac = (ratio.gt(1.0 + self.clip_epsilon) | ratio.lt(1.0 - self.clip_epsilon)).float().mean()

                    metrics["loss_pi"] += loss_pi.item()
                    metrics["loss_v"] += loss_v.item()
                    metrics["entropy"] += ent.item()
                    metrics["kl"] += kl.item()
                    metrics["clipfrac"] += clipfrac.item()
                    num_updates += 1
        else:
            # For non-recurrent policies, flatten everything
            Tflat = T * B
            o = obs.reshape(Tflat, *obs.shape[2:])
            a = actions.reshape(Tflat, *actions.shape[2:]) if actions.dim() > 2 else actions.reshape(Tflat)
            lp_old = old_logp.reshape(Tflat)
            v_old = values.reshape(Tflat)
            adv_flat = adv.reshape(Tflat)
            ret_flat = returns.reshape(Tflat)
            
            mb_size = max(1, min(self.batch_size, Tflat))
            
            for epoch in range(self.epochs):
                perm = torch.randperm(Tflat, device=self.device)
                for start in range(0, Tflat, mb_size):
                    end = min(start + mb_size, Tflat)
                    idx = perm[start:end]
                    
                    # Forward pass
                    new_logp, entropy, v = self.policy.evaluate_actions(o[idx], a[idx], h0=None, masks=None)
                    
                    # PPO loss computation
                    ratio = (new_logp - lp_old[idx]).exp()
                    surr1 = ratio * adv_flat[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_flat[idx]
                    loss_pi = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    if self.clip_value_loss:
                        v_clipped = v_old[idx] + (v - v_old[idx]).clamp(-self.clip_epsilon, self.clip_epsilon)
                        loss_v = 0.5 * torch.max((v - ret_flat[idx]).pow(2), (v_clipped - ret_flat[idx]).pow(2)).mean()
                    else:
                        loss_v = 0.5 * (v - ret_flat[idx]).pow(2).mean()
                    
                    ent = entropy.mean()
                    loss = loss_pi + self.value_coef * loss_v - self.entropy_coef * ent

                    # Backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Metrics
                    with torch.no_grad():
                        kl = (lp_old[idx] - new_logp).mean().abs()
                        clipfrac = (ratio.gt(1.0 + self.clip_epsilon) | ratio.lt(1.0 - self.clip_epsilon)).float().mean()

                    metrics["loss_pi"] += loss_pi.item()
                    metrics["loss_v"] += loss_v.item()
                    metrics["entropy"] += ent.item()
                    metrics["kl"] += kl.item()
                    metrics["clipfrac"] += clipfrac.item()
                    num_updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] = metrics[k] / max(1, num_updates)
            
        return {**metrics, "updates": num_updates}
