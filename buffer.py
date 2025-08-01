import torch


class RolloutBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
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
        # each should be (B, ...) or (B,) for batch of environments at a time step
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def set_initial_hidden(self, h0):
        self.h0 = h0

    def set_last_value(self, last_value):
        self.last_value = last_value  # (B,)

    def compute_gae(self):
        # convert lists to tensors: shape (T, B, ...)
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

        # normalize advantages across all timesteps and batch
        flat_adv = advantages.reshape(-1)
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

    def get_minibatches(self, minibatch_size):
        """
        Yield minibatches for PPO update.
        Assumes compute_gae() has been called.
        Keeps sequence length intact: batch is over environments (B); T is full rollout length.
        """
        b = self.batch
        T, B = b["obs"].shape[0], b["obs"].shape[1]
        indices = torch.randperm(B)
        for start in range(0, B, minibatch_size):
            idx = indices[start : start + minibatch_size]
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
                "last_values": b["last_values"][idx] if b["last_values"] is not None else torch.zeros(idx.shape[0], device=b["obs"].device),
            }
            if b["h0"] is not None:
                h0 = b["h0"]
                if isinstance(h0, tuple):
                    mb["h0"] = (h0[0][:, idx], h0[1][:, idx])
                else:
                    mb["h0"] = h0[:, idx]
            yield mb
