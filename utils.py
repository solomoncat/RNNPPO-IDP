import os
import json
import time
import torch
import numpy as np
from typing import Any, Dict, Optional


class SimpleLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.file = open(os.path.join(log_dir, "train.log"), "a", buffering=1)

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        timestamp = time.time() - self.start_time
        entry = {"t": round(timestamp, 3)}
        if step is not None:
            entry["step"] = step
        entry.update(data)
        line = json.dumps(entry)
        print(line, flush=True)
        if self.file:
            self.file.write(line + "\n")

    def scalar(self, name: str, value: float, step: int):
        self.log({name: value}, step)

    def close(self):
        if self.file:
            self.file.close()


def save_model(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, extra: Dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "timestamp": time.time()
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_model(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("extra", {})


def compute_cooperation_rate(history):
    if not history:
        return 0.0
    coop = sum(1 for aA, aB, _ in history if aA == 0 and aB == 0)
    total = len(history)
    return coop / total


def compute_defection_rate(history):
    if not history:
        return 0.0
    defections = sum(1 for aA, aB, _ in history if aA == 1 or aB == 1)
    total = len(history)
    return defections / total


def episode_summary(rewards_A, rewards_B):
    """
    Given per-step reward sequences (lists or arrays) for two agents,
    returns dict with totals and averages.
    """
    rA = np.array(rewards_A, dtype=float)
    rB = np.array(rewards_B, dtype=float)
    return {
        "sum_A": float(rA.sum()),
        "sum_B": float(rB.sum()),
        "mean_A": float(rA.mean()) if len(rA) > 0 else 0.0,
        "mean_B": float(rB.mean()) if len(rB) > 0 else 0.0,
    }
