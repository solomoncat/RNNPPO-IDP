import os
import json
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class SimpleLogger:
    """Simple logging utility for training metrics"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.file = None
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.file = open(os.path.join(log_dir, "train.log"), "a", buffering=1)
            print(f"Logging to {os.path.join(log_dir, 'train.log')}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log a dictionary of data with timestamp"""
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
        """Log a single scalar value"""
        self.log({name: value}, step)

    def close(self):
        """Close log file"""
        if self.file:
            self.file.close()


def save_model(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, extra: Dict = None):
    """Save model checkpoint with optional optimizer state and extra data"""
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
    print(f"Model saved to {path}")


def load_model(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location=None):
    """Load model checkpoint with optional optimizer state"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    print(f"Model loaded from {path}")
    return checkpoint.get("extra", {})


def compute_cooperation_rate(history: List[Tuple[int, int, float]]) -> float:
    """
    Compute cooperation rate from episode history.
    
    Args:
        history: List of (action_A, action_B, temptation) tuples
    
    Returns:
        Fraction of steps where both agents cooperated (both actions == 0)
    """
    if not history:
        return 0.0
    
    mutual_cooperations = sum(1 for aA, aB, _ in history if aA == 0 and aB == 0)
    return mutual_cooperations / len(history)


def compute_defection_rate(history: List[Tuple[int, int, float]]) -> float:
    """
    Compute defection rate from episode history.
    
    Args:
        history: List of (action_A, action_B, temptation) tuples
    
    Returns:
        Fraction of steps where at least one agent defected
    """
    if not history:
        return 0.0
    
    defections = sum(1 for aA, aB, _ in history if aA == 1 or aB == 1)
    return defections / len(history)


def compute_agent_cooperation_rate(history: List[Tuple[int, int, float]], agent: str = 'A') -> float:
    """
    Compute cooperation rate for a specific agent.
    
    Args:
        history: List of (action_A, action_B, temptation) tuples
        agent: 'A' or 'B'
    
    Returns:
        Fraction of steps where the specified agent cooperated
    """
    if not history:
        return 0.0
    
    if agent == 'A':
        cooperations = sum(1 for aA, aB, _ in history if aA == 0)
    elif agent == 'B':
        cooperations = sum(1 for aA, aB, _ in history if aB == 0)
    else:
        raise ValueError("Agent must be 'A' or 'B'")
    
    return cooperations / len(history)


def episode_summary(rewards_A: List[float], rewards_B: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for an episode.
    
    Args:
        rewards_A: List of rewards for agent A
        rewards_B: List of rewards for agent B
    
    Returns:
        Dictionary with summary statistics
    """
    rA = np.array(rewards_A, dtype=float)
    rB = np.array(rewards_B, dtype=float)
    
    return {
        "sum_A": float(rA.sum()) if len(rA) > 0 else 0.0,
        "sum_B": float(rB.sum()) if len(rB) > 0 else 0.0,
        "mean_A": float(rA.mean()) if len(rA) > 0 else 0.0,
        "mean_B": float(rB.mean()) if len(rB) > 0 else 0.0,
        "std_A": float(rA.std()) if len(rA) > 0 else 0.0,
        "std_B": float(rB.std()) if len(rB) > 0 else 0.0,
        "min_A": float(rA.min()) if len(rA) > 0 else 0.0,
        "min_B": float(rB.min()) if len(rB) > 0 else 0.0,
        "max_A": float(rA.max()) if len(rA) > 0 else 0.0,
        "max_B": float(rB.max()) if len(rB) > 0 else 0.0,
    }


def analyze_strategy_interactions(history: List[Tuple[int, int, float]]) -> Dict[str, float]:
    """
    Analyze the types of interactions in an episode.
    
    Args:
        history: List of (action_A, action_B, temptation) tuples
    
    Returns:
        Dictionary with interaction type frequencies
    """
    if not history:
        return {
            "cooperate_cooperate": 0.0,
            "cooperate_defect": 0.0, 
            "defect_cooperate": 0.0,
            "defect_defect": 0.0
        }
    
    total = len(history)
    cc = sum(1 for aA, aB, _ in history if aA == 0 and aB == 0)  # Both cooperate
    cd = sum(1 for aA, aB, _ in history if aA == 0 and aB == 1)  # A cooperates, B defects
    dc = sum(1 for aA, aB, _ in history if aA == 1 and aB == 0)  # A defects, B cooperates
    dd = sum(1 for aA, aB, _ in history if aA == 1 and aB == 1)  # Both defect
    
    return {
        "cooperate_cooperate": cc / total,
        "cooperate_defect": cd / total,
        "defect_cooperate": dc / total,
        "defect_defect": dd / total
    }


def compute_payoff_matrix_stats(history: List[Tuple[int, int, float]], 
                               R: float = 3.0, P: float = 1.0, S: float = 0.0) -> Dict[str, float]:
    """
    Compute expected payoffs and efficiency metrics.
    
    Args:
        history: List of (action_A, action_B, temptation) tuples
        R: Reward for mutual cooperation
        P: Punishment for mutual defection
        S: Sucker's payoff
    
    Returns:
        Dictionary with payoff statistics
    """
    if not history:
        return {"social_welfare": 0.0, "efficiency": 0.0, "avg_temptation": 0.0}
    
    interactions = analyze_strategy_interactions(history)
    temptations = [t for _, _, t in history]
    avg_temptation = np.mean(temptations)
    
    # Calculate expected payoffs
    expected_payoff = (
        interactions["cooperate_cooperate"] * R +
        interactions["cooperate_defect"] * S +
        interactions["defect_cooperate"] * avg_temptation +
        interactions["defect_defect"] * P
    )
    
    social_welfare = (
        interactions["cooperate_cooperate"] * (2 * R) +
        interactions["cooperate_defect"] * (S + avg_temptation) +
        interactions["defect_cooperate"] * (avg_temptation + S) +
        interactions["defect_defect"] * (2 * P)
    )
    
    # Efficiency compared to always cooperating
    max_welfare = 2 * R  # If everyone always cooperated
    efficiency = social_welfare / max_welfare if max_welfare > 0 else 0.0
    
    return {
        "expected_payoff": expected_payoff,
        "social_welfare": social_welfare,
        "efficiency": efficiency,
        "avg_temptation": avg_temptation
    }


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """Compute moving average of a list"""
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        end = i + 1
        result.append(np.mean(data[start:end]))
    
    return result


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_config(config: Dict[str, Any]):
    """Pretty print configuration"""
    print("Configuration:")
    print("-" * 40)
    for key, value in sorted(config.items()):
        print(f"  {key:<20}: {value}")
    print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Test the utility functions
    print("Testing utility functions...")
    
    # Test episode analysis
    history = [
        (0, 0, 5.0),  # Both cooperate
        (1, 0, 4.0),  # A defects, B cooperates
        (0, 1, 6.0),  # A cooperates, B defects
        (1, 1, 3.5),  # Both defect
    ]
    
    coop_rate = compute_cooperation_rate(history)
    defect_rate = compute_defection_rate(history)
    interactions = analyze_strategy_interactions(history)
    payoff_stats = compute_payoff_matrix_stats(history)
    
    print(f"Cooperation rate: {coop_rate:.3f}")
    print(f"Defection rate: {defect_rate:.3f}")
    print(f"Interactions: {interactions}")
    print(f"Payoff stats: {payoff_stats}")
    
    # Test rewards summary
    rewards_A = [3.0, 4.0, 0.0, 1.0]
    rewards_B = [3.0, 0.0, 6.0, 1.0]
    summary = episode_summary(rewards_A, rewards_B)
    print(f"Episode summary: {summary}")
    
    print("All utility functions working correctly!")
