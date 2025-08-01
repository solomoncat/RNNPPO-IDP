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
from evaluate import strategy_analysis, AlwaysCooperate, AlwaysDefect, TitForTat, GrimTrigger

def load_config(path):
    if path:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def make_trainer(cfg):
    device = torch.device(cfg.get("device", "cpu"))
    env = PrisonersDilemmaEnv(
        max_steps=cfg.get("env_max_steps", 100),
        history_length=cfg.get("history_length", 10),
        temptation_range=(cfg.get("temptation_low", 3.0), cfg.get("temptation_high", 10.0)),
        seed=cfg.get("seed", None)
    )
    sample_obs = env.reset()["A"]
    obs_dim = sample_obs.shape[0]

    agent = PPOAgent(
        obs_size=obs_dim,
        action_size=2,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 1),
        rnn_type=cfg.get("rnn_type", "LSTM"),
        device=device
    )
    policy = agent.policy  # underlying network
    ppo = PPO(
        policy,
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
    env, agent, ppo, device = make_trainer(cfg)
    logger = SimpleLogger(cfg.get("log_dir", "runs"))
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    rollout_length = cfg.get("rollout_length", 128)
    num_envs = cfg.get("num_envs", 16)
    num_iters = cfg.get("num_iterations", 1000)
    batch_size = cfg.get("batch_size", 64)
    shared = cfg.get("shared_policy", True)

    # simple single-env loop for baseline (vectorization can be added later)
    for iteration in range(1, num_iters + 1):
        buffer = RolloutBuffer(gamma=cfg.get("gamma", 0.99), lam=cfg.get("gae_lambda", 0.95))
        # reset hidden
        agent.reset_hidden(batch_size=1)
        # collect rollout
        obs_dict = env.reset()
        obs_A = obs_dict["A"]
        obs_B = obs_dict["B"]
        for t in range(rollout_length):
            outA = agent.act(obs_A)
            outB = agent.act(obs_B)
            aA = int(outA["action"])
            aB = int(outB["action"])
            logpA = torch.tensor(outA["log_prob"])
            logpB = torch.tensor(outB["log_prob"])
            vA = torch.tensor(outA["value"])
            vB = torch.tensor(outB["value"])

            next_obs, rewards, done, _ = env.step({"A": aA, "B": aB})
            rA = rewards["A"]
            rB = rewards["B"]
            obs_A = next_obs["A"]
            obs_B = next_obs["B"]

            mask = torch.tensor(0.0 if done else 1.0)
            # store both roles separately into buffer (shared policy case)
            buffer.add_step(
                obs=torch.tensor(obs_A, dtype=torch.float32),
                action=torch.tensor(aA),
                logprob=logpA,
                value=vA,
                reward=torch.tensor(rA, dtype=torch.float32),
                done=torch.tensor(done, dtype=torch.float32),
                mask=mask
            )
            buffer.add_step(
                obs=torch.tensor(obs_B, dtype=torch.float32),
                action=torch.tensor(aB),
                logprob=logpB,
                value=vB,
                reward=torch.tensor(rB, dtype=torch.float32),
                done=torch.tensor(done, dtype=torch.float32),
                mask=mask
            )
            if done:
                break
        # bootstrap last value
        last = agent.get_value(obs_A)
        buffer.set_last_value(torch.tensor(last, dtype=torch.float32))
        batch = buffer.compute_gae()
        stats = ppo.update(batch)
        logger.log({
            "iteration": iteration,
            "loss_pi": stats.get("loss_pi", 0.0),
            "loss_v": stats.get("loss_v", 0.0),
            "entropy": stats.get("entropy", 0.0),
            "kl": stats.get("kl", 0.0),
            "clipfrac": stats.get("clipfrac", 0.0)
        }, step=iteration)
        if iteration % cfg.get("save_interval", 100) == 0:
            path = save_dir / f"checkpoint_{iteration}.pt"
            save_model(str(path), agent.policy, optimizer=ppo.optimizer, extra={"iteration": iteration})

def evaluate(cfg):
    env_kwargs = {
        "max_steps": cfg.get("env_max_steps", 100),
        "history_length": cfg.get("history_length", 10),
        "temptation_range": (cfg.get("temptation_low", 3.0), cfg.get("temptation_high", 10.0)),
        "seed": cfg.get("seed", None)
    }
    device = torch.device(cfg.get("device", "cpu"))
    env = PrisonersDilemmaEnv(**env_kwargs)
    sample_obs = env.reset()["A"]
    obs_dim = sample_obs.shape[0]
    agent = PPOAgent(
        obs_size=obs_dim,
        action_size=2,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 1),
        rnn_type=cfg.get("rnn_type", "LSTM"),
        device=device
    )
    if cfg.get("load_checkpoint"):
        load_model(cfg["load_checkpoint"], agent.policy)
    baselines = {
        "AlwaysCooperate": AlwaysCooperate,
        "AlwaysDefect": AlwaysDefect,
        "TitForTat": TitForTat,
        "GrimTrigger": GrimTrigger
    }
    results = strategy_analysis(agent, env_kwargs, baselines, episodes=cfg.get("eval_episodes", 100),
                                max_steps=cfg.get("env_max_steps", 100))
    print("Evaluation summary:", results)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--load-checkpoint', type=str, default=None)
    args, rest = parser.parse_known_args()
    cfg = load_config(args.config)
    cfg.update(vars(args))
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    if cfg.get("mode") == "train":
        train(cfg)
    else:
        evaluate(cfg)
