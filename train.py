"""
train.py – Train RL agent for Hospital Resource Allocation
==========================================================
Usage:
    python train.py --config configs/qlearning_v1.yaml
    python train.py --config configs/qlearning_v2.yaml

MLOps features implemented:
    ✓ Config-driven (YAML)
    ✓ CSV experiment logging (run_id, episodes, avg_reward, avg_wait, params)
    ✓ Policy versioning (models/policy_v1.pkl, policy_v2.pkl, ...)
    ✓ Reproducible via random seeds
    ✓ MLflow tracking (if available)
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
import yaml
import numpy as np

# ── Try MLflow (graceful fallback) ───────────────────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[Warning] MLflow not installed – logging to CSV only.")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.hospital_env import HospitalEnv, HospitalConfig
from sim.q_agent import QLearningAgent


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_csv_log(log_dir: str, run_id: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"results_{run_id}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "episode", "total_reward", "avg_reward",
            "avg_wait_time", "epsilon", "q_table_size",
            "alpha", "gamma", "epsilon_init", "epsilon_decay",
        ])
    return path


def append_csv_log(path: str, row: list):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
def train(config_path: str):
    cfg = load_config(config_path)

    # ── Reproducibility ──────────────────────
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    import random; random.seed(seed)

    run_id = cfg.get("run_id", str(uuid.uuid4())[:8])
    n_episodes = cfg["training"]["n_episodes"]
    log_every   = cfg["training"].get("log_every", 10)
    save_checkpoints = cfg["training"].get("save_checkpoints", [100, 200])

    print(f"\n{'='*60}")
    print(f"  Hospital Resource Allocation – RL Training")
    print(f"  Run ID  : {run_id}")
    print(f"  Config  : {config_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"{'='*60}\n")

    # ── Environment ──────────────────────────
    env_cfg = HospitalConfig(
        episode_length=cfg["env"].get("episode_length", 100),
    )
    env = HospitalEnv(env_cfg)

    # ── Agent ────────────────────────────────
    agent_cfg = cfg["agent"]
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        epsilon=agent_cfg["epsilon"],
        epsilon_min=agent_cfg["epsilon_min"],
        epsilon_decay=agent_cfg["epsilon_decay"],
    )

    # ── Logging ──────────────────────────────
    log_dir = "experiments"
    csv_path = setup_csv_log(log_dir, run_id)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("hospital-resource-rl")
        mlflow_run = mlflow.start_run(run_name=run_id)
        mlflow.log_params({
            "alpha": agent_cfg["alpha"],
            "gamma": agent_cfg["gamma"],
            "epsilon": agent_cfg["epsilon"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "n_episodes": n_episodes,
            "seed": seed,
        })

    # ── Training loop ────────────────────────
    episode_rewards = []
    episode_waits   = []
    best_avg_reward = -np.inf
    start_time = time.time()

    for ep in range(1, n_episodes + 1):
        state_cont = env.reset()
        state = env.get_discrete_state()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state_cont, reward, done, info = env.step(action)
            next_state = env.get_discrete_state()

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        avg_wait = env.avg_wait_time()
        avg_reward = total_reward / env_cfg.episode_length
        episode_rewards.append(avg_reward)
        episode_waits.append(avg_wait)
        agent.decay_epsilon()

        # ── CSV log ──────────────────────────
        if ep % log_every == 0 or ep == 1:
            append_csv_log(csv_path, [
                run_id, ep, round(total_reward, 4), round(avg_reward, 4),
                round(avg_wait, 4), round(agent.epsilon, 4),
                agent.q_table_size(),
                agent_cfg["alpha"], agent_cfg["gamma"],
                agent_cfg["epsilon"], agent_cfg["epsilon_decay"],
            ])
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"AvgReward: {avg_reward:6.3f} | "
                f"AvgWait: {avg_wait:5.2f}h | "
                f"ε: {agent.epsilon:.3f} | "
                f"Q-states: {agent.q_table_size()}"
            )

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "avg_reward": avg_reward,
                    "avg_wait_time": avg_wait,
                    "epsilon": agent.epsilon,
                    "q_table_size": agent.q_table_size(),
                }, step=ep)

        # ── Save checkpoints ─────────────────
        if ep in save_checkpoints:
            version = save_checkpoints.index(ep) + 1
            policy_path = f"models/policy_v{version}.pkl"
            agent.save(policy_path)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save("models/policy_best.pkl")

    # ── Final policy ─────────────────────────
    final_path = f"models/policy_{run_id}_final.pkl"
    agent.save(final_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best avg reward : {best_avg_reward:.4f}")
    print(f"  CSV log         : {csv_path}")
    print(f"  Final policy    : {final_path}")
    print(f"{'='*60}\n")

    # ── Summary JSON ─────────────────────────
    summary = {
        "run_id": run_id,
        "config": config_path,
        "n_episodes": n_episodes,
        "best_avg_reward": round(best_avg_reward, 4),
        "final_avg_wait_time": round(float(np.mean(episode_waits[-20:])), 4),
        "final_epsilon": round(agent.epsilon, 4),
        "q_table_size": agent.q_table_size(),
        "training_time_s": round(elapsed, 2),
    }
    summary_path = f"experiments/summary_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(final_path)
        mlflow.end_run()

    return agent, episode_rewards, episode_waits


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/qlearning_v1.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(args.config)
