"""
evaluate.py – Baseline vs RL Policy Comparison
================================================
Runs both Fixed-Timer (balanced allocation always) and RL policy
on the same Hospital simulator and produces comparison table + plots.

Usage:
    python evaluate.py --policy models/policy_best.pkl --episodes 50
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.hospital_env import HospitalEnv, HospitalConfig
from sim.q_agent import QLearningAgent


# ─────────────────────────────────────────────
def run_episodes(env: HospitalEnv, agent=None, n_episodes: int = 50, seed: int = 99):
    """
    Run n episodes. If agent is None → fixed-timer baseline.
    Returns dict of metrics lists.
    """
    np.random.seed(seed)
    import random; random.seed(seed)

    all_rewards, all_waits, all_icu_utils = [], [], []
    all_queue_history = []

    for ep in range(n_episodes):
        state_cont = env.reset()
        state = env.get_discrete_state()
        ep_reward, ep_waits, ep_icu = 0.0, [], []
        queue_hist = []
        done = False

        while not done:
            if agent is None:
                action = env.fixed_timer_action()
            else:
                action = agent.select_action(state, greedy=True)

            next_cont, reward, done, info = env.step(action)
            state = env.get_discrete_state()
            ep_reward += reward
            ep_waits.append(info["weighted_wait"])
            ep_icu.append(info["icu_utilization"])
            queue_hist.append(info["queues"].copy())

        all_rewards.append(ep_reward / env.config.episode_length)
        all_waits.append(float(np.mean(ep_waits)))
        all_icu_utils.append(float(np.mean(ep_icu)))
        all_queue_history.append(np.array(queue_hist))

    return {
        "rewards": all_rewards,
        "waits": all_waits,
        "icu_utils": all_icu_utils,
        "queue_history": all_queue_history,
    }


# ─────────────────────────────────────────────
def print_comparison_table(baseline: dict, rl: dict):
    metrics = [
        ("Avg Wait Time (hours)", "waits", "lower"),
        ("Avg Reward", "rewards", "higher"),
        ("ICU Utilization", "icu_utils", "lower"),
    ]

    print("\n" + "=" * 60)
    print("  COMPARISON: Fixed-Timer Baseline vs RL Policy")
    print("=" * 60)
    print(f"{'Metric':<30} {'Fixed-Timer':>12} {'RL-Policy':>12} {'Δ':>10}")
    print("-" * 60)

    results = {}
    for label, key, direction in metrics:
        b_val = np.mean(baseline[key])
        r_val = np.mean(rl[key])
        delta = r_val - b_val
        sign  = "↑" if delta > 0 else "↓"
        pct   = abs(delta / b_val * 100) if b_val != 0 else 0
        better = (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0)
        marker = "✓" if better else "✗"
        print(f"  {label:<28} {b_val:>12.3f} {r_val:>12.3f} {sign}{pct:.1f}% {marker}")
        results[key] = {"baseline": b_val, "rl": r_val, "delta_pct": pct, "better": better}

    print("=" * 60)

    # SDG Impact
    wait_improve = (np.mean(baseline["waits"]) - np.mean(rl["waits"])) / np.mean(baseline["waits"]) * 100
    print(f"\n  SDG-3 Impact: RL reduces avg patient wait time by {wait_improve:.1f}%")
    print(f"  This supports SDG 3 (Good Health & Well-Being) by improving")
    print(f"  hospital throughput and reducing critical care delays.\n")

    return results


# ─────────────────────────────────────────────
def plot_results(baseline: dict, rl: dict, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)

    episodes = range(1, len(baseline["rewards"]) + 1)
    dept_names = ["Emergency", "ICU", "General Ward", "OT"]
    colors = {"baseline": "#E74C3C", "rl": "#2ECC71"}

    # ── Plot 1: Avg Reward over Episodes ─────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, baseline["rewards"], color=colors["baseline"],
            alpha=0.6, label="Fixed-Timer Baseline", linewidth=1.5)
    ax.plot(episodes, rl["rewards"], color=colors["rl"],
            alpha=0.8, label="RL Policy (Q-learning)", linewidth=2)

    # Rolling averages
    w = min(5, len(episodes))
    ax.plot(episodes, np.convolve(baseline["rewards"], np.ones(w)/w, mode="same"),
            color=colors["baseline"], linewidth=2.5, linestyle="--")
    ax.plot(episodes, np.convolve(rl["rewards"], np.ones(w)/w, mode="same"),
            color=colors["rl"], linewidth=2.5, linestyle="--")

    ax.set_title("Average Reward per Episode", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "plot_reward_comparison.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ── Plot 2: Avg Wait Time over Episodes ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, baseline["waits"], color=colors["baseline"],
            alpha=0.5, linewidth=1.5, label="Fixed-Timer")
    ax.plot(episodes, rl["waits"], color=colors["rl"],
            alpha=0.8, linewidth=2, label="RL Policy")
    ax.fill_between(episodes, baseline["waits"], rl["waits"],
                    where=[b > r for b, r in zip(baseline["waits"], rl["waits"])],
                    alpha=0.2, color="#2ECC71", label="RL advantage zone")
    ax.set_title("Average Patient Wait Time per Episode (hours)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Wait Time (hours)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "plot_wait_comparison.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ── Plot 3: Queue lengths last episode ───
    bl_q = np.array(baseline["queue_history"][-1])  # shape (T, 4)
    rl_q = np.array(rl["queue_history"][-1])
    timesteps = range(len(bl_q))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Queue Lengths Over Time (Last Episode)", fontsize=14, fontweight="bold")
    for i, (ax, dept) in enumerate(zip(axes.flat, dept_names)):
        ax.plot(timesteps, bl_q[:, i], color=colors["baseline"],
                label="Fixed-Timer", alpha=0.7, linewidth=1.5)
        ax.plot(timesteps, rl_q[:, i], color=colors["rl"],
                label="RL Policy", linewidth=2)
        ax.set_title(dept)
        ax.set_xlabel("Timestep (hours)")
        ax.set_ylabel("Queue Length (patients)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "plot_queue_lengths.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p3}")

    return [p1, p2, p3]


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="models/policy_best.pkl")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    print("\n[Evaluate] Loading environment...")
    env = HospitalEnv()

    print("[Evaluate] Running Fixed-Timer Baseline...")
    baseline = run_episodes(env, agent=None, n_episodes=args.episodes)

    print("[Evaluate] Running RL Policy...")
    agent = QLearningAgent.load(args.policy)
    rl = run_episodes(env, agent=agent, n_episodes=args.episodes)

    results = print_comparison_table(baseline, rl)

    print("\n[Evaluate] Generating plots...")
    plots = plot_results(baseline, rl)

    # Save comparison JSON
    comparison = {
        "n_episodes": args.episodes,
        "baseline_avg_wait": round(float(np.mean(baseline["waits"])), 4),
        "rl_avg_wait": round(float(np.mean(rl["waits"])), 4),
        "baseline_avg_reward": round(float(np.mean(baseline["rewards"])), 4),
        "rl_avg_reward": round(float(np.mean(rl["rewards"])), 4),
        "plots": plots,
    }
    with open("reports/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n  Evaluation complete. Check reports/ folder.")


if __name__ == "__main__":
    main()
