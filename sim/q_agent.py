import random
"""
Q-Learning Agent for Hospital Resource Allocation
===================================================
Algorithm: Q-learning (off-policy TD control)
Why Q-learning?
    The state space (queue lengths + resource pressure + ICU availability)
    is naturally discretizable and small enough for a tabular Q-table.
    Q-learning is guaranteed to converge to the optimal policy in this setting
    and is interpretable – we can inspect the Q-table directly.

    DQN would be overkill for a state space of ~500 discrete states.
"""

import numpy as np
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning with ε-greedy exploration.

    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') – Q(s, a)]

    Params:
        alpha       – learning rate
        gamma       – discount factor
        epsilon     – initial exploration rate
        epsilon_min – floor for exploration
        epsilon_decay – multiplicative decay per episode
        n_actions   – number of discrete actions
    """

    def __init__(
        self,
        n_actions: int = 9,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict so unseen states start at 0
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))
        self.episode = 0

    # ── Action selection ─────────────────────
    def select_action(self, state: tuple, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    # ── Q-table update ───────────────────────
    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    # ── Epsilon decay ────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    # ── Persistence ─────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "episode": self.episode,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "n_actions": self.n_actions,
            }, f)
        print(f"[Agent] Saved policy → {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            n_actions=data["n_actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
        )
        agent.q_table = defaultdict(lambda: np.zeros(data["n_actions"]), data["q_table"])
        agent.epsilon = data["epsilon"]
        agent.episode = data["episode"]
        print(f"[Agent] Loaded policy ← {path}  (episode {agent.episode}, ε={agent.epsilon:.3f})")
        return agent

    def q_table_size(self) -> int:
        return len(self.q_table)


# Need random import
