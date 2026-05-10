"""
Hospital Resource Allocation Simulator
======================================
RL Environment for allocating hospital resources (ICU beds, doctors, nurses, ventilators)
across departments (Emergency, ICU, General Ward, OT) to minimize patient wait time
and maximize care quality.

SDG Link: SDG 3 – Good Health and Well-Being
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List


# ─────────────────────────────────────────────
# Hospital Configuration
# ─────────────────────────────────────────────
@dataclass
class HospitalConfig:
    num_departments: int = 4           # Emergency, ICU, General, OT
    total_doctors: int = 20
    total_nurses: int = 40
    total_icu_beds: int = 10
    total_ventilators: int = 8
    total_general_beds: int = 50
    max_queue_per_dept: int = 20
    episode_length: int = 100          # timesteps per episode (e.g., 1 timestep = 1 hour)

    dept_names: tuple = ("Emergency", "ICU", "General Ward", "OT")

    # Patient arrival rates per department per timestep (Poisson lambda)
    arrival_rates: tuple = (3.0, 1.0, 2.0, 0.5)

    # Severity weights: higher = more critical
    severity_weights: tuple = (1.5, 2.0, 1.0, 1.2)


# ─────────────────────────────────────────────
# Hospital Environment
# ─────────────────────────────────────────────
class HospitalEnv:
    """
    State:
        - queue lengths at each department          (4 values)
        - doctors currently allocated per dept      (4 values)
        - nurses currently allocated per dept       (4 values)
        - available ICU beds                        (1 value)
        - available ventilators                     (1 value)
        Total state dim: 14 (discretized for Q-learning)

    Action:
        One of 9 discrete actions:
          0 – Balanced allocation (default)
          1 – Boost Emergency resources
          2 – Boost ICU resources
          3 – Boost General Ward resources
          4 – Boost OT resources
          5 – Emergency + ICU priority
          6 – ICU + General priority
          7 – Emergency critical surge mode
          8 – Reduce all (conserve)

    Reward:
        Negative of weighted average wait time across departments
        Bonus for keeping ICU/ventilator utilization < 90%
        Penalty for queue overflow
    """

    DEPT_EMERGENCY = 0
    DEPT_ICU = 1
    DEPT_GENERAL = 2
    DEPT_OT = 3

    def __init__(self, config: HospitalConfig = None):
        self.config = config or HospitalConfig()
        self.n_actions = 9
        self.action_space_n = self.n_actions

        # State bins for discretization (for Q-table)
        self.queue_bins = [0, 3, 7, 12, 20]   # 4 levels
        self.resource_bins = [0, 5, 10, 15, 20]

        self.reset()

    def reset(self) -> np.ndarray:
        cfg = self.config
        # Queues per department
        self.queues = np.array([
            random.randint(0, 5),
            random.randint(0, 3),
            random.randint(0, 4),
            random.randint(0, 2),
        ], dtype=float)

        # Initial balanced allocation
        self.doctors = np.array([5, 5, 7, 3], dtype=float)
        self.nurses  = np.array([10, 10, 15, 5], dtype=float)
        self.icu_beds_used = random.randint(0, 4)
        self.vents_used    = random.randint(0, 3)

        self.timestep = 0
        self.wait_times_history = []
        return self._get_state()

    # ── Internal helpers ─────────────────────
    def _get_state(self) -> np.ndarray:
        """Returns raw continuous state vector."""
        return np.concatenate([
            self.queues,
            self.doctors,
            self.nurses,
            [self.config.total_icu_beds - self.icu_beds_used],
            [self.config.total_ventilators - self.vents_used],
        ])

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete tuple for Q-table key."""
        queues = state[:4]
        discretized = []
        for q in queues:
            bin_idx = np.digitize(q, self.queue_bins) - 1
            discretized.append(min(bin_idx, len(self.queue_bins) - 2))
        # Summarize resource pressure
        doc_pressure = int(np.sum(self.doctors) / self.config.total_doctors * 3)
        icu_free = int((self.config.total_icu_beds - self.icu_beds_used) / self.config.total_icu_beds * 3)
        discretized.extend([doc_pressure, icu_free])
        return tuple(discretized)

    def get_discrete_state(self) -> tuple:
        return self._discretize_state(self._get_state())

    # ── Action → Resource Allocation ────────
    def _apply_action(self, action: int):
        cfg = self.config
        # Reset to base then adjust
        if action == 0:   # Balanced
            self.doctors = np.array([5.0, 5.0, 7.0, 3.0])
            self.nurses  = np.array([10.0, 10.0, 15.0, 5.0])
        elif action == 1:  # Boost Emergency
            self.doctors = np.array([8.0, 4.0, 6.0, 2.0])
            self.nurses  = np.array([15.0, 8.0, 13.0, 4.0])
        elif action == 2:  # Boost ICU
            self.doctors = np.array([4.0, 8.0, 6.0, 2.0])
            self.nurses  = np.array([8.0, 16.0, 12.0, 4.0])
        elif action == 3:  # Boost General
            self.doctors = np.array([4.0, 4.0, 10.0, 2.0])
            self.nurses  = np.array([8.0, 8.0, 20.0, 4.0])
        elif action == 4:  # Boost OT
            self.doctors = np.array([4.0, 4.0, 6.0, 6.0])
            self.nurses  = np.array([8.0, 8.0, 14.0, 10.0])
        elif action == 5:  # Emergency + ICU
            self.doctors = np.array([7.0, 7.0, 5.0, 1.0])
            self.nurses  = np.array([13.0, 14.0, 11.0, 2.0])
        elif action == 6:  # ICU + General
            self.doctors = np.array([3.0, 7.0, 8.0, 2.0])
            self.nurses  = np.array([6.0, 14.0, 17.0, 3.0])
        elif action == 7:  # Emergency critical surge
            self.doctors = np.array([10.0, 4.0, 5.0, 1.0])
            self.nurses  = np.array([18.0, 8.0, 11.0, 3.0])
        elif action == 8:  # Conserve
            self.doctors = np.array([4.0, 3.0, 6.0, 2.0])
            self.nurses  = np.array([8.0, 6.0, 14.0, 4.0])

        # Clip to total resources
        total_doc = self.doctors.sum()
        if total_doc > cfg.total_doctors:
            self.doctors = self.doctors / total_doc * cfg.total_doctors
        total_nur = self.nurses.sum()
        if total_nur > cfg.total_nurses:
            self.nurses = self.nurses / total_nur * cfg.total_nurses

    # ── Step ────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.config
        self._apply_action(action)

        # Arrivals
        arrivals = np.array([
            np.random.poisson(rate) for rate in cfg.arrival_rates
        ], dtype=float)

        # Service rate depends on staff allocation
        service_rates = (self.doctors * 1.5 + self.nurses * 0.5) / 10.0

        # Update queues
        self.queues = np.clip(self.queues + arrivals - service_rates, 0, cfg.max_queue_per_dept)

        # ICU / ventilator usage fluctuates with ICU queue
        icu_demand = int(self.queues[self.DEPT_ICU] * 0.6)
        self.icu_beds_used = min(icu_demand, cfg.total_icu_beds)
        vent_demand = int(self.queues[self.DEPT_ICU] * 0.4)
        self.vents_used = min(vent_demand, cfg.total_ventilators)

        # Wait times per dept (queue / service_rate, in hours)
        wait_times = np.where(service_rates > 0, self.queues / service_rates, self.queues * 2)
        weighted_wait = np.dot(wait_times, cfg.severity_weights) / sum(cfg.severity_weights)
        self.wait_times_history.append(weighted_wait)

        # ── Reward ──────────────────────────
        reward = -weighted_wait

        # Bonus: keep ICU utilization reasonable
        icu_util = self.icu_beds_used / cfg.total_icu_beds
        if icu_util < 0.9:
            reward += 0.5
        else:
            reward -= 1.0

        # Penalty: queue overflow
        overflow = np.sum(self.queues >= cfg.max_queue_per_dept)
        reward -= overflow * 2.0

        self.timestep += 1
        done = self.timestep >= cfg.episode_length
        info = {
            "queues": self.queues.copy(),
            "wait_times": wait_times.copy(),
            "weighted_wait": weighted_wait,
            "icu_utilization": icu_util,
            "vents_used": self.vents_used,
            "doctors": self.doctors.copy(),
            "nurses": self.nurses.copy(),
        }
        return self._get_state(), reward, done, info

    # ── Fixed-timer baseline ─────────────────
    def fixed_timer_action(self) -> int:
        """Always returns balanced allocation – our baseline policy."""
        return 0

    def avg_wait_time(self) -> float:
        if not self.wait_times_history:
            return 0.0
        return float(np.mean(self.wait_times_history))
