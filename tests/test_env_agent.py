"""
tests/test_env_agent.py – Unit tests for Hospital RL Environment & Agent
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sim.hospital_env import HospitalEnv, HospitalConfig
from sim.q_agent import QLearningAgent


class TestHospitalEnv:
    def setup_method(self):
        self.env = HospitalEnv()

    def test_reset_returns_state(self):
        state = self.env.reset()
        assert state is not None
        assert len(state) == 14  # 4 queues + 4 doctors + 4 nurses + icu + vents

    def test_step_valid_action(self):
        self.env.reset()
        for action in range(self.env.n_actions):
            self.env.reset()
            state, reward, done, info = self.env.step(action)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert "queues" in info

    def test_episode_terminates(self):
        cfg = HospitalConfig(episode_length=10)
        env = HospitalEnv(cfg)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
        assert steps == 10

    def test_queues_bounded(self):
        self.env.reset()
        for _ in range(50):
            state, _, done, info = self.env.step(np.random.randint(0, 9))
            assert all(0 <= q <= self.env.config.max_queue_per_dept for q in info["queues"])
            if done:
                self.env.reset()

    def test_discrete_state_is_tuple(self):
        self.env.reset()
        ds = self.env.get_discrete_state()
        assert isinstance(ds, tuple)

    def test_fixed_timer_action(self):
        assert self.env.fixed_timer_action() == 0


class TestQLearningAgent:
    def setup_method(self):
        self.agent = QLearningAgent(n_actions=9)

    def test_action_in_range(self):
        state = (0, 1, 2, 0, 2, 1)
        for _ in range(20):
            action = self.agent.select_action(state)
            assert 0 <= action < 9

    def test_greedy_action(self):
        state = (0, 1, 2, 0, 2, 1)
        self.agent.q_table[state][3] = 100.0  # make action 3 clearly best
        action = self.agent.select_action(state, greedy=True)
        assert action == 3

    def test_update_changes_q_value(self):
        s  = (0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 0, 0, 0, 0)
        before = self.agent.q_table[s][0]
        self.agent.update(s, 0, 1.0, s2, False)
        after = self.agent.q_table[s][0]
        assert after != before

    def test_epsilon_decays(self):
        eps_before = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon <= eps_before

    def test_epsilon_floor(self):
        self.agent.epsilon = 0.06
        self.agent.epsilon_min = 0.05
        self.agent.epsilon_decay = 0.5
        self.agent.decay_epsilon()
        assert self.agent.epsilon >= self.agent.epsilon_min

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "policy_test.pkl")
        self.agent.q_table[(0, 1, 2, 0, 1, 0)][5] = 42.0
        self.agent.save(path)
        loaded = QLearningAgent.load(path)
        assert abs(loaded.q_table[(0, 1, 2, 0, 1, 0)][5] - 42.0) < 1e-6
