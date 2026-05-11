"""
api.py – FastAPI REST API for Hospital RL Policy
=================================================
Serves the trained RL agent for real-time resource allocation decisions.

Endpoints:
    GET  /health             – liveness check
    POST /allocate           – get resource allocation for current hospital state
    GET  /monitoring/metrics – monitoring metrics (drift detection, logs)
    GET  /policy/info        – policy metadata

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
import sys
import time
import json
import logging
from datetime import datetime
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.hospital_env import HospitalEnv, HospitalConfig
from sim.q_agent import QLearningAgent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hospital-rl-api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hospital Resource Allocation – RL API",
    description="Q-Learning based resource allocator for SDG-3 hospital optimization",
    version="1.0.0",
)

# ── Global state ─────────────────────────────────────────────────────────────
POLICY_PATH = os.environ.get("POLICY_PATH", "models/policy_best.pkl")
agent: Optional[QLearningAgent] = None
env = HospitalEnv()

# Monitoring buffers (sliding window – last 100 predictions)
prediction_log: deque = deque(maxlen=100)
wait_time_log:  deque = deque(maxlen=100)
BASELINE_AVG_WAIT = 4.5    # hours – from fixed-timer baseline


@app.on_event("startup")
def load_policy():
    global agent
    if os.path.exists(POLICY_PATH):
        agent = QLearningAgent.load(POLICY_PATH)
        logger.info(f"Policy loaded from {POLICY_PATH}")
    else:
        logger.warning(f"No policy found at {POLICY_PATH}. Using random.")


# ── Request / Response Models ─────────────────────────────────────────────────
class HospitalState(BaseModel):
    emergency_queue: int = Field(ge=0, le=20, description="Patients waiting in Emergency")
    icu_queue:       int = Field(ge=0, le=20, description="Patients waiting for ICU")
    general_queue:   int = Field(ge=0, le=20, description="Patients in General Ward queue")
    ot_queue:        int = Field(ge=0, le=20, description="Patients waiting for OT")
    icu_beds_used:   int = Field(ge=0, le=10, description="ICU beds occupied")
    vents_used:      int = Field(ge=0, le=8,  description="Ventilators in use")


ACTION_NAMES = {
    0: "Balanced Allocation",
    1: "Boost Emergency Resources",
    2: "Boost ICU Resources",
    3: "Boost General Ward Resources",
    4: "Boost OT Resources",
    5: "Emergency + ICU Priority",
    6: "ICU + General Priority",
    7: "Emergency Critical Surge Mode",
    8: "Conserve Resources",
}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "policy_loaded": agent is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/allocate")
def allocate(state: HospitalState):
    """
    Given current hospital queues/resource state,
    returns the RL policy's recommended resource allocation.
    """
    t0 = time.time()

    # Build discrete state for Q-table lookup
    env.queues = np.array([
        state.emergency_queue, state.icu_queue,
        state.general_queue, state.ot_queue,
    ], dtype=float)
    env.icu_beds_used = state.icu_beds_used
    env.vents_used    = state.vents_used
    env.doctors = np.array([5.0, 5.0, 7.0, 3.0])   # reset for state computation
    env.nurses  = np.array([10.0, 10.0, 15.0, 5.0])

    discrete_state = env.get_discrete_state()

    if agent is None:
        action = 0
    else:
        action = agent.select_action(discrete_state, greedy=True)

    env._apply_action(action)

    latency_ms = round((time.time() - t0) * 1000, 2)

    response = {
        "action_id": action,
        "action_name": ACTION_NAMES[action],
        "recommended_allocation": {
            "doctors": {
                "Emergency": round(env.doctors[0]),
                "ICU":       round(env.doctors[1]),
                "General":   round(env.doctors[2]),
                "OT":        round(env.doctors[3]),
            },
            "nurses": {
                "Emergency": round(env.nurses[0]),
                "ICU":       round(env.nurses[1]),
                "General":   round(env.nurses[2]),
                "OT":        round(env.nurses[3]),
            },
        },
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ── Prediction logging (for monitoring) ──
    log_entry = {
        "timestamp": response["timestamp"],
        "action": action,
        "queues": [state.emergency_queue, state.icu_queue,
                   state.general_queue, state.ot_queue],
        "icu_util": state.icu_beds_used / 10,
    }
    prediction_log.append(log_entry)
    logger.info(f"Allocation: {ACTION_NAMES[action]} | latency={latency_ms}ms")

    return response


@app.get("/monitoring/metrics")
def monitoring_metrics():
    """
    Returns monitoring metrics for drift detection and system health.
    
    Monitoring Plan (for real-world deployment):
    - Average wait time drift: alert if > 20% above baseline
    - ICU utilization drift: alert if avg > 85% for 6h
    - Queue overflow events: count per 24h
    - Prediction confidence: avg Q-value gap between best/second-best action
    - Safety rules: no department should have 0 doctors for > 2 consecutive timesteps
    """
    if not prediction_log:
        return {"status": "no_data", "n_predictions": 0}

    recent = list(prediction_log)
    action_dist = {}
    icu_utils = []

    for entry in recent:
        a = entry["action"]
        action_dist[ACTION_NAMES[a]] = action_dist.get(ACTION_NAMES[a], 0) + 1
        icu_utils.append(entry["icu_util"])

    avg_icu = float(np.mean(icu_utils)) if icu_utils else 0

    return {
        "n_predictions": len(recent),
        "avg_icu_utilization": round(avg_icu, 3),
        "icu_drift_alert": avg_icu > 0.85,
        "action_distribution": action_dist,
        "monitoring_window": "last 100 predictions",
        "baseline_avg_wait_hours": BASELINE_AVG_WAIT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/policy/info")
def policy_info():
    if agent is None:
        raise HTTPException(status_code=503, detail="No policy loaded")
    return {
        "policy_path": POLICY_PATH,
        "episode": agent.episode,
        "epsilon": round(agent.epsilon, 4),
        "q_table_states": agent.q_table_size(),
        "n_actions": agent.n_actions,
        "algorithm": "Q-Learning",
    }
