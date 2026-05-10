# Final Report – Hospital Resource Allocation using Reinforcement Learning & MLOps

**Course:** 24AM6AEMLO – Machine Learning Operations  
**SDG:** SDG 3 – Good Health and Well-Being  
**Date:** May 2026

---

## 1. Problem Statement

Hospitals must allocate scarce resources — doctors, nurses, ICU beds, ventilators — across departments in real time. Static fixed-timer allocation ignores fluctuating demand, leading to preventable delays and deaths. We model this as a Reinforcement Learning problem where an agent learns a dynamic allocation policy to minimize patient wait times.

---

## 2. SDG Connection

**SDG 3 – Good Health and Well-Being** targets universal health coverage and reduced preventable mortality. Our RL policy reduces average patient wait times by ~11%, directly:
- Improving Emergency response → fewer critical deteriorations
- Optimizing ICU utilization → better survival outcomes
- Enabling scalable, data-driven hospital management

---

## 3. Simulator

`sim/hospital_env.py` models a hospital with 4 departments:

| Department | Arrival Rate (λ/hr) | Severity Weight |
|------------|---------------------|-----------------|
| Emergency  | 3.0                 | 1.5             |
| ICU        | 1.0                 | 2.0 (highest)   |
| General Ward| 2.0                | 1.0             |
| OT         | 0.5                 | 1.2             |

Resources: 20 doctors, 40 nurses, 10 ICU beds, 8 ventilators, 50 general beds.  
Each timestep = 1 hour. Episodes = 100 timesteps (100-hour window).

---

## 4. RL Design

**Algorithm:** Q-Learning (tabular, off-policy)  
**Reason:** State space is naturally discrete (~300 states). Tabular Q-learning converges provably and is interpretable — no need for a neural network.

**State:** Queue lengths per department + doctor/nurse allocation summary + ICU/vent availability → discretized to 6-tuple.

**Actions:** 9 discrete allocation strategies (balanced, boost-emergency, boost-ICU, etc.)

**Reward:** `−weighted_wait_time + 0.5·(ICU_ok) − 2·(overflow_depts)`

**Exploration:** ε-greedy, ε: 1.0 → 0.05 with decay 0.97/episode

---

## 5. MLOps Implementation

| Component | Tool/Method | Status |
|-----------|-------------|--------|
| Experiment config | YAML configs | ✅ |
| Experiment logging | CSV + JSON per run | ✅ |
| MLflow tracking | mlflow (optional) | ✅ |
| Policy versioning | policy_v1.pkl, policy_v2.pkl, policy_best.pkl | ✅ |
| Git branching | main/dev/feature | ✅ |
| Git tags | exp-qlearning-1, exp-qlearning-2 | ✅ |
| CI/CD | GitHub Actions (lint → train → evaluate → docker) | ✅ |
| Containerization | Docker + docker-compose | ✅ |
| REST API | FastAPI /allocate endpoint | ✅ |
| Monitoring plan | Drift detection + prediction logs | ✅ |
| Reproducibility | `python train.py --config configs/qlearning_v1.yaml` | ✅ |
| Unit tests | 12 pytest tests, all passing | ✅ |

---

## 6. Results

### Baseline vs RL Comparison (50 evaluation episodes)

| Metric | Fixed-Timer | RL-Policy | Improvement |
|--------|-------------|-----------|-------------|
| Avg Wait Time (hrs) | 5.84 | ~5.2 | **−11%** |
| Avg Reward | −6.95 | ~−6.60 | **+5%** |
| ICU Util | 6.6% | 7.5% | managed |
| Queue Overflows | higher | lower | ✓ |

### Key Observations

- **When RL performs better:** High-load periods (Emergency surge, ICU pressure) — RL correctly shifts resources by choosing actions 5 (Emergency+ICU) or 7 (Surge mode).
- **When RL behaves unexpectedly:** In very low-load episodes, RL sometimes over-allocates to Emergency (legacy from high-reward training scenarios), causing minor ICU under-staffing.
- **Sensitivity to traffic pattern:** When Emergency arrival λ increases from 3 → 5, RL maintains performance by switching to surge mode. Fixed-timer degrades linearly.

---

## 7. Monitoring Plan (Deployment)

For real-world deployment in a hospital, we would track:
- **Average wait-time** per department — rolling 6h window, alert if > baseline + 20%
- **Maximum queue length** — alert if any dept > 15 patients
- **ICU utilization** — alert if > 85% for 2+ consecutive hours
- **Safety rule** — zero-doctor alert if any dept has 0 doctors for > 2 timesteps
- **Action drift** — retrain trigger if action distribution shifts significantly from training

---

## 8. Reproducibility

```bash
# Clone and reproduce exactly:
git clone https://github.com/YOUR_USERNAME/hospital-rl-mlops.git
cd hospital-rl-mlops
pip install -r requirements.txt
python train.py --config configs/qlearning_v1.yaml
# → Produces: experiments/results_qlearning-v1.csv, models/policy_best.pkl
python evaluate.py --policy models/policy_best.pkl --episodes 50
# → Produces: reports/*.png, reports/comparison.json
```

Seed is fixed in each config (`seed: 42`) for full reproducibility.

---

## 9. Limitations

1. Tabular Q-table cannot generalize to unseen states (mass casualty events)
2. Single-agent — real hospitals have decentralized decisions
3. Simulator is a simplification (no shift changes, equipment failures, etc.)

**Future Work:** DQN for continuous state; multi-agent RL; real hospital data integration via HL7/FHIR APIs.

---

## 10. Demo

See `README.md` for step-by-step instructions to run from the Git repo.  
Plots in `reports/`:
- `plot_reward_comparison.png` — RL vs Fixed-Timer reward over episodes
- `plot_wait_comparison.png` — Wait time reduction visualization
- `plot_queue_lengths.png` — Per-department queue dynamics
