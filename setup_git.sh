#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# setup_git.sh – Initialize repo and push to GitHub
# Usage:
#   chmod +x setup_git.sh
#   ./setup_git.sh https://github.com/YOUR_USERNAME/hospital-rl-mlops.git
# ─────────────────────────────────────────────────────────────

set -e

REMOTE="${1:-}"
if [ -z "$REMOTE" ]; then
    echo "Usage: ./setup_git.sh <github-remote-url>"
    echo "Example: ./setup_git.sh https://github.com/yourname/hospital-rl-mlops.git"
    exit 1
fi

echo "=== Initializing Git repo ==="
git init
git config user.email "student@bmsce.ac.in"
git config user.name "Hospital RL Team"

# ── .gitignore ───────────────────────────────
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
mlruns/
*.log
.DS_Store
*.egg-info/
dist/
build/
EOF

# ── Initial commit on main ───────────────────
git add .
git commit -m "feat: initial hospital RL + MLOps project

- Hospital resource allocation RL environment (sim/)
- Q-Learning agent with ε-greedy exploration
- Train/evaluate scripts with CSV + JSON logging
- FastAPI serving endpoint (api.py)
- Dockerfile + docker-compose (API + MLflow)
- GitHub Actions CI/CD pipeline
- Unit tests (12 passing)
- README with full reproducibility instructions

SDG 3 – Good Health and Well-Being"

# ── Tag first experiment ──────────────────────
git tag -a exp-qlearning-1 -m "First Q-learning experiment: α=0.1, γ=0.95, 300 episodes"

# ── Dev branch ───────────────────────────────
git checkout -b dev
git checkout main

echo ""
echo "=== Pushing to GitHub ==="
git remote add origin "$REMOTE"
git push -u origin main
git push origin dev
git push origin --tags

echo ""
echo "✅ Done! Your repo is live at: $REMOTE"
echo ""
echo "Next steps:"
echo "  1. Go to GitHub → Settings → Actions → Enable workflows"
echo "  2. Push a commit to trigger CI/CD:"
echo "     git checkout dev && git push"
echo "  3. Add a second experiment commit:"
echo "     python train.py --config configs/qlearning_v2.yaml"
echo "     git add experiments/ models/"
echo "     git commit -m 'exp: qlearning v2 – slower decay, 500 episodes'"
echo "     git tag -a exp-qlearning-2 -m 'Q-learning v2: α=0.15, γ=0.99, 500 episodes'"
echo "     git push && git push --tags"
