#!/bin/bash
# Deploy IMC-FL project code to Cayuga cluster.
# Run from local machine: bash scripts/cayuga/deploy.sh

set -e

REMOTE="<USER>@<CLUSTER>"
SSH_KEY="$HOME/.ssh/cayuga_key2"
REMOTE_DIR="<PROJECT_ROOT>"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Deploying from: $LOCAL_DIR"
echo "Deploying to:   $REMOTE:$REMOTE_DIR"
echo ""

# Create remote directory
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p $REMOTE_DIR/output/batch $REMOTE_DIR/logs"

# Sync project code (exclude data, output, venv, cache)
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude 'output/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$LOCAL_DIR/" \
    "$REMOTE:$REMOTE_DIR/"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps on Cayuga:"
echo "  ssh -i $SSH_KEY $REMOTE"
echo "  cd $REMOTE_DIR"
echo ""
echo "  # First time: setup environment"
echo "  bash scripts/cayuga/setup_env.sh"
echo ""
echo "  # Submit batch job"
echo "  sbatch scripts/cayuga/submit_batch.sh"
echo ""
echo "  # Monitor"
echo "  squeue -u ole2001"
echo "  tail -f logs/batch_*.out"
