#!/bin/bash
# W&B Sweep 脚本
# 用法: bash scripts/run_sweep.sh

set -e

echo "Creating W&B sweep..."
SWEEP_ID=$(wandb sweep configs/sweeps/kl_lambda.yaml 2>&1 | grep -oP "sweep ID: \K\S+")
echo "Sweep ID: $SWEEP_ID"

echo "Starting sweep agent..."
wandb agent $SWEEP_ID
