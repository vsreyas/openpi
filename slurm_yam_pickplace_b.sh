#!/bin/bash
#SBATCH --job-name=pi05_yam_pickplace_b
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --exclude=babel-p5-28,babel-p5-20,babel-o5-24
#SBATCH --output=/data/user_data/sreyasv/yam_logs/logs/pi05_yam_pickplace_b_%j.out
#SBATCH --error=/data/user_data/sreyasv/yam_logs/logs/pi05_yam_pickplace_b_%j.err

# =============================================================================
# YAM: Pi-0.5 LoRA fine-tune on pick-place, traj id >= 014512.
# Filter resolves to 8 episodes / 8923 frames. Multi-traj BC corpus.
#
# Defaults to --resume so SLURM requeues pick up the latest checkpoint. First
# run starts from scratch (no checkpoint -> no-op resume).
#
# Usage:
#   sbatch slurm_yam_pickplace_b.sh
#   RUN_TAG=v2 sbatch slurm_yam_pickplace_b.sh
# =============================================================================

RUN_TAG=${RUN_TAG:-v1}
BATCH_SIZE=${BATCH_SIZE:-32}

export OPENPI_DATA_HOME=/data/group_data/rl/sreyasv/base_dump
export HF_LEROBOT_HOME=/data/group_data/maxlab/common_datasets/sreyasv/yam_lerobot/lerobot_home
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1

CKPT_ROOT=/data/group_data/maxlab/common_datasets/sreyasv/vla_project/experiments/openpi_runs/checkpoints

mkdir -p /data/user_data/sreyasv/yam_logs/logs
mkdir -p "$CKPT_ROOT"

cd /home/sreyasv/Projects/dsrl_pi0/openpi
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0
echo "Starting pi05_yam_pickplace_b_lora run_tag=${RUN_TAG} batch_size=${BATCH_SIZE}"

python scripts/train.py pi05_yam_pickplace_b_lora \
    --exp-name=pi05_yam_pickplace_b_${RUN_TAG} \
    --batch-size ${BATCH_SIZE} \
    --num-workers 2 \
    --checkpoint-base-dir="$CKPT_ROOT" \
    --resume
