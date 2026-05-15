#!/bin/bash
#SBATCH --job-name=yam-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log

# ============================================================================
# YAM Pi0.5 LoRA training on HPC (JAX, single node, multi-GPU)
#
# Usage:
#   sbatch examples/yam/train_hpc.sbatch
#
# Before first run, customize the paths below and stage data on NFS.
# See examples/yam/TRAINING.md for full setup instructions.
# ============================================================================

# --- Customize these for your HPC ---
OPENPI_DIR="/nfs/path/openpi"
CACHE_DIR="/nfs/path/cache"
CONFIG_NAME="pi05_yam_simpletest_lora"
REPO_ID=""  # Leave empty to use config default, or set e.g. "local/my_new_dataset"

# --- Environment variables ---
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_LEROBOT_HOME="${CACHE_DIR}/huggingface/lerobot"
export OPENPI_DATA_HOME="${CACHE_DIR}/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0   
# --- Build repo-id flags ---
NORM_REPO_FLAG=""
TRAIN_REPO_FLAG=""
if [ -n "${REPO_ID}" ]; then
    NORM_REPO_FLAG="--repo-id ${REPO_ID}"
    TRAIN_REPO_FLAG="--data.repo-id ${REPO_ID}"
fi

cd "${OPENPI_DIR}"

# --- Compute norm stats ---
uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" ${NORM_REPO_FLAG}

# --- Training ---
uv run scripts/train.py "${CONFIG_NAME}" \
    ${TRAIN_REPO_FLAG} \
    --exp-name=yam_lora_v1 \
    --wandb-enabled \
    --overwrite
