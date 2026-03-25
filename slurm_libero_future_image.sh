#!/bin/bash
#SBATCH --job-name=pi05_libero_waypoint
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --constraint=RTX_PRO_6000
#SBATCH --output=/data/user_data/sreyasv/libero_logs/logs/pi05_libero_waypoint_%x_%j.out
#SBATCH --error=/data/user_data/sreyasv/libero_logs/logs/pi05_libero_waypoint_%x_%j.err

# =============================================================================
# Libero: Pi-0.5 with waypoint-image conditioning (4 image tokens)
# Uses LIBERO LeRobot dataset with waypoint_image
# =============================================================================
#
# Usage:
#   sbatch slurm_libero_future_image.sh
#
# =============================================================================

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/data/hf_cache/datasets/LIBERO/ilp/ilp-libero/.cache

cd /data/user_data/sreyasv/Repos/openpi

uv run scripts/train.py pi05_libero \
    --data.repo-id Soumojit048/ilp-libero \
    --exp-name=pi05_libero_waypoint \
    --batch-size=32 \
    --weight-loader.params-path /data/hf_cache/pi-models/openpi/openpi-assets/checkpoints/pi05_base/params
