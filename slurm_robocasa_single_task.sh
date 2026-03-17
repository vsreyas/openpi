#!/bin/bash
#SBATCH --job-name=pi05_robocasa_single_task
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --constraint=L40S
#SBATCH --output=/data/user_data/sreyasv/robocasa_logs/logs/pi05_single_task_%j.out
#SBATCH --error=/data/user_data/sreyasv/robocasa_logs/logs/pi05_single_task_%j.err

# =============================================================================
# RoboCasa: Pi-0.5 single-task fine-tuning on PickPlaceCounterToCabinet
# LoRA on VLM, full fine-tune on action head
# Scene restricted to layout_id=1, style_id=1
#
# Usage:
#   sbatch slurm_robocasa_single_task.sh              # default 40 demos
#   NUM_DEMOS=10 sbatch slurm_robocasa_single_task.sh # override demo count
# =============================================================================

NUM_DEMOS=${NUM_DEMOS:-3}

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /home/sreyasv/Projects/dsrl_pi0/openpi
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0
echo "Starting single-task training: PickPlaceCounterToCabinet, layout=1 style=1, num_demos=${NUM_DEMOS}"

conda run -n dsrl_pi0 python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name=pi05_pickplace_cab_L1S1_${NUM_DEMOS}demos \
    --batch-size 32 \
    --data.num-demos ${NUM_DEMOS} \
    --overwrite