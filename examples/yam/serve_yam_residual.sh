#!/bin/bash
# =============================================================================
# YAM bundled residual policy server
# =============================================================================
# Loads pi0.5 (via openpi) + the jaxrl2 residual actor in one process.
# Exposes:
#   method="infer"               base + residual composed into absolute joints
#   method="update_actor_params" hot-swap residual params (used during training)
#
# Wire-compatible with examples/yam/eval.py (the deploy-time client).
# =============================================================================

set -e

source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

device_id=${device_id:-0}
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Server gets the lion's share of VRAM (pi0.5 LoRA is heavy).
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVER_MEM_FRACTION:-0.7}

PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

BASE_CONFIG=${BASE_CONFIG:-pi05_yam_simpletest_lora}
BASE_DIR=${BASE_DIR:?"Set BASE_DIR to your pi0.5 YAM checkpoint directory"}
BASE_REPO_ID=${BASE_REPO_ID:-}
RESIDUAL_CHECKPOINT=${RESIDUAL_CHECKPOINT:-}
RESIDUAL_ARGS=${RESIDUAL_ARGS:?"Set RESIDUAL_ARGS to the trainer's launch_args.json"}
DEFAULT_PROMPT=${DEFAULT_PROMPT:-"pick up the lego block and place into the box"}

ARGS=(
    --base-config "$BASE_CONFIG"
    --base-dir "$BASE_DIR"
    --residual-args "$RESIDUAL_ARGS"
    --port "$PORT"
    --host "$HOST"
    --default-prompt "$DEFAULT_PROMPT"
    --mem-fraction "$XLA_PYTHON_CLIENT_MEM_FRACTION"
)
if [[ -n "$BASE_REPO_ID" ]]; then
    ARGS+=( --base-repo-id "$BASE_REPO_ID" )
fi
if [[ -n "$RESIDUAL_CHECKPOINT" ]]; then
    ARGS+=( --residual-checkpoint "$RESIDUAL_CHECKPOINT" )
fi

# `openpi.examples` is not a package (no __init__.py under openpi/examples/), so
# invoke the script by path. PYTHONPATH points at the openpi repo root so
# `from openpi.*` imports resolve via the editable install.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
python -u "${SCRIPT_DIR}/serve_yam_residual.py" "${ARGS[@]}"
