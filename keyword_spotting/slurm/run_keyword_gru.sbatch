#!/bin/bash
#SBATCH --job-name=keyword-gru
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-47

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"
FEATURE_CONFIGS=(
  "16_mels_delta_delta"
  "24_mels_delta"
  "40_mels"
  "48_mels"
)

# Sweep values for SpecAugment mask sizes. Adjust these arrays to change sweep.
FREQ_MASKS=(0 4 8 12)
TIME_MASKS=(10 30 50)
EPOCHS="${EPOCHS:-400}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

  mkdir -p "${RUN_DIR}"
  [ -f "${OUT_FILE}" ] && mv -f "${OUT_FILE}" "${RUN_DIR}/"
  [ -f "${ERR_FILE}" ] && mv -f "${ERR_FILE}" "${RUN_DIR}/"
}

trap organize_logs_on_exit EXIT

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi

source "${VENV_ACTIVATE}"

export MPLBACKEND=Agg

# Export explicit plot grouping identifiers for `src/main.py` to use
# `PLOT_JOB` will be the Slurm array job id (or job id), and `PLOT_TASK` the array task id.
PLOT_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PLOT_TASK="${SLURM_ARRAY_TASK_ID:-0}"
export PLOT_JOB PLOT_TASK

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_keyword_gru.sbatch" >&2
  exit 1
fi

TOTAL_FEATURES=${#FEATURE_CONFIGS[@]}
TOTAL_FREQS=${#FREQ_MASKS[@]}
TOTAL_TIMES=${#TIME_MASKS[@]}
TOTAL_TASKS=$(( TOTAL_FEATURES * TOTAL_FREQS * TOTAL_TIMES ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Map array task to (feature, freq_mask, time_mask)
IDX=${SLURM_ARRAY_TASK_ID}
FEATURE_IDX=$(( IDX / ( TOTAL_FREQS * TOTAL_TIMES ) ))
REM=$(( IDX % ( TOTAL_FREQS * TOTAL_TIMES ) ))
FREQ_IDX=$(( REM / TOTAL_TIMES ))
TIME_IDX=$(( REM % TOTAL_TIMES ))

FEATURE_CONFIG="${FEATURE_CONFIGS[FEATURE_IDX]}"
FREQ_MASK="${FREQ_MASKS[FREQ_IDX]}"
TIME_MASK="${TIME_MASKS[TIME_IDX]}"

echo "Running feature config: ${FEATURE_CONFIG} (task ${SLURM_ARRAY_TASK_ID})"
echo "SpecAugment freq_mask=${FREQ_MASK} time_mask=${TIME_MASK}"

python -u src/main.py \
  --feature-config "${FEATURE_CONFIG}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --batch-size-train "${BATCH_SIZE_TRAIN}" \
  --batch-size-eval "${BATCH_SIZE_EVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-layers "${NUM_LAYERS}" \
  --spec-augment \
  --freq-mask-param "${FREQ_MASK}" \
  --time-mask-param "${TIME_MASK}" \
  "$@"
