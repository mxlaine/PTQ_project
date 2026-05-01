#!/bin/bash
#SBATCH --job-name=keyword-gru-size
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-5

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"

# Sweep hidden_size x feature_config to find the smallest model that holds ~97%.
HIDDEN_SIZES=(32 48 64)
FEATURE_CONFIGS_LIST=("24_mels_delta" "16_mels_delta_delta")

EPOCHS="${EPOCHS:-400}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"

# Pick SpecAugment values from the wider sweep result; defaults match the mid-range.
FREQ_MASK="${FREQ_MASK:-16}"
TIME_MASK="${TIME_MASK:-25}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-size-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-size-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

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

PLOT_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PLOT_TASK="${SLURM_ARRAY_TASK_ID:-0}"
export PLOT_JOB PLOT_TASK

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_keyword_gru_size_sweep.sbatch" >&2
  exit 1
fi

TOTAL_HIDDEN=${#HIDDEN_SIZES[@]}
TOTAL_FCFG=${#FEATURE_CONFIGS_LIST[@]}
TOTAL_TASKS=$(( TOTAL_HIDDEN * TOTAL_FCFG ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Map array task to (hidden_size, feature_config).
IDX=${SLURM_ARRAY_TASK_ID}
FCFG_IDX=$(( IDX / TOTAL_HIDDEN ))
HID_IDX=$(( IDX % TOTAL_HIDDEN ))

HIDDEN_SIZE="${HIDDEN_SIZES[HID_IDX]}"
FEATURE_CONFIG="${FEATURE_CONFIGS_LIST[FCFG_IDX]}"

echo "Running task ${SLURM_ARRAY_TASK_ID}: hidden_size=${HIDDEN_SIZE} feature_config=${FEATURE_CONFIG}"
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
