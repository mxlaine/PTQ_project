#!/bin/bash
#SBATCH --job-name=keyword-gru-sa
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-63

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"

# SpecAugment sweep: freq_mask x time_mask x hidden_size.
# Best config from job 17719851: lr=2e-3, wd=0, cosine, 16_mels_delta_delta.
# Total = 4 * 4 * 4 = 64 array tasks (matches --array=0-63 above).
FREQ_VALUES=(0 5 10 15)
TIME_VALUES=(0 6 8 12)
HIDDEN_SIZES=(16 32 48 64)
SEED="${SEED:-0}"

LR=2e-3
WEIGHT_DECAY=0
EPOCHS="${EPOCHS:-325}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-10}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
FEATURE_CONFIG="${FEATURE_CONFIG:-16_mels_delta_delta}"
DROPOUT="${DROPOUT:-0.1}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-sa-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-sa-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

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

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_specaugment_sweep.sbatch" >&2
  exit 1
fi

TOTAL_FREQ=${#FREQ_VALUES[@]}
TOTAL_TIME=${#TIME_VALUES[@]}
TOTAL_HIDDEN=${#HIDDEN_SIZES[@]}
TOTAL_TASKS=$(( TOTAL_FREQ * TOTAL_TIME * TOTAL_HIDDEN ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Map task -> (freq_mask, time_mask, hidden_size) — freq slowest-changing.
IDX=${SLURM_ARRAY_TASK_ID}
HID_IDX=$(( IDX % TOTAL_HIDDEN ))
IDX=$(( IDX / TOTAL_HIDDEN ))
TIME_IDX=$(( IDX % TOTAL_TIME ))
FREQ_IDX=$(( IDX / TOTAL_TIME ))

FREQ_MASK="${FREQ_VALUES[FREQ_IDX]}"
TIME_MASK="${TIME_VALUES[TIME_IDX]}"
HIDDEN_SIZE="${HIDDEN_SIZES[HID_IDX]}"

echo "Running task ${SLURM_ARRAY_TASK_ID}: freq_mask=${FREQ_MASK} time_mask=${TIME_MASK} hidden_size=${HIDDEN_SIZE} seed=${SEED}"
echo "Fixed config: lr=${LR} wd=${WEIGHT_DECAY} feature=${FEATURE_CONFIG} dropout=${DROPOUT} warmup=${LR_WARMUP_EPOCHS}"

python -u src/main.py \
  --feature-config "${FEATURE_CONFIG}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --batch-size-train "${BATCH_SIZE_TRAIN}" \
  --batch-size-eval "${BATCH_SIZE_EVAL}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seed "${SEED}" \
  --use-new-gru \
  --spec-augment \
  --freq-mask-param "${FREQ_MASK}" \
  --time-mask-param "${TIME_MASK}" \
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS}" \
  --dropout "${DROPOUT}" \
  "$@"
