#!/bin/bash
#SBATCH --job-name=keyword-gru-lr-wd
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

# LR x weight_decay x hidden_size sweep. Other hyperparams fixed at best known config.
# Total = 4 * 4 * 4 = 64 array tasks (matches --array=0-63 above).
LR_VALUES=(5e-4 1e-3 2e-3 5e-3)
WD_VALUES=(0 1e-5 1e-4 1e-3)
HIDDEN_SIZES=(16 32 48 64)
SEED="${SEED:-0}"

EPOCHS="${EPOCHS:-325}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-10}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
FEATURE_CONFIG="${FEATURE_CONFIG:-16_mels_delta_delta}"
FREQ_MASK="${FREQ_MASK:-5}"
TIME_MASK="${TIME_MASK:-6}"
DROPOUT="${DROPOUT:-0.1}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-lr-wd-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-lr-wd-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

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
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_lr_wd_sweep.sbatch" >&2
  exit 1
fi

TOTAL_LRS=${#LR_VALUES[@]}
TOTAL_WDS=${#WD_VALUES[@]}
TOTAL_HIDDEN=${#HIDDEN_SIZES[@]}
TOTAL_TASKS=$(( TOTAL_LRS * TOTAL_WDS * TOTAL_HIDDEN ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Map array task to (lr, wd, hidden_size) — slowest-changing first.
IDX=${SLURM_ARRAY_TASK_ID}
HID_IDX=$(( IDX % TOTAL_HIDDEN ))
IDX=$(( IDX / TOTAL_HIDDEN ))
WD_IDX=$(( IDX % TOTAL_WDS ))
LR_IDX=$(( IDX / TOTAL_WDS ))

LR="${LR_VALUES[LR_IDX]}"
WEIGHT_DECAY="${WD_VALUES[WD_IDX]}"
HIDDEN_SIZE="${HIDDEN_SIZES[HID_IDX]}"

echo "Running task ${SLURM_ARRAY_TASK_ID}: lr=${LR} weight_decay=${WEIGHT_DECAY} hidden_size=${HIDDEN_SIZE} seed=${SEED}"
echo "Fixed config: feature_config=${FEATURE_CONFIG} dropout=${DROPOUT} freq_mask=${FREQ_MASK} time_mask=${TIME_MASK} warmup=${LR_WARMUP_EPOCHS}"

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
