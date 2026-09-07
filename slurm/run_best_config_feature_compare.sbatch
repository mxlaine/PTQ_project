#!/bin/bash
#SBATCH --job-name=kws-feat-compare
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-23

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"

# Delta-ablation comparison: same best config as job 17792221
# (16_mels_delta_delta, lr=2e-3, wd=0, dropout=0.1, warmup=10, freq_mask=5,
# time_mask=0, 97.38% h=64) applied to 16_mels and 48_mels (no deltas).
# 2 feature configs x 4 hidden sizes x 3 seeds = 24 tasks.
FEATURE_CONFIGS=(16_mels 48_mels)
HIDDEN_SIZES=(16 32 48 64)
SEEDS=(0 1 2)

EPOCHS=325
LR=2e-3
WEIGHT_DECAY=0
DROPOUT=0.1
LR_WARMUP_EPOCHS=10
FREQ_MASK=5
TIME_MASK=0
BATCH_SIZE_TRAIN=64
BATCH_SIZE_EVAL=1024

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi
  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/kws-feat-compare-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/kws-feat-compare-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
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
  echo "Submit with: sbatch slurm/run_best_config_feature_compare.sbatch" >&2
  exit 1
fi

N_SEEDS=${#SEEDS[@]}
N_HIDDEN=${#HIDDEN_SIZES[@]}
N_FEATURE=${#FEATURE_CONFIGS[@]}
TOTAL_TASKS=$(( N_FEATURE * N_HIDDEN * N_SEEDS ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Task mapping: seed innermost, then hidden, then feature
# (matches build_ptq_index.py feature-compare sweep order).
IDX=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$(( IDX % N_SEEDS ))
HID_IDX=$(( (IDX / N_SEEDS) % N_HIDDEN ))
FEAT_IDX=$(( IDX / (N_SEEDS * N_HIDDEN) ))

FEATURE_CONFIG="${FEATURE_CONFIGS[FEAT_IDX]}"
HIDDEN_SIZE="${HIDDEN_SIZES[HID_IDX]}"
SEED="${SEEDS[SEED_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: feature=${FEATURE_CONFIG} hidden=${HIDDEN_SIZE} seed=${SEED}"
echo "lr=${LR} wd=${WEIGHT_DECAY} dropout=${DROPOUT} warmup=${LR_WARMUP_EPOCHS} freq_mask=${FREQ_MASK} time_mask=${TIME_MASK}"

PLOT_JOB="${SLURM_ARRAY_JOB_ID}"
PLOT_TASK="${SLURM_ARRAY_TASK_ID}"
export PLOT_JOB PLOT_TASK

python -u src/main.py \
  --feature-config "${FEATURE_CONFIG}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --dropout "${DROPOUT}" \
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS}" \
  --batch-size-train "${BATCH_SIZE_TRAIN}" \
  --batch-size-eval "${BATCH_SIZE_EVAL}" \
  --spec-augment \
  --freq-mask-param "${FREQ_MASK}" \
  --time-mask-param "${TIME_MASK}" \
  --use-new-gru
