#!/bin/bash
#SBATCH --job-name=keyword-gru-dropout
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --array=0-24

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"

# Top-5 SpecAugment configs (kept for historical continuity; original sweep was on
# 24_mels_delta but that feature is retired — we now run dropout sweep on 16_mels_delta_delta).
# Rank 1: freq=5,  time=6,  warmup=10
# Rank 2: freq=6,  time=5,  warmup=0
# Rank 3: freq=12, time=0,  warmup=0
# Rank 4: freq=6,  time=0,  warmup=0
# Rank 5: freq=6,  time=25, warmup=0
FREQ_MASKS=(5 6 12 6 6)
TIME_MASKS=(6 5 0 0 25)
LR_WARMUPS=(10 0 0 0 0)

DROPOUT_VALUES=(0.1 0.2 0.3 0.4 0.5)

EPOCHS="${EPOCHS:-325}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
FEATURE_CONFIG="${FEATURE_CONFIG:-16_mels_delta_delta}"
SEED="${SEED:-0}"

TOTAL_CFGS=${#FREQ_MASKS[@]}
TOTAL_DROPOUTS=${#DROPOUT_VALUES[@]}
TOTAL_TASKS=$(( TOTAL_CFGS * TOTAL_DROPOUTS ))

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-dropout-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-dropout-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

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
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_dropout_sweep.sbatch" >&2
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

# Map task ID to (config, dropout): configs are outer, dropouts are inner.
IDX=${SLURM_ARRAY_TASK_ID}
CFG_IDX=$(( IDX / TOTAL_DROPOUTS ))
DROPOUT_IDX=$(( IDX % TOTAL_DROPOUTS ))

FREQ_MASK="${FREQ_MASKS[CFG_IDX]}"
TIME_MASK="${TIME_MASKS[CFG_IDX]}"
LR_WARMUP="${LR_WARMUPS[CFG_IDX]}"
DROPOUT="${DROPOUT_VALUES[DROPOUT_IDX]}"

echo "Running task ${SLURM_ARRAY_TASK_ID}: cfg=${CFG_IDX} freq=${FREQ_MASK} time=${TIME_MASK} warmup=${LR_WARMUP} dropout=${DROPOUT}"

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
  --lr-warmup-epochs "${LR_WARMUP}" \
  --dropout "${DROPOUT}" \
  "$@"
