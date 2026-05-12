#!/bin/bash
#SBATCH --job-name=keyword-gru-best5-warmrestart
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-1


set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"

# Warm-restart retry on rank-1 best5 config (freq=5, time=6, dropout=0.1, warmup=10).
# Task 0: cosine baseline reproduce — sanity reference for seed/sampler variance.
# Task 1: cosine-warm-restarts T_0=200, T_mult=1, ceiling_decay=0.5
#         → one restart at epoch ~210 (after 10-epoch warmup), peak halved to 5e-4 in cycle 2.
SCHEDULERS=("cosine" "cosine-warm-restarts")
RESTART_CEILS=(1.0 0.5)

# Shared rank-1 best5 hyperparameters
FREQ_MASK=5
TIME_MASK=6
LR_WARMUP=10
DROPOUT=0.1

EPOCHS="${EPOCHS:-325}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-64}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1024}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
FEATURE_CONFIG="${FEATURE_CONFIG:-16_mels_delta_delta}"
SEED="${SEED:-0}"
LR_T0="${LR_T0:-200}"
LR_T_MULT="${LR_T_MULT:-1}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/keyword-gru-best5-warmrestart-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/keyword-gru-best5-warmrestart-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

  mkdir -p "${RUN_DIR}"
  [ -f "${OUT_FILE}" ] && mv -f "${OUT_FILE}" "${RUN_DIR}/"
  [ -f "${ERR_FILE}" ] && mv -f "${ERR_FILE}" "${RUN_DIR}/"
}

trap organize_logs_on_exit EXIT

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi

module load scicomp-python-env/2025.2
source "${VENV_ACTIVATE}"

export MPLBACKEND=Agg

PLOT_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PLOT_TASK="${SLURM_ARRAY_TASK_ID:-0}"
export PLOT_JOB PLOT_TASK

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script is configured as a Slurm array job. Submit with: sbatch slurm/run_best5_warmrestart.sbatch" >&2
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "2" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..1" >&2
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}
SCHEDULER="${SCHEDULERS[IDX]}"
RESTART_CEIL="${RESTART_CEILS[IDX]}"

echo "Running task ${IDX}: scheduler=${SCHEDULER} t0=${LR_T0} t_mult=${LR_T_MULT} ceiling_decay=${RESTART_CEIL} freq=${FREQ_MASK} time=${TIME_MASK} warmup=${LR_WARMUP} dropout=${DROPOUT} use_new_gru=True"

python -u src/main.py \
  --feature-config "${FEATURE_CONFIG}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --batch-size-train "${BATCH_SIZE_TRAIN}" \
  --batch-size-eval "${BATCH_SIZE_EVAL}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seed "${SEED}" \
  --spec-augment \
  --freq-mask-param "${FREQ_MASK}" \
  --time-mask-param "${TIME_MASK}" \
  --lr-warmup-epochs "${LR_WARMUP}" \
  --dropout "${DROPOUT}" \
  --use-new-gru \
  --lr-scheduler "${SCHEDULER}" \
  --lr-t0 "${LR_T0}" \
  --lr-t-mult "${LR_T_MULT}" \
  --lr-restart-ceiling-decay "${RESTART_CEIL}" \
  "$@"
