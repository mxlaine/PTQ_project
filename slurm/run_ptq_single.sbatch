#!/bin/bash
#SBATCH --job-name=ptq-single
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"

# Stage 1: validate the PTQ pipeline on the strongest available FP32 checkpoint.
# Source: job 17728669 task 19 (best SpecAugment run, FP32 test acc = 97.38%).
# Pass criterion: INT8 test acc within 2 pp of FP32 baseline.
FEATURE_CONFIG="${FEATURE_CONFIG:-16_mels_delta_delta}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/17728669/best_keyword_gru_16_mels_delta_delta_17728669_19.pt}"

# Training-time config that produced this checkpoint (job 17728669 best params).
SPEC_AUGMENT="${SPEC_AUGMENT:-1}"
FREQ_MASK_PARAM="${FREQ_MASK_PARAM:-5}"
TIME_MASK_PARAM="${TIME_MASK_PARAM:-0}"
DROPOUT="${DROPOUT:-0.1}"

CALIB_BATCHES="${CALIB_BATCHES:-16}"
PERCENTILE="${PERCENTILE:-99.99}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/ptq_stage1/${SLURM_JOB_ID:-local}}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    return
  fi
  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/ptq-single-${SLURM_JOB_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/ptq-single-${SLURM_JOB_ID}.err"
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

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Stage 1 PTQ validation"
echo "  feature_config = ${FEATURE_CONFIG}"
echo "  hidden_size    = ${HIDDEN_SIZE}"
echo "  seed           = ${SEED}"
echo "  checkpoint     = ${CHECKPOINT}"
echo "  spec_augment   = ${SPEC_AUGMENT} (freq=${FREQ_MASK_PARAM}, time=${TIME_MASK_PARAM})"
echo "  dropout        = ${DROPOUT}"
echo "  calib_batches  = ${CALIB_BATCHES}"
echo "  percentile     = ${PERCENTILE}"
echo "  out_dir        = ${OUT_DIR}"

SPEC_AUGMENT_FLAG=""
if [ "${SPEC_AUGMENT}" = "1" ]; then
  SPEC_AUGMENT_FLAG="--spec-augment"
fi

python -u src/calibrate_ptq.py \
  --feature-config "${FEATURE_CONFIG}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seed "${SEED}" \
  --checkpoint "${CHECKPOINT}" \
  --calib-batches "${CALIB_BATCHES}" \
  --percentile "${PERCENTILE}" \
  --dropout "${DROPOUT}" \
  --freq-mask-param "${FREQ_MASK_PARAM}" \
  --time-mask-param "${TIME_MASK_PARAM}" \
  ${SPEC_AUGMENT_FLAG} \
  --out-dir "${OUT_DIR}"

echo "--- Stage 1 result ---"
python -c "
import json
with open('${OUT_DIR}/summary.json') as f:
    s = json.load(f)
print(f\"FP32 test acc : {s['fp32_test_acc']:.4f}%\")
print(f\"INT8 test acc : {s['int8_test_acc']:.4f}%\")
print(f\"Delta (pp)    : {s['delta_pp']:.4f}\")
print(f\"Pass (<= 2pp) : {'YES' if s['delta_pp'] <= 2.0 else 'NO'}\")
"
