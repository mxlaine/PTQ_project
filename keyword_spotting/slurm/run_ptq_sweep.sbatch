#!/bin/bash
#SBATCH --job-name=ptq-sweep
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-v100-32g
#SBATCH --constraint=skl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-35

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lainem31/PTQ_project/keyword_spotting}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/work/lainem31/PTQ_project/.venv/bin/activate}"

# Advisor-mandated grid: 3 feature configs x 4 hidden sizes x 3 seeds = 36 tasks.
FEATURE_CONFIGS=(16_mels 16_mels_delta_delta 48_mels)
HIDDEN_SIZES=(16 32 48 64)
SEEDS=(0 1 2)

# Source training config: best params from job 17728669 (16_mels_delta_delta SpecAugment sweep).
# Best run: freq_mask=5, time_mask=0, hidden=64 -> 97.38% test acc.
# Fixed: lr=2e-3 wd=0 dropout=0.1 warmup=10 scheduler=cosine epochs=325.
# These are training-time hyperparams; passed through to KeywordGRU constructor so model
# topology (SpecAugment submodules, dropout slots) matches training. At eval they are no-ops.
SPEC_AUGMENT="${SPEC_AUGMENT:-1}"          # 1 => pass --spec-augment to calibrate_ptq.py
FREQ_MASK_PARAM="${FREQ_MASK_PARAM:-5}"
TIME_MASK_PARAM="${TIME_MASK_PARAM:-0}"
DROPOUT="${DROPOUT:-0.1}"

# CHECKPOINT_INDEX must point to a JSON file mapping (feature_config, hidden_size, seed) -> path.
# Format: [{"feature_config": "...", "hidden_size": 64, "seed": 0, "checkpoint": "/abs/path.pt"}, ...]
CHECKPOINT_INDEX="${CHECKPOINT_INDEX:-${PROJECT_ROOT}/results/ptq_checkpoint_index.json}"

CALIB_BATCHES="${CALIB_BATCHES:-16}"
PERCENTILE="${PERCENTILE:-99.99}"

OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/results/ptq_sweep}"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"

organize_logs_on_exit() {
  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi
  RUN_DIR="${PROJECT_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}"
  OUT_FILE="${PROJECT_ROOT}/slurm/ptq-sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  ERR_FILE="${PROJECT_ROOT}/slurm/ptq-sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
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
  echo "Submit with: sbatch slurm/run_ptq_sweep.sbatch" >&2
  exit 1
fi

if [ ! -f "${CHECKPOINT_INDEX}" ]; then
  echo "CHECKPOINT_INDEX not found: ${CHECKPOINT_INDEX}" >&2
  echo "Create it as a JSON list of {feature_config, hidden_size, seed, checkpoint} entries." >&2
  exit 1
fi

N_FEATURES=${#FEATURE_CONFIGS[@]}
N_HIDDENS=${#HIDDEN_SIZES[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$(( N_FEATURES * N_HIDDENS * N_SEEDS ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$(( IDX % N_SEEDS ))
HID_IDX=$(( (IDX / N_SEEDS) % N_HIDDENS ))
FEAT_IDX=$(( IDX / (N_SEEDS * N_HIDDENS) ))

FEATURE_CONFIG="${FEATURE_CONFIGS[FEAT_IDX]}"
HIDDEN_SIZE="${HIDDEN_SIZES[HID_IDX]}"
SEED="${SEEDS[SEED_IDX]}"

CHECKPOINT=$(python -c "
import json
with open('${CHECKPOINT_INDEX}') as f:
    entries = json.load(f)
for e in entries:
    if e['feature_config']=='${FEATURE_CONFIG}' and int(e['hidden_size'])==${HIDDEN_SIZE} and int(e['seed'])==${SEED}:
        print(e['checkpoint']); break
else:
    print('NOT_FOUND')
")

if [ "${CHECKPOINT}" = "NOT_FOUND" ] || [ -z "${CHECKPOINT}" ]; then
  echo "No checkpoint entry for ${FEATURE_CONFIG} h=${HIDDEN_SIZE} seed=${SEED} in ${CHECKPOINT_INDEX}" | tee /dev/stderr
  exit 0
fi

OUT_DIR="${OUT_BASE}/${SLURM_ARRAY_JOB_ID:-local}/${FEATURE_CONFIG}_h${HIDDEN_SIZE}_s${SEED}"
mkdir -p "${OUT_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID}: feature=${FEATURE_CONFIG} hidden=${HIDDEN_SIZE} seed=${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Out dir: ${OUT_DIR}"

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

# Append one row to a shared CSV (per-job aggregation done offline).
RESULTS_CSV="${OUT_BASE}/${SLURM_ARRAY_JOB_ID:-local}/ptq_results.csv"
if [ ! -f "${RESULTS_CSV}" ]; then
  echo "feature_config,hidden_size,seed,fp32_acc,int8_acc,delta_pp,params,macs" > "${RESULTS_CSV}"
fi
python -c "
import json
with open('${OUT_DIR}/summary.json') as f:
    s = json.load(f)
print(f\"{s['feature_config']},{s['hidden_size']},{s['seed']},{s['fp32_test_acc']:.4f},{s['int8_test_acc']:.4f},{s['delta_pp']:.4f},{s['params']},{s['macs']}\")
" >> "${RESULTS_CSV}"
