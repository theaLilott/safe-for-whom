#!/bin/bash
# ============================================================
# launch_rq1.sh — Submit one SLURM job per language for RQ1
# ============================================================
#
# Usage:
#   bash launch_rq1.sh <input_csv> <output_dir> [languages...]
#
# Examples:
#   # All 9 languages
#   bash launch_rq1.sh results/rq1_generations_qwen3_14b_new_langs.csv results/rq1_judge/
#
#   # Specific languages only
#   bash launch_rq1.sh results/rq1_generations.csv results/rq1_judge/ english german indonesian
#
# Each language gets its own SLURM job → parallel evaluation.
# Output: results/rq1_judge/judge_english.csv, judge_german.csv, etc.
# ============================================================

set -euo pipefail

BASE_DIR="/NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy"
INPUT_CSV="${1:?Usage: bash launch_rq1.sh <input_csv> <output_dir> [languages...]}"
OUTPUT_DIR="${2:?Usage: bash launch_rq1.sh <input_csv> <output_dir> [languages...]}"
shift 2

# ── Language list: from CLI args, or auto-detect from CSV ──
if [[ $# -gt 0 ]]; then
    LANGUAGES=("$@")
else
    LANGUAGES=($(python3 -c "
import pandas as pd
df = pd.read_csv('${INPUT_CSV}')
for lang in sorted(df['lang'].unique()):
    print(lang)
"))
fi

# ── Evaluation config ──
PROMPT_COL="prompt_english"
RESPONSE_COL="response_english"
LANG_COL="lang"
MODEL="Qwen/Qwen3-32B"
BATCH_SIZE=4
MAX_NEW_TOKENS=2048
N_PASSES=3
TEMPERATURE=0.6

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${BASE_DIR}/logs"

echo "============================================"
echo "RQ1 Judge Pipeline"
echo "============================================"
echo "Input:      ${INPUT_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model:      ${MODEL}"
echo "Passes:     ${N_PASSES} (T=${TEMPERATURE})"
echo "Languages:  ${LANGUAGES[*]}"
echo ""

JOB_IDS=()

for LANG in "${LANGUAGES[@]}"; do
    OUTPUT_CSV="${OUTPUT_DIR}/judge_${LANG}.csv"

    if [[ -f "${OUTPUT_CSV}" ]]; then
        echo "[SKIP] ${LANG}: output already exists at ${OUTPUT_CSV}"
        continue
    fi

    JOB_ID=$(sbatch \
        --job-name="judge_${LANG}" \
        -o "${BASE_DIR}/logs/judge_${LANG}_%j.out" \
        -e "${BASE_DIR}/logs/judge_${LANG}_%j.err" \
        -D "${BASE_DIR}" \
        --partition="h200,h100sxm,h100nvl" \
        --gres="gpu:1" \
        --cpus-per-task=12 \
        --nodes=1 \
        --mem=60G \
        --time="0-08:00" \
        --mail-type="END,FAIL" \
        --mail-user="mkemperm@mpi-sws.org" \
        --export=ALL \
        --wrap="
. ${BASE_DIR}/venv/bin/activate
export HF_HOME='/SWS/llms/nobackup'

srun python3 judge.py \
    --input '${INPUT_CSV}' \
    --output '${OUTPUT_CSV}' \
    --prompt-col '${PROMPT_COL}' \
    --response-col '${RESPONSE_COL}' \
    --lang-col '${LANG_COL}' \
    --lang '${LANG}' \
    --model '${MODEL}' \
    --batch-size ${BATCH_SIZE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --n-passes ${N_PASSES} \
    --temperature ${TEMPERATURE}
" | awk '{print $NF}')

    JOB_IDS+=("${JOB_ID}")
    echo "[SUBMIT] ${LANG}: job ${JOB_ID} -> ${OUTPUT_CSV}"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls ${BASE_DIR}/logs/judge_*.out"
echo "Merge:    python3 merge_results.py ${OUTPUT_DIR}/ results/rq1_judged.csv"