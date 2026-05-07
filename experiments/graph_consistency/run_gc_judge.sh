#!/usr/bin/env bash
# Graph Consistency Study – Phase 2
# Judges generated questions for each gen-model × judge-model combination.
#
# Prerequisites:
#   export OPENAI_API_KEY=<your-key>
#   bash run_gc_study.sh   (produces gc_study_data/*.json)
#
# Run:
#   bash run_gc_judge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://llmapi.paratera.com"
JUDGE_SAMPLER_TYPE="glm"
CONCURRENCY=32

# All generation models (must match filenames in gc_study_data/)
GEN_MODELS=(
    "qwen35plus"
    "qwen3_235b"
    "qwen3_30b"
    "deepseekv4pro"
    "deepseekv4flash"
)

run_judge() {
    local gen_model_name="$1"
    local judge_model_name="$2"
    local judge_model_id="$3"

    echo ""
    echo "=========================================="
    echo "  Gen: ${gen_model_name}  |  Judge: ${judge_model_id}"
    echo "=========================================="

    python "${SCRIPT_DIR}/judge_consistency.py" \
        --gen_model_name    "${gen_model_name}" \
        --judge_model_name  "${judge_model_name}" \
        --judge_model       "${judge_model_id}" \
        --judge_base_url    "${BASE_URL}" \
        --judge_sampler_type "${JUDGE_SAMPLER_TYPE}" \
        --concurrency       "${CONCURRENCY}"
}

# ── Judge models ───────────────────────────────────────────────────────────────
declare -A JUDGES=(
    ["qwen35plus"]="Qwen3.5-Plus"
    ["glm5"]="GLM-5"
    ["deepseekv4pro"]="DeepSeek-V4-Pro"
)

for judge_name in "qwen35plus" "glm5" "deepseekv4pro"; do
    for gen in "${GEN_MODELS[@]}"; do
        run_judge "${gen}" "${judge_name}" "${JUDGES[${judge_name}]}"
    done
done

echo ""
echo "All judging done. Results in ${SCRIPT_DIR}/gc_study_results/"
