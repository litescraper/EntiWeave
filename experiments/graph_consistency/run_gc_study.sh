#!/usr/bin/env bash
# Graph Consistency Study – Phase 1
# Generates questions for each model using the paratera API.
#
# Prerequisites:
#   export OPENAI_API_KEY=<your-key>
#   python sample_graph.py   (produces sampled_graphs_2.json)
#
# Run:
#   bash run_gc_study.sh
#   bash run_gc_study.sh --retry_failed   # re-attempt previously failed entries

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://llmapi.paratera.com"
SAMPLER_TYPE="glm"
CONCURRENCY=4
EXTRA_ARGS="${1:-}"   # pass --retry_failed as first arg if needed

run_model() {
    local model_name="$1"
    local model_id="$2"

    echo ""
    echo "=========================================="
    echo "  Model: ${model_id}  (name: ${model_name})"
    echo "=========================================="

    python "${SCRIPT_DIR}/generate_gc_study.py" \
        --model_name  "${model_name}" \
        --model       "${model_id}" \
        --base_url    "${BASE_URL}" \
        --sampler_type "${SAMPLER_TYPE}" \
        --concurrency "${CONCURRENCY}" \
        ${EXTRA_ARGS}
}

# ── Generation models ─────────────────────────────────────────────────────────
run_model "qwen35plus"      "Qwen3.5-Plus"
run_model "qwen3_235b"      "Qwen3-235B-A22B-Thinking-2507"
run_model "qwen3_30b"       "Qwen3-30B-A3B-Instruct-2507"
run_model "deepseekv4pro"   "DeepSeek-V4-Pro"
run_model "deepseekv4flash" "DeepSeek-V4-Flash"

echo ""
echo "All models done. Results in ${SCRIPT_DIR}/gc_study_data/"
