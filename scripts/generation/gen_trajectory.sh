#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export UV_CACHE_DIR=${UV_CACHE_DIR:-"$REPO_ROOT/.uv-cache"}
export HF_HOME=${HF_HOME:-"$REPO_ROOT/.cache/huggingface"}
read -r -a PYTHON_CMD <<< "${PYTHON_CMD:-uv run python}"

# ==============================================================================
# Trajectory Data Generation Runner
# ==============================================================================
#
# Extra CLI arguments are appended to the Python command:
#
#   ./scripts/generation/gen_trajectory.sh --max-steps 80
#   ./scripts/generation/gen_trajectory.sh deepdive
#   RUN_ID=deepdive_20260506_1200 ./scripts/generation/gen_trajectory.sh deepdive
# ==============================================================================

# Generation scope
NUM=${NUM:-800}
DATASET=${DATASET:-""}
DATA_PATH=${DATA_PATH:-""}
RUN_ID=${RUN_ID:-""}
DB_RUN_ID=${DB_RUN_ID:-""}
OUTPUT_PATH=${OUTPUT_PATH:-"output/trajectory"}
SEED=${SEED:-42}

if [ $# -gt 0 ] && [ -z "$DATASET" ] && [ -z "$DATA_PATH" ] && [[ "$1" != -* ]]; then
    DATASET=$1
    shift
fi

if [ -z "$DATA_PATH" ] && [ -n "$DATASET" ]; then
    if [[ "$DATASET" == */* ]]; then
        DATA_PATH="$DATASET"
    elif [[ "$DATASET" == *.jsonl ]]; then
        DATA_PATH="./data/dataset/$DATASET"
    else
        DATA_PATH="./data/dataset/${DATASET}.jsonl"
    fi
fi

if [ -n "$DATA_PATH" ] && [ ! -f "$DATA_PATH" ]; then
    echo "Data file not found: $DATA_PATH" >&2
    exit 1
fi

# Agent config
AGENT_TYPE=${AGENT_TYPE:-"react"}
AGENT_SPEC=${AGENT_SPEC:-""}
SEARCH_MODEL=${SEARCH_MODEL:-"GLM-4.7"}
SEARCH_BASE_URL=${SEARCH_BASE_URL:-"https://llmapi.paratera.com"}
SEARCH_TYPE=${SEARCH_TYPE:-"glm"}

# Search/runtime config
CONCURRENCY=${CONCURRENCY:-8}
MAX_STEPS=${MAX_STEPS:-30}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-50000}
MAX_SEARCH_RESULTS=${MAX_SEARCH_RESULTS:-10}
MAX_CONVERSATIONAL_TURNS=${MAX_CONVERSATIONAL_TURNS:-""}
BROWSER_BASE_URL=${BROWSER_BASE_URL:-"http://localhost:8001"}

# Sampling config
TEMPERATURE=${TEMPERATURE:-0.7}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
TOP_P=${TOP_P:-0.85}

# Context compression config
CONTEXT_STRATEGY=${CONTEXT_STRATEGY:-"sliding_window"}
TOKENIZER_PATH=${TOKENIZER_PATH:-""}
MAX_TOKENS=${MAX_TOKENS:-110000}
MAX_TOOL_CONTEXT=${MAX_TOOL_CONTEXT:-""}
RESET_INTERVAL=${RESET_INTERVAL:-""}

# Generation data defaults to no judging. Set SKIP_JUDGE=false to judge too.
SKIP_JUDGE=${SKIP_JUDGE:-true}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-2}
JUDGE_MODEL=${JUDGE_MODEL:-"glm-4.7-flash:latest"}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-"http://localhost:11434"}
JUDGE_TYPE=${JUDGE_TYPE:-"ollama"}
JUDGE_VOTING_N=${JUDGE_VOTING_N:-1}

CMD_ARGS=(
    "--num" "$NUM"
    "--output-path" "$OUTPUT_PATH"
    "--seed" "$SEED"
    "--agent-type" "$AGENT_TYPE"
    "--search-model" "$SEARCH_MODEL"
    "--search-base-url" "$SEARCH_BASE_URL"
    "--search-type" "$SEARCH_TYPE"
    "--concurrency" "$CONCURRENCY"
    "--judge-concurrency" "$JUDGE_CONCURRENCY"
    "--max-steps" "$MAX_STEPS"
    "--max-context-length" "$MAX_CONTEXT_LENGTH"
    "--max-search-results" "$MAX_SEARCH_RESULTS"
    "--temperature" "$TEMPERATURE"
    "--presence-penalty" "$PRESENCE_PENALTY"
    "--repetition-penalty" "$REPETITION_PENALTY"
    "--top-p" "$TOP_P"
    "--context-strategy" "$CONTEXT_STRATEGY"
    "--max-tokens" "$MAX_TOKENS"
    "--judge-model" "$JUDGE_MODEL"
    "--judge-base-url" "$JUDGE_BASE_URL"
    "--judge-type" "$JUDGE_TYPE"
    "--judge-voting-n" "$JUDGE_VOTING_N"
    "--browser-base-url" "$BROWSER_BASE_URL"
)

if [ -n "$DATA_PATH" ]; then
    CMD_ARGS+=("--data-path" "$DATA_PATH")
fi

if [ -n "$RUN_ID" ]; then
    CMD_ARGS+=("--eval-run-id" "$RUN_ID")
fi

if [ -n "$DB_RUN_ID" ]; then
    CMD_ARGS+=("--run-id" "$DB_RUN_ID")
fi

if [ -n "$AGENT_SPEC" ]; then
    CMD_ARGS+=("--agent-spec" "$AGENT_SPEC")
fi

if [ -n "$MAX_CONVERSATIONAL_TURNS" ]; then
    CMD_ARGS+=("--max-conversational-turns" "$MAX_CONVERSATIONAL_TURNS")
fi

if [ -n "$TOKENIZER_PATH" ]; then
    CMD_ARGS+=("--tokenizer-path" "$TOKENIZER_PATH")
fi

if [ -n "$MAX_TOOL_CONTEXT" ]; then
    CMD_ARGS+=("--max-tool-context" "$MAX_TOOL_CONTEXT")
fi

if [ -n "$RESET_INTERVAL" ]; then
    CMD_ARGS+=("--reset-interval" "$RESET_INTERVAL")
fi

if [ "$SKIP_JUDGE" = "true" ]; then
    CMD_ARGS+=("--skip-judge")
else
    CMD_ARGS+=("--no-skip-judge")
fi

echo "Running trajectory generation:"
echo
echo "Generation scope"
echo "  num:                  $NUM"
echo "  dataset:              ${DATASET:-<none>}"
echo "  data path:            ${DATA_PATH:-<mongodb>}"
echo "  eval run id:          ${RUN_ID:-<new run>}"
echo "  db run id:            ${DB_RUN_ID:-<none>}"
echo "  output base path:     $OUTPUT_PATH"
echo "  seed:                 $SEED"
echo
echo "Agent config"
echo "  agent type:           $AGENT_TYPE"
echo "  agent spec:           ${AGENT_SPEC:-<default>}"
echo "  search model:         $SEARCH_MODEL"
echo "  search base url:      $SEARCH_BASE_URL"
echo "  search type:          $SEARCH_TYPE"
echo
echo "Runtime config"
echo "  concurrency:          $CONCURRENCY"
echo "  max steps:            $MAX_STEPS"
echo "  browser base url:     $BROWSER_BASE_URL"
echo "  skip judge:           $SKIP_JUDGE"
echo

"${PYTHON_CMD[@]}" ./cli/generate_trajectory.py "${CMD_ARGS[@]}" "$@"
