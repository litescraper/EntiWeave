#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ==============================================================================
# QA Dataset Generation Runner
# ==============================================================================
#
# Extra CLI arguments are appended to the Python command:
#
#   ./scripts/generation/generate_qa.sh --run-id run_123 --no-think
# ==============================================================================

# Generation scope
NUM=${NUM:-1}
RUN_ID=${RUN_ID:-""}
TOPIC=${TOPIC:-""}
CATEGORY=${CATEGORY:-""}

# Model config
BASE_URL=${BASE_URL:-"https://llmapi.paratera.com"}
MODEL=${MODEL:-"GLM-5.1"}
SAMPLER_TYPE=${SAMPLER_TYPE:-"glm"}
THINK=${THINK:-true}

# Browser/tool config
BROWSER_URL=${BROWSER_URL:-"http://localhost:8001/"}

# Graph expansion config
MAX_COMPLEXITY=${MAX_COMPLEXITY:-20}
MAX_SEARCH_STEPS=${MAX_SEARCH_STEPS:-2}
PRUNE_FACTOR=${PRUNE_FACTOR:-0.85}

# Fuzzification config
FUZZ_CONSTANT_PROB=${FUZZ_CONSTANT_PROB:-1.0}
FUZZ_USE_CATEGORY_PROB=${FUZZ_USE_CATEGORY_PROB:-0.05}

# Seed generation config
SEED_MAX_STEPS=${SEED_MAX_STEPS:-10}
SEED_GEN_THINKING=${SEED_GEN_THINKING:-true}

CMD_ARGS=(
    # Generation scope
    "--num" "$NUM"

    # Model config
    "--base-url" "$BASE_URL"
    "--model" "$MODEL"
    "--sampler-type" "$SAMPLER_TYPE"

    # Browser/tool config
    "--browser-url" "$BROWSER_URL"

    # Graph expansion config
    "--max-complexity" "$MAX_COMPLEXITY"
    "--max-search-steps" "$MAX_SEARCH_STEPS"
    "--prune-factor" "$PRUNE_FACTOR"

    # Fuzzification config
    "--fuzz-constant-prob" "$FUZZ_CONSTANT_PROB"
    "--fuzz-use-category-prob" "$FUZZ_USE_CATEGORY_PROB"

    # Seed generation config
    "--seed-max-steps" "$SEED_MAX_STEPS"
)

if [ -n "$RUN_ID" ]; then
    CMD_ARGS+=("--run-id" "$RUN_ID")
fi

if [ -n "$TOPIC" ]; then
    CMD_ARGS+=("--topic" "$TOPIC")
fi

if [ -n "$CATEGORY" ]; then
    CMD_ARGS+=("--category" "$CATEGORY")
fi

if [ "$THINK" = "true" ]; then
    CMD_ARGS+=("--think")
else
    CMD_ARGS+=("--no-think")
fi

if [ "$SEED_GEN_THINKING" = "true" ]; then
    CMD_ARGS+=("--seed-gen-thinking")
else
    CMD_ARGS+=("--no-seed-gen-thinking")
fi

echo "Running QA generation:"
echo
echo "Generation scope"
echo "  num:                  $NUM"
echo "  run id:               ${RUN_ID:-<new run>}"
echo "  topic:                ${TOPIC:-<none>}"
echo "  category:             ${CATEGORY:-<none>}"
echo
echo "Model config"
echo "  base url:             $BASE_URL"
echo "  model:                $MODEL"
echo "  sampler type:         $SAMPLER_TYPE"
echo "  think:                $THINK"
echo "  seed gen thinking:    $SEED_GEN_THINKING"
echo
echo "Browser/tool config"
echo "  browser url:          $BROWSER_URL"
echo
echo "Graph expansion config"
echo "  max complexity:       $MAX_COMPLEXITY"
echo "  max search steps:     $MAX_SEARCH_STEPS"
echo "  prune factor:         $PRUNE_FACTOR"
echo
echo "Fuzzification config"
echo "  fuzz constant prob:   $FUZZ_CONSTANT_PROB"
echo "  fuzz category prob:   $FUZZ_USE_CATEGORY_PROB"
echo
echo "Seed generation config"
echo "  seed max steps:       $SEED_MAX_STEPS"
echo

python3 ./cli/generate_qa.py "${CMD_ARGS[@]}" "$@"
