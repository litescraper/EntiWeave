#!/usr/bin/env bash
# First-time setup: decrypt all dataset files from data/encrypted/ into
# data/dataset/ and data/eval/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if command -v uv &>/dev/null; then
    PYTHON="uv run python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "Error: python3 not found. Install Python 3.12+ or uv." >&2
    exit 1
fi

# Step 1: decrypt encrypted dataset files
if [ ! -d "data/encrypted" ] || [ -z "$(find data/encrypted -name '*.enc' 2>/dev/null)" ]; then
    echo "Warning: no encrypted files found in data/encrypted/, skipping decryption." >&2
else
    echo "Decrypting datasets..."
    $PYTHON -m data.util.dataset_package decrypt --password "entiweave2026"
fi

# Step 2: download train splits from HuggingFace
echo ""
echo "Downloading EntiWeave train splits from HuggingFace..."
$PYTHON -m data.util.download_dataset

# Step 3: convert train_sft split to Swift SFT format
echo ""
echo "Preparing Swift SFT dataset..."
$PYTHON -m data.util.prepare_sft_dataset
