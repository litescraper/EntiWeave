#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"

cat > "$ENV_FILE" <<'EOF'
TOOLS_API_KEY="secret"
MONGODB_URL="Your MongoDB URL here"
OPENAI_API_KEY="Your OpenAI API key here"
JINA_API_KEY="Your JINA API key here"
SERPER_API_KEY="Your SERPER API key here"
EOF

echo "Loaded environment variables to $ENV_FILE"
