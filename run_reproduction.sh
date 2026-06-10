#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${ANTCHAT_API_KEY:-}" ]]; then
  echo "Missing ANTCHAT_API_KEY. Run: export ANTCHAT_API_KEY=\"YOUR_REAL_KEY\"" >&2
  exit 1
fi

python3 extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.real.json \
  --overwrite \
  --limit 1 \
  --model deepseek-v4-flash

python3 AuthChain.py \
  --input data/reproduction/sample_extract_info.real.json \
  --output data/reproduction/sample_authchain.real.json \
  --overwrite \
  --limit 1 \
  --model deepseek-v4-flash \
  --include-diagnostics \
  --max-revisions 0 \
  --max-retries 5 \
  --request-sleep 70

echo "Done:"
echo "  data/reproduction/sample_extract_info.real.json"
echo "  data/reproduction/sample_authchain.real.json"
