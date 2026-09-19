#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv found. Run ./setup.sh first." >&2
  exit 1
fi

echo "Open http://localhost:8000"
echo "(models load in the background; the page shows a loading screen until they're warm)"
echo

# Loopback by default: the dashboard and /ws have no authentication, so binding
# 0.0.0.0 would let anyone on the network spend your GPU. Browsers also refuse
# getUserMedia outside a secure context, so a LAN visitor gets no video anyway.
# Set HOST=0.0.0.0 if you genuinely want to reach it from another device.
exec .venv/bin/python -m uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
