#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== scene_ai setup ==="

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: this MVP targets Apple Silicon macOS. MLX will not work elsewhere."
fi

# ---------------------------------------------------------------- python 3.11+
PY=""
for c in python3.12 python3.13 python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done

if [[ -z "$PY" ]] && command -v python3 >/dev/null 2>&1; then
  v="$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')"
  # 3.11 - 3.13 only: 3.14 has no usable wheels yet for this stack
  if [[ "$v" -ge 311 && "$v" -le 313 ]]; then PY="$(command -v python3)"; fi
fi

if [[ -z "$PY" ]]; then
  echo "No Python 3.11-3.13 found. Provisioning Python 3.12…"
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12
    PY="$(uv python find 3.12)"
  elif command -v brew >/dev/null 2>&1; then
    brew install python@3.12
    PY="$(brew --prefix)/bin/python3.12"
  else
    echo "ERROR: need Homebrew or uv to install Python 3.12. See README." >&2
    exit 1
  fi
fi

echo "Using Python: $PY  ($("$PY" --version))"

# ---------------------------------------------------------------------- venv
if [[ ! -d .venv ]]; then
  echo "Creating virtualenv…"
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

echo "Installing dependencies (a few minutes on first run)…"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VPY" -r requirements.txt
else
  "$VPY" -m pip install --quiet --upgrade pip wheel
  "$VPY" -m pip install -r requirements.txt
fi

# ------------------------------------------------------------- model weights
echo
echo "Pre-downloading model weights…"
"$VPY" download_models.py

chmod +x run.sh
echo
echo "Ready. Run ./run.sh"
