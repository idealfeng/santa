#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python not found in PATH." >&2
  exit 1
fi

echo "+ $PYTHON_BIN -m venv .venv"
"$PYTHON_BIN" -m venv .venv

echo "+ source .venv/bin/activate"
source .venv/bin/activate

echo "+ python -m pip install -U pip"
python -m pip install -U pip

echo "+ python -m pip install -U setuptools wheel"
python -m pip install -U setuptools wheel

echo "+ python -m pip install -U -e \".[train,notebooks]\""
python -m pip install -U -e ".[train,notebooks]"

echo "+ (optional) python -m pip install -U -e \".[dev]\""
python -m pip install -U -e ".[dev]" || true

echo "+ python scripts/build/build_fastcollide.py (optional)"
python scripts/build/build_fastcollide.py || true
