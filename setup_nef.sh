#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[NEF setup] Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "[NEF setup] Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "[NEF setup] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[NEF setup] Installing package in editable mode with dev extras"
python -m pip install -e ".[dev]"

echo "[NEF setup] Running test suite"
PYTHONPATH=src python -m pytest -q

echo
echo "[NEF setup] Done."
echo "Activate env with: source .venv/bin/activate"
