#!/usr/bin/env bash
# One-click build, install and test.
#
# Usage: ./run_tests.sh [GPU_DEVICE_ID]
# Run inside a Python environment where CuPy is already installed.
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
export GTLS_GPU_DEVICE="${1:-${GTLS_GPU_DEVICE:-0}}"

# Reinstall only gputls itself, leaving the existing dependencies untouched
"$PYTHON" -m pip install --quiet --no-deps --force-reinstall .

"$PYTHON" tests/test_gtls.py
