#!/bin/bash
set -euo pipefail

BASEDIR=$(cd -- "$(dirname -- "$0")" && pwd)
Version="0.4.4"
Date="Oct-2025"
cd "$BASEDIR"
./changeVersion.sh "$Version" "$Date"
./src/gputls/move.sh
python3 -m pip wheel --no-deps --wheel-dir dist .
python3 -m pip install --force-reinstall --no-deps "dist/gputls-$Version-py3-none-any.whl"
python3 KeplerLongCurveSingleTest.py
