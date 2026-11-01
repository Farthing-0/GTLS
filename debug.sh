#!/bin/bash
BASEDIR=$(dirname "$0")
Version="0.5.0"
Date="Jan-2026"
cd "$BASEDIR"
rm -r dist
./changeVersion.sh $Version $Date
./src/gputls/move.sh
# python -m pip uninstall -y gputls --break-system-packages
python -m pip uninstall -y gputls
python -m build
# python -m pip install dist/gputls-0.3.1-py3-none-any.whl --break-system-packages
python -m pip install dist/gputls-$Version-py3-none-any.whl
python KeplerLongCurveSingleTest.py
