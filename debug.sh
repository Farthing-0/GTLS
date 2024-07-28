#!/bin/bash
BASEDIR=$(dirname "$0")
Version="0.4.2"
Date="July 2024"
cd "$BASEDIR"
rm -r dist
./changeVersion.sh $Version $Date
./src/gputls/move.sh
# python3 -m pip uninstall -y gputls --break-system-packages
python3 -m pip uninstall -y gputls
python3 -m build
# python3 -m pip install dist/gputls-0.3.1-py3-none-any.whl --break-system-packages
python3 -m pip install dist/gputls-$Version-py3-none-any.whl
python3 KeplerLongCurveSingleTest.py
