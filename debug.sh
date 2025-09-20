#!/bin/bash
BASEDIR=$(dirname "$0")
Version="0.4.5"
Date="Sep 2025"
cd "$BASEDIR"
./changeVersion.sh $Version $Date
./src/gputls/move.sh
python3 KeplerLongCurveSingleTest.py
