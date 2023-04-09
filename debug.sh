#!/bin/bash
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
./src/gputls/move.sh
python3 -m pip uninstall -y gputls
python3 -m build
python3 -m pip install dist/gputls-0.2.0-py3-none-any.whl
# rm GTLS.ptx GTLS.cubin
# nvcc --ptx GPUFun.cu -rdc=true -o GTLS.ptx
# nvcc -rdc=true --cubin GPUFun.cu -o GTLS.cubin
python3 test1.py