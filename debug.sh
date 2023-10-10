#!/bin/bash
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
./src/gputls/move.sh
# python3 -m pip uninstall -y gputls --break-system-packages
python3 -m pip uninstall -y gputls
python3 -m build
# python3 -m pip install dist/gputls-0.3.1-py3-none-any.whl --break-system-packages
python3 -m pip install dist/gputls-0.3.1-py3-none-any.whl
# python3 test1.py
python3 generatedTest.py

# bakup commands
# rm GTLS.ptx GTLS.cubin
# nvcc --ptx GPUFun.cu -rdc=true -o GTLS.ptx
# nvcc -rdc=true --cubin GPUFun.cu -o GTLS.cubin