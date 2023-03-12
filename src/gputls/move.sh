#!/bin/bash 
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

Fun=$(cat ${SCRIPTPATH}//GPUFun.cu)
rm ${SCRIPTPATH}/GPUFun.py
echo "def getGPUCode():
    GPUCode = \"\"\"
$Fun
\"\"\"
    return GPUCode" >> ${SCRIPTPATH}/GPUFun.py