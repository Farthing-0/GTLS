#!/usr/bin/env bash
# Reproduce nsys or ncu captures after the untraced benchmark has finished.
# ncu requires administrator access on hosts with RmProfilingAdminOnly=1.
set -euo pipefail
ROUND1_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROUND1_EXPERIMENT=/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114
ROUND1_OUTPUT="$ROUND1_EXPERIMENT/round1-window-reuse-20261114"
ROUND1_TOOLS="$ROUND1_EXPERIMENT/bottleneck-analysis-20261114/tools/ncu2025/opt/nvidia/nsight-compute/2025.3.1"
ROUND1_PYTHON=/home/farthing/GTLS/GTLS/.venv/bin/python
ROUND1_NVRTC=/home/farthing/GTLS/GTLS/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab}"
export LD_LIBRARY_PATH="$ROUND1_NVRTC${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$(id -u)" == 0 ]]; then
    export CUPY_CACHE_DIR="$ROUND1_OUTPUT/root-cache/cupy"
    export NUMBA_CACHE_DIR="$ROUND1_OUTPUT/root-cache/numba"
fi
ROUND1_MODE="${1:?Usage: profile.sh nsys|ncu}"
for ROUND1_VARIANT in baseline optimized; do
    ROUND1_CASE="$ROUND1_OUTPUT/$ROUND1_MODE-$ROUND1_VARIANT"
    mkdir -p "$ROUND1_CASE"
    if [[ "$ROUND1_MODE" == nsys ]]; then
        if [[ -e "$ROUND1_CASE/trace.nsys-rep" ]]; then
            echo "Keeping existing $ROUND1_CASE/trace.nsys-rep"
        else
            "$ROUND1_TOOLS/host/target-linux-x64/nsys" profile --trace=cuda,nvtx,osrt \
                --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
                --cuda-memory-usage=true --output="$ROUND1_CASE/trace" \
                "$ROUND1_PYTHON" "$ROUND1_REPO/bench/window_reuse/validate.py" \
                --mode profile --variant "$ROUND1_VARIANT" --output "$ROUND1_CASE" \
                > "$ROUND1_CASE/profile.log" 2>&1
        fi
        if [[ ! -e "$ROUND1_CASE/trace.sqlite" ]]; then
            "$ROUND1_TOOLS/host/target-linux-x64/nsys" export --type=sqlite \
                --output="$ROUND1_CASE/trace.sqlite" "$ROUND1_CASE/trace.nsys-rep" \
                > "$ROUND1_CASE/export.log" 2>&1
        fi
    elif [[ "$ROUND1_MODE" == ncu ]]; then
        if [[ -e "$ROUND1_CASE/profile.ncu-rep" ]]; then
            echo "Keeping existing $ROUND1_CASE/profile.ncu-rep"
        else
            if [[ "$ROUND1_VARIANT" == baseline ]]; then
                ROUND1_FILTER='regex:^searchChi2G(8|16|32)$'
            else
                ROUND1_FILTER='regex:^searchChi2AdjacentG(8|16|32)$'
            fi
            "$ROUND1_TOOLS/ncu" --profile-from-start off --target-processes all \
                --kernel-name "$ROUND1_FILTER" --launch-count 3 \
                --section SpeedOfLight --section MemoryWorkloadAnalysis \
                --section ComputeWorkloadAnalysis --section SchedulerStats \
                --section WarpStateStats --section Occupancy --section LaunchStats \
                --clock-control none --cache-control all --export "$ROUND1_CASE/profile" \
                "$ROUND1_PYTHON" "$ROUND1_REPO/bench/window_reuse/validate.py" \
                --mode probe --variant "$ROUND1_VARIANT" --output "$ROUND1_CASE" \
                > "$ROUND1_CASE/profile.log" 2>&1
        fi
        "$ROUND1_TOOLS/ncu" --import "$ROUND1_CASE/profile.ncu-rep" --page raw --csv \
            > "$ROUND1_CASE/metrics.csv"
        "$ROUND1_TOOLS/ncu" --import "$ROUND1_CASE/profile.ncu-rep" --page details \
            > "$ROUND1_CASE/details.txt"
    else
        echo "Expected nsys or ncu" >&2
        exit 2
    fi
    echo "Completed $ROUND1_MODE $ROUND1_VARIANT"
done
