#!/usr/bin/env bash
# Reproduce nsys or ncu captures after the untraced benchmark has finished.
# ncu requires administrator access on hosts with RmProfilingAdminOnly=1.
set -euo pipefail
ROUND2_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROUND2_EXPERIMENT=/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114
ROUND2_OUTPUT="$ROUND2_EXPERIMENT/round2-candidate-screening-20261114"
ROUND2_TOOLS="$ROUND2_EXPERIMENT/bottleneck-analysis-20261114/tools/ncu2025/opt/nvidia/nsight-compute/2025.3.1"
ROUND2_PYTHON=/home/farthing/GTLS/GTLS/.venv/bin/python
ROUND2_NVRTC=/home/farthing/GTLS/GTLS/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab}"
export LD_LIBRARY_PATH="$ROUND2_NVRTC${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$(id -u)" == 0 ]]; then
    export CUPY_CACHE_DIR="$ROUND2_OUTPUT/root-cache/cupy"
    export NUMBA_CACHE_DIR="$ROUND2_OUTPUT/root-cache/numba"
fi
ROUND2_MODE="${1:?Usage: profile.sh nsys|ncu}"
for ROUND2_VARIANT in baseline optimized; do
    ROUND2_CASE="$ROUND2_OUTPUT/$ROUND2_MODE-$ROUND2_VARIANT"
    mkdir -p "$ROUND2_CASE"
    if [[ "$ROUND2_MODE" == nsys ]]; then
        if [[ -e "$ROUND2_CASE/trace.nsys-rep" ]]; then
            echo "Keeping existing $ROUND2_CASE/trace.nsys-rep"
        else
            "$ROUND2_TOOLS/host/target-linux-x64/nsys" profile --trace=cuda,nvtx,osrt \
                --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
                --cuda-memory-usage=true --output="$ROUND2_CASE/trace" \
                "$ROUND2_PYTHON" "$ROUND2_REPO/bench/candidate_screening/validate.py" \
                --mode profile --cool-to 65 --variant "$ROUND2_VARIANT" --output "$ROUND2_CASE" \
                > "$ROUND2_CASE/profile.log" 2>&1
        fi
        if [[ ! -e "$ROUND2_CASE/trace.sqlite" ]]; then
            "$ROUND2_TOOLS/host/target-linux-x64/nsys" export --type=sqlite \
                --output="$ROUND2_CASE/trace.sqlite" "$ROUND2_CASE/trace.nsys-rep" \
                > "$ROUND2_CASE/export.log" 2>&1
        fi
    elif [[ "$ROUND2_MODE" == ncu ]]; then
        if [[ -e "$ROUND2_CASE/profile.ncu-rep" ]]; then
            echo "Keeping existing $ROUND2_CASE/profile.ncu-rep"
        else
            if [[ "$ROUND2_VARIANT" == baseline ]]; then
                ROUND2_FILTER='regex:^searchChi2AdjacentG(8|16|32)$'
                ROUND2_COUNT=3
            else
                ROUND2_FILTER='regex:^(scanScreening|screenCandidates|searchChi2ScreenedG(8|16|32))$'
                ROUND2_COUNT=5
            fi
            "$ROUND2_TOOLS/ncu" --profile-from-start off --target-processes all \
                --kernel-name "$ROUND2_FILTER" --launch-count "$ROUND2_COUNT" \
                --section SpeedOfLight --section MemoryWorkloadAnalysis \
                --section ComputeWorkloadAnalysis --section SchedulerStats \
                --section WarpStateStats --section Occupancy --section LaunchStats \
                --clock-control none --cache-control all --export "$ROUND2_CASE/profile" \
                "$ROUND2_PYTHON" "$ROUND2_REPO/bench/candidate_screening/validate.py" \
                --mode probe --variant "$ROUND2_VARIANT" --output "$ROUND2_CASE" \
                > "$ROUND2_CASE/profile.log" 2>&1
        fi
        "$ROUND2_TOOLS/ncu" --import "$ROUND2_CASE/profile.ncu-rep" --page raw --csv \
            > "$ROUND2_CASE/metrics.csv"
        "$ROUND2_TOOLS/ncu" --import "$ROUND2_CASE/profile.ncu-rep" --page details \
            > "$ROUND2_CASE/details.txt"
    else
        echo "Expected nsys or ncu" >&2
        exit 2
    fi
    echo "Completed $ROUND2_MODE $ROUND2_VARIANT"
done
