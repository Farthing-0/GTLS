#!/usr/bin/env bash
set -euo pipefail
WORKSPACE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
WORKSPACE_TRACE="${1:?Usage: profile.sh OUTPUT_DIR [run.py options]}"
shift
WORKSPACE_NSYS="${NSYS_BIN:-/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114/bottleneck-analysis-20261114/tools/ncu2025/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys}"
mkdir -p "$WORKSPACE_TRACE"
if [[ -e "$WORKSPACE_TRACE/trace.nsys-rep" ]]; then
    echo "Trace already exists: $WORKSPACE_TRACE/trace.nsys-rep"
    exit 1
fi
"$WORKSPACE_NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true \
    --output="$WORKSPACE_TRACE/trace" \
    "$WORKSPACE_REPO/.venv/bin/python" "$WORKSPACE_REPO/bench/workspace/run.py" \
    --profile --output "$WORKSPACE_TRACE/run" "$@" > "$WORKSPACE_TRACE/profile.log" 2>&1
"$WORKSPACE_NSYS" export --type=sqlite --output="$WORKSPACE_TRACE/trace.sqlite" \
    "$WORKSPACE_TRACE/trace.nsys-rep" > "$WORKSPACE_TRACE/export.log" 2>&1
"$WORKSPACE_NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
    "$WORKSPACE_TRACE/trace.nsys-rep" > "$WORKSPACE_TRACE/stats.csv" 2> "$WORKSPACE_TRACE/stats.log"
