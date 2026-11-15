"""Find the newest installed nsys, including the copy bundled with Nsight Compute."""
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess


def find_nsys():
    override = os.environ.get('NSYS_BIN')
    if override:
        binary = shutil.which(override)
        if binary is None:
            raise FileNotFoundError(f'NSYS_BIN is not executable: {override}')
        return binary
    repo = Path(__file__).resolve().parents[2]
    candidates = [shutil.which('nsys')]
    patterns = [
        '/opt/nvidia/nsight-systems/*/target-linux-x64/nsys',
        '/opt/nvidia/nsight-systems/*/bin/nsys',
        '/opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys',
        str(repo.parents[1] / 'GTLS-Compare/*/bottleneck-analysis-*/tools/*/opt/nvidia/'
            'nsight-compute/*/host/target-linux-x64/nsys'),
    ]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    available = []
    for candidate in set(candidates) - {None}:
        try:
            version = subprocess.check_output([candidate, '--version'], text=True,
                                              stderr=subprocess.STDOUT, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r'version (\d+(?:\.\d+)+)', version, re.IGNORECASE)
        if match:
            available.append((tuple(map(int, match[1].split('.'))), candidate))
    if not available:
        raise FileNotFoundError('No working nsys found; install Nsight Systems or set NSYS_BIN')
    return max(available)[1]


if __name__ == '__main__':
    print(find_nsys())
