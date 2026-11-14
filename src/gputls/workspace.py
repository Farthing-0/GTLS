"""Per-device reusable storage for one period batch.

The sort and screening stages never overlap on the search stream. Their large
scratch arrays share one arena, which stays allocated across batch boundaries.
This prevents a freed prefix block from being split and pinned by the next sort.
"""
import cupy as cp
import numpy as np

ALIGNMENT = 512
LAYOUT_ALLOWANCE = 8 * ALIGNMENT


def bucket_count(n):
    """Three points per histogram bucket balances scatter locality and local sort."""
    return (int(n)+2)//3


def bytes_per_period(n, patched, starts=0, screening=False):
    # Sort: index(4N), phase(8N), count(4B), cursor(4B), B=ceil(N/3).
    # Patch: index(4N), residual(4P). Screening: prefix(32P).
    scratch = max(12*n+8*bucket_count(n), 4*n+4*patched, 32*patched if screening else 0)
    return scratch + 16*patched + 4*starts


class BatchWorkspace:
    def __init__(self, capacity, n, patched, starts=0, screening=False):
        self.capacity = int(capacity)
        self.n = int(n)
        self.patched = int(patched)
        self.index_bytes = (capacity*4*n+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        self.bucket_count = bucket_count(n)
        bucket_bytes = (capacity*4*self.bucket_count+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        self.cursor_offset = 3*self.index_bytes+bucket_bytes
        scratch = max(self.cursor_offset+bucket_bytes, self.index_bytes+capacity*4*patched,
                      capacity*32*patched if screening else 0)
        self.regions = {}
        total = 0
        for name, size in [('scratch', scratch), ('weighted', capacity*8*patched),
                           ('cumulative', capacity*8*patched), ('lower', capacity*4*starts)]:
            total = (total+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
            self.regions[name] = (total, size)
            total += size
        self.storage = cp.empty(total, dtype=cp.uint8)

    def view(self, region, shape, dtype, offset=0):
        base, limit = self.regions[region]
        dtype = np.dtype(dtype)
        size = int(np.prod(shape))*dtype.itemsize
        if offset < 0 or offset+size > limit:
            raise ValueError(f'Workspace view exceeds {region}: {offset}+{size}>{limit}')
        return cp.ndarray(shape, dtype=dtype, memptr=self.storage.data+base+offset)
