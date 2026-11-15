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


def bytes_per_period(n, patched, starts=0, screening=False, prefix_bytes=24, weighted_bytes=8,
                     prefix_points=None):
    # Sort: index(4N), compact phase key(4N), count(4B), cursor(4B), B=ceil(N/3).
    # Equal weights use12 bytes per sparse prefix node and4P weighted residuals;
    # general weights use24 bytes per sparse node and8P weighted pairs.
    # Integer sums use8P. The default prefix_points retains the full-prefix layout.
    prefix_points = patched if prefix_points is None else prefix_points
    scratch = max(8*n+8*bucket_count(n), prefix_bytes*prefix_points if screening else 0)
    return scratch + (8+weighted_bytes)*patched + 4*starts


class BatchWorkspace:
    def __init__(self, capacity, n, patched, starts=0, screening=False, prefix_bytes=24, weighted_bytes=8,
                 prefix_points=None):
        self.capacity = int(capacity)
        self.n = int(n)
        self.patched = int(patched)
        self.index_bytes = (capacity*4*n+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        self.bucket_count = bucket_count(n)
        bucket_bytes = (capacity*4*self.bucket_count+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        self.cursor_offset = 2*self.index_bytes+bucket_bytes
        prefix_points = patched if prefix_points is None else prefix_points
        scratch = max(self.cursor_offset+bucket_bytes, capacity*prefix_bytes*prefix_points if screening else 0)
        self.regions = {}
        total = 0
        for name, size in [('scratch', scratch), ('weighted', capacity*weighted_bytes*patched),
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
