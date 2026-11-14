"""Repeated batches must reuse physical storage, including variable-weight searches."""
import cupy as cp
import numpy as np
import pytest

from gputls import core, gtls
from gputls.workspace import BatchWorkspace, LAYOUT_ALLOWANCE, bytes_per_period, bucket_count
from test_candidate_screening import _screened_curve


@pytest.mark.parametrize("screening", [False, True])
@pytest.mark.parametrize("prefix_bytes", [12, 24])
@pytest.mark.parametrize("weighted_bytes", [4, 8])
def test_workspace_alignment_and_stage_aliases(screening, prefix_bytes, weighted_bytes):
    capacity, n, patched, starts = 7, 101, 133, 239
    workspace = BatchWorkspace(capacity, n, patched, starts, screening, prefix_bytes, weighted_bytes)
    index_bytes = workspace.index_bytes
    ranges = []
    for offset, width, dtype in [(0, n, cp.int32), (index_bytes, n, cp.uint32),
                                (2*index_bytes, bucket_count(n), cp.int32),
                                (workspace.cursor_offset, bucket_count(n), cp.int32)]:
        view = workspace.view('scratch', (capacity,width), dtype, offset)
        assert view.data.ptr % np.dtype(dtype).alignment == 0
        ranges.append((view.data.ptr, view.data.ptr+view.nbytes))
    for i, (begin, end) in enumerate(ranges):
        assert all(end <= b or begin >= e for b,e in ranges[i+1:])
    prefix = workspace.view('scratch', (capacity,patched,prefix_bytes), cp.uint8) if screening else None
    weighted = workspace.view('weighted', (capacity,patched,weighted_bytes//4), cp.float32)
    cumulative = workspace.view('cumulative', (capacity,patched), cp.int64)
    lower = workspace.view('lower', (capacity,starts), cp.float32)
    for value,view in enumerate([x for x in (prefix, weighted, cumulative, lower) if x is not None], 1):
        view.fill(value)
    for value,view in enumerate([x for x in (prefix, weighted, cumulative, lower) if x is not None], 1):
        assert bool(cp.all(view == value))
    assert workspace.storage.nbytes <= capacity*bytes_per_period(
        n, patched, starts, screening, prefix_bytes, weighted_bytes)+LAYOUT_ALLOWANCE


def test_repeated_tail_and_refine_batches_reuse_storage(monkeypatch):
    original = core.MultiGPUSearch.batch_size
    checked = []

    class DriverAllocations(cp.cuda.MemoryHook):
        name = 'WorkspaceDriverAllocations'

        def __init__(self):
            self.sizes=[]

        def alloc_postprocess(self, device_id, mem_size, mem_ptr):
            self.sizes.append(mem_size)

    def probe(search, n_periods):
        primary = search.primary
        assert primary.screening
        periods = np.linspace(2, 2.5, 33)
        expected = {refine:primary.chi2_batch(periods,33,refine) for refine in (False,True)}
        storage = primary.workspace.storage.data.ptr
        hook = DriverAllocations()
        with hook:
            for size,refine in [(33,False),(7,True),(1,False),(33,True),(7,False),(33,False)]:
                actual = primary.chi2_batch(periods[:size],33,refine)
                np.testing.assert_array_equal(actual,expected[refine][:size])
                assert primary.workspace.storage.data.ptr == storage
        assert max(hook.sizes,default=0) < 65536
        checked.append(True)
        return original(search,n_periods)

    monkeypatch.setattr(core.MultiGPUSearch,'batch_size',probe)
    t,y,dy = _screened_curve()
    # Odd sample count exercises offsets that were not naturally FP64-aligned.
    result=gtls(t[:-1],y[:-1],dy[:-1],verbose=False).power(periods=np.linspace(2,2.5,65),
            T0_fit_margin=.01,T0_refine_margin=.005,GPUDeviceID=0)
    assert checked and np.isfinite(result.period)
    assert cp.get_default_memory_pool().used_bytes()==0


def test_oom_splits_batch_without_skipping_periods(monkeypatch):
    original_size = core.MultiGPUSearch.batch_size
    original_batch = core.GPUSearch.chi2_batch
    checked=[]

    def fail_large(search,periods,size,refine):
        if len(periods)>2:
            raise cp.cuda.memory.OutOfMemoryError(4096,4096)
        return original_batch(search,periods,size,refine)

    def probe(search,n_periods):
        periods=np.linspace(2,2.5,17)
        expected=search.chi2(periods,17,False)
        monkeypatch.setattr(core.GPUSearch,'chi2_batch',fail_large)
        try:
            actual=search.chi2(periods,17,False)
        finally:
            monkeypatch.setattr(core.GPUSearch,'chi2_batch',original_batch)
        np.testing.assert_array_equal(actual,expected)
        checked.append(True)
        return original_size(search,n_periods)

    monkeypatch.setattr(core.MultiGPUSearch,'batch_size',probe)
    gtls(*_screened_curve(),verbose=False).power(periods=np.linspace(2,2.5,33),GPUDeviceID=0)
    assert checked
