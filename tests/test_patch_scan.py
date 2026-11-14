"""Fused permutation and fixed-point scan preserve the original point arithmetic."""
import cupy as cp
import numpy as np
import pytest

from gputls import core


@pytest.mark.parametrize('n,patched', [(1, 9), (17, 401), (511, 1027), (513, 4099)])
def test_patch_scan_matches_weight_formula_and_integer_prefix(n, patched):
    rng = np.random.default_rng(721)
    residual = rng.normal(0, .2, n).astype(np.float32)
    residual[::7] = np.float32(1.5 / 1024)  # ties must round to even
    dy = np.exp(rng.uniform(-16, 4, n)).astype(np.float32)
    indices = np.stack([rng.permutation(n) for _ in range(3)]).astype(np.int32)
    weights = np.float32(1) / np.maximum(dy*dy, np.float32(1e-10))
    order = indices[:, np.arange(patched) % n]
    expected = np.stack((weights[order]*residual[order], weights[order]), axis=-1)
    fixed = np.rint(residual[order]*np.float32(1024)).astype(np.int64)

    module = core.load_module()
    prepared = cp.empty((n, 2), cp.float32)
    module.get_function('prepareWeights')(((n+255)//256,), (256,),
        (cp.asarray(residual), cp.asarray(dy), prepared, np.int32(n)))
    weighted = cp.empty((3, patched, 2), cp.float32)
    cumulative = cp.empty((3, patched), cp.int64)
    threads = min(1024, max(32, 1 << ((patched+1)//2-1).bit_length()))
    module.get_function('patchScan')((3,), (threads,),
        (cp.asarray(indices), prepared, weighted, cumulative, np.float32(1024),
         np.int32(n), np.int32(patched)))
    assert weighted.get().tobytes() == expected.tobytes()
    np.testing.assert_array_equal(cumulative.get(), np.cumsum(fixed, axis=1))
