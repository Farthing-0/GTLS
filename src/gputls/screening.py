"""Piecewise linear certificates for the existing FP32 weighted statistic.

The two targets are the stored FP32 signal and its FP32 square, separately.
The approximation never supplies a final chi2 or a fitted parameter.
"""
import numpy as np
from numba import njit


MIN_WIDTH = 512
RELATIVE_TOLERANCE = 0.01
# Loose-to-tight certificates reduce exact work without changing the statistic.
SCREEN_TOLERANCES = (0.05, 0.015, 0.005, 0.00025)
UNIFORM_PREFIX_STRIDE = 16
GENERAL_PREFIX_STRIDE = 8


def prefix_quantization(patched_size, max_weighted_residual, max_weight):
    """Power-of-two scales with a certified int64 double-prefix range.

    Pad to the complete scan tile, including internal carry calculations. Every
    quantized point is bounded by limit; the sum of its triangular multiplicities
    is therefore below 2**61. Scaling binary32 by a normal power of two is exact
    unless it underflows, in which case rounding to an integer still gives zero.
    Thus each reconstructed point differs by at most half a quantization step.
    """
    size = (int(patched_size)+511)//512*512
    limit = (2**61)//(size*(size+1)//2)-2 if size else 0
    if limit <= 0:
        return None
    maxima = np.array([max_weighted_residual, max_weight], dtype=np.float64)
    if not np.all(np.isfinite(maxima)) or np.any(maxima < 0):
        return None
    exponents = []
    for maximum in maxima:
        exponent = min(126, int(np.floor(np.log2(limit/maximum)))) if maximum else 0
        if exponent < -126:
            return None
        # Resolve a possible rounded log2 at an exact power-of-two boundary.
        while np.ldexp(maximum, exponent) > limit:
            exponent -= 1
        if exponent < -126:
            return None
        exponents.append(exponent)
    scale = np.exp2(exponents).astype(np.float32)
    inverse = np.exp2(-np.array(exponents, dtype=np.float64))
    radius = np.where(maxima == 0, 0., np.nextafter(.5*inverse, np.inf))
    return scale, inverse, radius


@njit(cache=True, fastmath=False)
def _adaptive_knots(target, scale, tolerance):
    """Original knot selection without temporary arrays for each segment.

    Keep separate multiply/add operations and first-maximum tie handling.
    The actual sparse coefficients are still reconstructed and certified below.
    """
    n = len(target)
    chosen = np.zeros(n, dtype=np.bool_)
    chosen[0] = True
    chosen[n-1] = True
    pending = [(0, n-1)]
    while pending:
        left, right = pending.pop()
        if right-left <= 1:
            continue
        delta0 = target[right, 0]-target[left, 0]
        delta1 = target[right, 1]-target[left, 1]
        max_error = -1.0
        middle = left+1
        for index in range(left+1, right):
            fraction = float(index-left)/float(right-left)
            approx0 = target[left, 0]+fraction*delta0
            approx1 = target[left, 1]+fraction*delta1
            e0 = abs(target[index, 0]-approx0)/scale[0]
            e1 = abs(target[index, 1]-approx1)/scale[1]
            if np.isnan(e0) or np.isnan(e1):
                # NumPy argmax selects the first NaN; NaN > tolerance is false.
                max_error = np.nan
                break
            error = max(e0, e1)
            if error > max_error:
                max_error = error
                middle = index
        if max_error > tolerance:
            chosen[middle] = True
            pending.append((left, middle))
            pending.append((middle, right))
    return np.flatnonzero(chosen)


@njit(cache=True, fastmath=False)
def _reconstruction_error(target, positions, coefficients):
    """Measure the represented template in one reverse pass without dense arrays.

    Zero impulses do not change a finite slope. Keep both prefix additions in
    order; their FP64 rounding error is covered by the host certificate budget.
    """
    n = len(target)
    k = len(positions)-1
    slope0 = slope1 = value0 = value1 = error0 = error1 = 0.0
    for i in range(n+1, 1, -1):
        if k >= 0 and positions[k] == i:
            slope0 += coefficients[k, 0]
            slope1 += coefficients[k, 1]
            k -= 1
        value0 += slope0
        value1 += slope1
        e0 = abs(value0-target[i-2, 0])
        e1 = abs(value1-target[i-2, 1])
        if np.isnan(e0) or np.isnan(e1):
            return np.array([np.nan, np.nan])
        error0 = max(error0, e0)
        error1 = max(error1, e1)
    return np.array([error0, error1])


def template_certificate(signal, tolerance=RELATIVE_TOLERANCE):
    """Joint adaptive knots; sparse second differences and measured error bounds."""
    signal = np.asarray(signal, dtype=np.float32)
    target = np.column_stack((signal, signal * signal)).astype(np.float64)
    n = len(signal)
    maxima = np.max(np.abs(target), axis=0)
    scale = np.maximum(maxima, np.finfo(float).tiny)
    knots = _adaptive_knots(target, scale, tolerance)
    slopes = np.diff(target[knots], axis=0) / np.diff(knots)[:, None]
    positions = np.r_[0, 1, knots[1:-1] + 1, n, n + 1].astype(np.int32)
    coefficients = np.vstack((target[0], slopes[0] - target[0], np.diff(slopes, axis=0),
                              -target[-1] - slopes[-1], target[-1]))
    keep = np.any(coefficients != 0, axis=1)
    positions, coefficients = positions[keep], coefficients[keep]
    # Reconstruct the actual rounded coefficients, including the zero extension.
    # Bound reconstruction roundoff as well, so tolerance is never a certificate.
    measured = np.nextafter(_reconstruction_error(target, positions, coefficients), np.inf)
    mass = np.sum(np.abs(coefficients), axis=0)
    # Two FP64 prefix scans contribute <= gamma_(2N)*N*sum|c|, N=n+2.
    # For GPU-supported int32 widths this is far below32*eps*N^2*sum|c|,
    # including rounding in the mass/budget calculation. nextafter above also
    # covers the final subtraction from the exactly represented FP32 target.
    reconstruction_error = 32 * np.finfo(float).eps * (n + 2)**2 * mass
    error = np.nextafter(measured + reconstruction_error, np.inf)
    return positions, coefficients, error, maxima


def certificates(signal, widths, max_weighted_residual, max_weight, patched_size,
                 tolerance=RELATIVE_TOLERANCE, point_error=None):
    """Flatten templates and include the FP64 scan/evaluation error budgets.

    Each 512-point tile uses 16 sequential samples per thread and a block scan.
    Local prefixes, exclusive offsets and reconstruction use <= 48 rounding
    operations per prefix level, plus one carry per tile. Two levels therefore
    have <= 2*(ceil(P/512)+48) operations along a path. A factor of four
    covers propagation, evaluation, and outward rounding of the host budgets.
    P*max|x| bounds every prefix mass, including wrap-around and extreme weights.
    """
    positions, coefficients, metadata = [], [], []
    offset = 0
    p = int(patched_size)
    u = 2.0**-53
    steps = 2 * ((p + 511)//512 + 48)
    gamma = steps*u/(1-steps*u)
    maxima = np.array([max_weighted_residual, max_weight], dtype=float)
    if point_error is not None:
        maxima = np.nextafter(maxima+point_error, np.inf)
    for row, width in zip(signal, widths):
        if width < MIN_WIDTH:
            metadata.append([offset, 0, 0, 0, 0, 0, 0, 0])
            continue
        pos, coeff, error, peaks = template_certificate(row[:width], tolerance)
        count = len(pos)
        evaluation_gamma = (count + 8)*u/(1-(count + 8)*u)
        prefix_error = 4*(gamma + evaluation_gamma)*p*p*maxima*np.sum(np.abs(coeff), axis=0)
        if point_error is not None:
            # Apply the original template to the pointwise quantization error.
            # This is separate from its piecewise-linear approximation error.
            prefix_error += width*peaks*point_error*1.0000000001
        metadata.append([offset, count, *error, *peaks, *np.nextafter(prefix_error, np.inf)])
        offset += count
        positions.extend(pos)
        coefficients.extend(coeff)
    # Window masses use differences of first-level prefixes. Add a global absolute
    # allowance, which remains valid after cancellation following a large outlier.
    mass_error = 4*gamma*p*maxima
    if point_error is not None:
        # Outward sparse endpoints already include the complete quantized
        # window. Only its width, not the preceding prefix, needs this radius.
        mass_error += max(widths)*point_error*1.0000000001
    mass_error = np.nextafter(mass_error, np.inf)
    return (np.asarray(positions, np.int32), np.asarray(coefficients, np.float64).reshape(-1, 2),
            np.asarray(metadata, np.float64), mass_error)


@njit(cache=True,fastmath=False)
def sparse_certificate(positions, coefficients, metadata, stride, padded_size,
                       max_weighted_residual, max_weight, uniform=False):
    """Resample sparse dot coefficients onto the prefix grid for every start residue.

    Linear interpolation of a discrete double prefix with |second difference| <= M
    has error <= M*j*(stride-j)/2 <= M*stride**2/8. Summing this radius times |c|
    covers the resampled dot. A separate FP64 allowance covers coefficient splits,
    merged coefficients and the longer FMA chain. The supplied base certificate
    already covers prefix rounding, with padded_size including the last grid cell.
    Final candidate statistics and their reduction order do not use this approximation.
    """
    columns = 1 if uniform else 2
    maxima = (max_weighted_residual, max_weight)
    capacity = 2*len(positions)*stride
    out_positions = np.empty(capacity, np.int32)
    out_coefficients = np.zeros((capacity, 2), np.float64)
    out_metadata = np.empty((len(metadata)*stride, 8), np.float64)
    end = 0
    u = 2.**-53
    for duration in range(len(metadata)):
        begin, count = int(metadata[duration, 0]), int(metadata[duration, 1])
        allowance = np.zeros(2, np.float64)
        for column in range(columns):
            mass = 0.
            for k in range(begin, begin+count):
                mass += abs(coefficients[k, column])
            mass = np.nextafter(mass*(1+8*count*u), np.inf)
            steps = 4*count+16
            gamma = steps*u/(1-steps*u)
            interpolation = (stride*stride/8)*maxima[column]*mass
            coefficient_rounding = 4*gamma*padded_size*padded_size*maxima[column]*mass
            allowance[column] = (interpolation+coefficient_rounding)*1.0000000001
        for residue in range(stride):
            start = end
            for k in range(begin, begin+count):
                cell = (positions[k]+residue-1)//stride
                rem = (positions[k]+residue-1)%stride
                for side in range(2):
                    weight = (stride-rem)/stride if side == 0 else rem/stride
                    if weight == 0:
                        continue
                    index = cell+side+2
                    # Sorted input cells can only revisit the last two output cells.
                    if end > start and out_positions[end-1] == index:
                        slot = end-1
                    elif end > start+1 and out_positions[end-2] == index:
                        slot = end-2
                    else:
                        slot = end
                        out_positions[end] = index
                        out_coefficients[end, 0] = 0.
                        out_coefficients[end, 1] = 0.
                        end += 1
                    for column in range(columns):
                        out_coefficients[slot, column] += coefficients[k, column]*weight
            stop = end
            end = start
            for k in range(start, stop):
                if out_coefficients[k, 0] != 0 or out_coefficients[k, 1] != 0:
                    out_positions[end] = out_positions[k]
                    out_coefficients[end] = out_coefficients[k]
                    end += 1
            row = duration*stride+residue
            out_metadata[row] = metadata[duration]
            out_metadata[row, 0] = start
            out_metadata[row, 1] = end-start
            for column in range(columns):
                out_metadata[row, 6+column] = np.nextafter(
                    metadata[duration, 6+column]+allowance[column], np.inf)
    return out_positions[:end].copy(), out_coefficients[:end].copy(), out_metadata


@njit(cache=True, fastmath=False)
def uniform_sparse_certificate(positions, coefficients, metadata, stride, padded_size, max_z):
    return sparse_certificate(positions, coefficients, metadata, stride, padded_size,
                              max_z, 0., True)
