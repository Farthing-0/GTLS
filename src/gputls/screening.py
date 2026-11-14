"""Piecewise linear certificates for the existing FP32 weighted statistic.

The two targets are the stored FP32 signal and its FP32 square, separately.
The approximation never supplies a final chi2 or a fitted parameter.
"""
import numpy as np


MIN_WIDTH = 512
RELATIVE_TOLERANCE = 0.01


def template_certificate(signal, tolerance=RELATIVE_TOLERANCE):
    """Joint adaptive knots; sparse second differences and measured error bounds."""
    signal = np.asarray(signal, dtype=np.float32)
    target = np.column_stack((signal, signal * signal)).astype(np.float64)
    n = len(signal)
    maxima = np.max(np.abs(target), axis=0)
    scale = np.maximum(maxima, np.finfo(float).tiny)
    knots = {0, n - 1}
    pending = [(0, n - 1)]
    while pending:
        left, right = pending.pop()
        if right - left <= 1:
            continue
        fraction = np.arange(1, right - left, dtype=float) / (right - left)
        approx = target[left] + fraction[:, None] * (target[right] - target[left])
        error = np.max(np.abs(target[left + 1:right] - approx) / scale, axis=1)
        offset = int(np.argmax(error))
        if error[offset] > tolerance:
            middle = left + 1 + offset
            knots.add(middle)
            pending.extend(((left, middle), (middle, right)))
    knots = np.array(sorted(knots))
    slopes = np.diff(target[knots], axis=0) / np.diff(knots)[:, None]
    positions = np.r_[0, 1, knots[1:-1] + 1, n, n + 1].astype(np.int32)
    coefficients = np.vstack((target[0], slopes[0] - target[0], np.diff(slopes, axis=0),
                              -target[-1] - slopes[-1], target[-1]))
    keep = np.any(coefficients != 0, axis=1)
    positions, coefficients = positions[keep], coefficients[keep]
    # Reconstruct the actual rounded coefficients, including the zero extension.
    # Bound reconstruction roundoff as well, so tolerance is never a certificate.
    impulses = np.zeros((n + 2, 2), dtype=np.longdouble)
    impulses[positions] = coefficients
    represented = np.cumsum(np.cumsum(impulses[::-1], axis=0), axis=0)[::-1][2:]
    mass = np.sum(np.abs(coefficients), axis=0)
    reconstruction_error = 32 * np.finfo(float).eps * (n + 2)**2 * mass
    error = np.nextafter(np.asarray(np.max(np.abs(represented - target), axis=0), float)
                         + reconstruction_error, np.inf)
    return positions, coefficients, error, maxima


def certificates(signal, widths, max_weighted_residual, max_weight, patched_size):
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
    for row, width in zip(signal, widths):
        if width < MIN_WIDTH:
            metadata.append([offset, 0, 0, 0, 0, 0, 0, 0])
            continue
        pos, coeff, error, peaks = template_certificate(row[:width])
        count = len(pos)
        evaluation_gamma = (count + 8)*u/(1-(count + 8)*u)
        prefix_error = 4*(gamma + evaluation_gamma)*p*p*maxima*np.sum(np.abs(coeff), axis=0)
        metadata.append([offset, count, *error, *peaks, *np.nextafter(prefix_error, np.inf)])
        offset += count
        positions.extend(pos)
        coefficients.extend(coeff)
    # Window masses use differences of first-level prefixes. Add a global absolute
    # allowance, which remains valid after cancellation following a large outlier.
    mass_error = np.nextafter(4*gamma*p*maxima, np.inf)
    return (np.asarray(positions, np.int32), np.asarray(coefficients, np.float64).reshape(-1, 2),
            np.asarray(metadata, np.float64), mass_error)
