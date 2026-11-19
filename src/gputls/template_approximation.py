"""TLS2 2.0.4 piecewise-linear scoring templates for the CUDA search."""

# The fitting routines below are from TLS2 core_fused.py.
# MIT License
#
# Copyright (c) 2018 Michael Hippke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numba
import numpy


TLS2_PL_MIN_LENGTH = 64
TLS2_PL_EPS = 0.02
TLS2_PL_A2_SPREAD_MAX = 0.1


@numba.njit(cache=True)
def _segment_ok(a, b, c, tol):
    """Linear interpolant from sample b to sample c within tol of a[b..c]?"""
    s = (a[c] - a[b]) / (c - b)
    for j in range(b + 1, c):
        if abs(a[b] + s * (j - b) - a[j]) > tol:
            return False
    return True


@numba.njit(cache=True)
def _greedy_knots(a, tol):
    """Integer knots 0 = q_0 < ... < q_K = len(a) - 1 such that the linear
    interpolant through (q_k, a[q_k]) stays within tol of every sample.
    Each segment is grown by doubling, then bisection (O(L log L))."""
    n = len(a)
    knots = numpy.empty(n, dtype=numpy.int64)
    knots[0] = 0
    k = 1
    b = 0
    while b < n - 1:
        good = b + 1  # always valid
        step = 1
        bad = -1
        while True:
            c = b + 2 * step
            if c > n - 1:
                c = n - 1
            if c <= good:
                break
            if _segment_ok(a, numpy.int64(b), c, tol):  # b never literal-typed
                good = c
                if c == n - 1:
                    break
                step *= 2
            else:
                bad = c
                break
        if bad > 0:
            while bad - good > 1:
                c = (good + bad) // 2
                if _segment_ok(a, numpy.int64(b), c, tol):
                    good = c
                else:
                    bad = c
        knots[k] = good
        k += 1
        b = good
    return knots[:k]


@numba.njit(cache=True)
def _pl_lsq(a, q):
    """Least-squares values v at the knots q of the piecewise-linear fit to a
    (hat-function basis, tridiagonal normal equations, Thomas algorithm).
    Returns (v, fitted samples)."""
    nq = len(q)
    diag = numpy.zeros(nq)
    off = numpy.zeros(nq)
    rhs = numpy.zeros(nq)
    for s in range(nq - 1):
        h = q[s + 1] - q[s]
        last = q[s + 1] if s == nq - 2 else q[s + 1] - 1
        for j in range(q[s], last + 1):
            t = (j - q[s]) / h
            h0 = 1.0 - t
            diag[s] += h0 * h0
            diag[s + 1] += t * t
            off[s] += h0 * t
            rhs[s] += h0 * a[j]
            rhs[s + 1] += t * a[j]
    # Thomas algorithm (symmetric tridiagonal, positive definite)
    cp = numpy.zeros(nq)
    dp = numpy.zeros(nq)
    cp[0] = off[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, nq):
        den = diag[i] - off[i - 1] * cp[i - 1]
        cp[i] = off[i] / den if i < nq - 1 else 0.0
        dp[i] = (rhs[i] - off[i - 1] * dp[i - 1]) / den
    v = numpy.zeros(nq)
    v[nq - 1] = dp[nq - 1]
    for i in range(nq - 2, -1, -1):
        v[i] = dp[i] - cp[i] * v[i + 1]
    p = numpy.empty(len(a))
    for s in range(nq - 1):
        h = q[s + 1] - q[s]
        for j in range(q[s], q[s + 1] + 1):
            t = (j - q[s]) / h
            p[j] = (1.0 - t) * v[s] + t * v[s + 1]
    return v, p


def pl_fit(a, eps):
    """Piecewise-linear approximation of the samples a (len >= 2), idea L3.

    Knots from _greedy_knots(a, eps * max|a|); then least-squares values at
    the knots (eps <= 0: every sample is a knot, exact). Returns
    (positions, coefficients, sum of the approximation, approximation): for
    any x, sum_j p_j x[i + j] = sum_t c_t D[i + pos_t] with D the double
    prefix sum D[k] = sum_{l<k} S[l], S[l] = sum_{q<l} x[q]. The
    coefficients are the second differences of the zero-padded p.
    """
    a = numpy.asarray(a, dtype=float)
    n = len(a)
    j = numpy.arange(n)
    if eps > 0:
        q = _greedy_knots(a, eps * numpy.max(numpy.abs(a)))
        v, p = _pl_lsq(a, q)
    else:
        q = j.copy()
        v = a.copy()
        p = a.copy()
    s = numpy.diff(v) / numpy.diff(q)  # slopes of the K segments
    pos = [0, 1]
    coef = [v[0], s[0] - v[0]]
    for k in range(1, len(q) - 1):
        pos.append(q[k] + 1)
        coef.append(s[k] - s[k - 1])
    pos += [n, n + 1]
    coef += [-v[-1] - s[-1], v[-1]]
    pos = numpy.array(pos, dtype=numpy.int64)
    coef = numpy.array(coef)
    keep = coef != 0
    return pos[keep], coef[keep], float(numpy.sum(p)), p



def scoring_templates(lc_arr, eps):
    """Pack TLS2's sparse second differences and means for each template row.

    Row metadata contains coefficient offset, residual and squared-template
    term counts, and the original profile length. A zero term count selects
    GTLS's full-template kernel.
    """
    metadata = numpy.zeros((len(lc_arr), 4), dtype=numpy.int32)
    sums = numpy.zeros((len(lc_arr), 4))
    positions, coefficients = [], []
    offset = 0
    for index, signal in enumerate(lc_arr):
        a = 1 - numpy.asarray(signal, dtype=float)
        if len(a) < TLS2_PL_MIN_LENGTH:
            continue
        pos, coeff, total, _ = pl_fit(a, eps)
        if eps > 0 and len(coeff) >= len(a):
            continue
        squared = a * a
        sq_pos, sq_coeff, sq_total, _ = pl_fit(squared, eps)
        metadata[index] = (offset, len(coeff), len(sq_coeff), len(a))
        sums[index] = (total, sq_total, numpy.sum(squared), 0)
        offset += len(coeff) + len(sq_coeff)
        positions.extend((pos, sq_pos))
        coefficients.extend((coeff, sq_coeff))
    if not coefficients:
        return None
    return metadata, sums, numpy.concatenate(positions).astype(numpy.int32), numpy.concatenate(coefficients)
