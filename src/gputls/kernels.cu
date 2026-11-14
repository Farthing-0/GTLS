// Block-wide inclusive prefix sum of one value per thread (blockDim.x a multiple of 32).
// Returns the thread's inclusive sum and stores the block total in *blockTotal.
template <typename T>
__device__ __forceinline__ T blockInclusiveScan(T value, T* blockTotal) {
    __shared__ T warpSums[32];
    int laneIndex = threadIdx.x & 31;
    int warpIndex = threadIdx.x >> 5;
    int nWarps = blockDim.x >> 5;
    for (int offset = 1; offset < 32; offset <<= 1) {
        T neighborValue = __shfl_up_sync(0xffffffff, value, offset);
        if (laneIndex >= offset) {
            value += neighborValue;
        }
    }
    if (laneIndex == 31) {
        warpSums[warpIndex] = value;
    }
    __syncthreads();
    if (warpIndex == 0) {
        T warpPrefix = laneIndex < nWarps ? warpSums[laneIndex] : (T)0;
        for (int offset = 1; offset < 32; offset <<= 1) {
            T neighborValue = __shfl_up_sync(0xffffffff, warpPrefix, offset);
            if (laneIndex >= offset) {
                warpPrefix += neighborValue;
            }
        }
        warpSums[laneIndex] = warpPrefix;
    }
    __syncthreads();
    if (warpIndex > 0) {
        value += warpSums[warpIndex - 1];
    }
    *blockTotal = warpSums[nWarps - 1];
    __syncthreads();  // warpSums is reused by the next call
    return value;
}

extern "C"{
    #define SECONDS_PER_DAY 86400              // Seconds in a day

    /**
     * Phase sort, stage 1: fold and count the points of each bucket floor(phase * tSize)
     * (about one point per bucket). phase = time / period - floor(time / period), with the
     * division done as a multiplication by the correctly rounded 1 / period (float64 division
     * is a long instruction sequence, multiplication a single instruction).
     */
    __global__ void foldBuckets(const double* time, const double* inversePeriods, double* phase,
                                int* counts, int tSize) {
        int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (pointIndex >= tSize) {
            return;
        }
        double cycles = time[pointIndex] * inversePeriods[periodIndex];
        double pointPhase = cycles - (int)(cycles);
        long long rowOffset = (long long)periodIndex * tSize;
        phase[rowOffset + pointIndex] = pointPhase;
        int bucket = min((int)(pointPhase * tSize), tSize - 1);
        atomicAdd(counts + rowOffset + bucket, 1);
    }

    /**
     * Exclusive prefix sum of each row of counts (one block per row): the first position of
     * every bucket in the sorted order.
     */
    __global__ void scanCounts(const int* counts, int* cursor, int tSize) {
        long long rowOffset = (long long)blockIdx.x * tSize;
        int carry = 0;
        for (int tileStart = 0; tileStart < tSize; tileStart += blockDim.x) {
            int bucket = tileStart + threadIdx.x;
            int bucketCount = bucket < tSize ? counts[rowOffset + bucket] : 0;
            int tileTotal;
            int inclusiveSum = blockInclusiveScan(bucketCount, &tileTotal);
            if (bucket < tSize) {
                cursor[rowOffset + bucket] = carry + inclusiveSum - bucketCount;
            }
            carry += tileTotal;
        }
    }

    /**
     * Phase sort, stage 2: scatter the point indices into their buckets. cursor holds the
     * first position of each bucket on entry and one past its last position on exit.
     */
    __global__ void bucketScatter(const double* phase, int* cursor, int* sortIndex, int tSize) {
        int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (pointIndex >= tSize) {
            return;
        }
        long long rowOffset = (long long)periodIndex * tSize;
        int bucket = min((int)(phase[rowOffset + pointIndex] * tSize), tSize - 1);
        sortIndex[rowOffset + atomicAdd(cursor + rowOffset + bucket, 1)] = pointIndex;
    }

    /**
     * Phase sort, stage 3: insertion sort of each bucket by (phase, index). Buckets increase
     * with phase, so the result equals a stable argsort of the phases.
     */
    __global__ void sortBuckets(const double* phase, const int* counts, const int* bucketEnds,
                                int* sortIndex, int tSize) {
        int bucket = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (bucket >= tSize) {
            return;
        }
        long long rowOffset = (long long)periodIndex * tSize;
        int bucketSize = counts[rowOffset + bucket];
        if (bucketSize < 2) {
            return;
        }
        int* bucketPoints = sortIndex + rowOffset + bucketEnds[rowOffset + bucket] - bucketSize;
        const double* rowPhase = phase + rowOffset;
        for (int next = 1; next < bucketSize; next++) {
            int keyPoint = bucketPoints[next];
            double keyPhase = rowPhase[keyPoint];
            int slot = next - 1;
            while (slot >= 0 && (rowPhase[bucketPoints[slot]] > keyPhase ||
                                 (rowPhase[bucketPoints[slot]] == keyPhase && bucketPoints[slot] > keyPoint))) {
                bucketPoints[slot + 1] = bucketPoints[slot];
                slot--;
            }
            bucketPoints[slot + 1] = keyPoint;
        }
    }

    /**
     * Range (in samples) of physically plausible durations of each period, as TLS 2:
     * T14 / P = r * (4 P / (pi G M))^(1/3) / P, capped at upperLimit. The shortest duration
     * uses rShort = R_sun * R_star_min (small planet) with gmShort = pi G M_sun m_short, the
     * longest rLong = R_sun * R_star_max + 2 R_jup with gmLong = pi G M_sun m_long (see
     * grid.T14 and grid.duration_limit_masses).
     */
    __global__ void durationsGrid(const double* periods, int* durationsMax, int* durationsMin,
                                 const double* tLength, const int* tSize, const int* periodSize,
                                 double rShort, double gmShort, double rLong, double gmLong,
                                 double upperLimit) {
        int periodIndex = blockDim.x * blockIdx.x + threadIdx.x;

        if (periodIndex < (*periodSize)) {
            double timeSpan = *tLength;
            int pointCount = *tSize;

            // A transit can be split over the start and end of the data, so the window of
            // the longest duration is widened by (transits + 1) / transits
            double periodDays = periods[periodIndex];
            double naiveTransitCount = timeSpan / periodDays;
            double correctionFactor = (naiveTransitCount + 1.0) / naiveTransitCount;

            double periodSeconds = periodDays * SECONDS_PER_DAY;
            double durationMinFraction = rShort * pow((4.0 * periodSeconds) / gmShort, 1.0 / 3.0) / periodSeconds;
            double durationMaxFraction = rLong * pow((4.0 * periodSeconds) / gmLong, 1.0 / 3.0) / periodSeconds;
            durationMinFraction = (durationMinFraction > upperLimit) ? upperLimit : durationMinFraction;
            durationMaxFraction = (durationMaxFraction > upperLimit) ? upperLimit : durationMaxFraction;

            int durationMinSamples = floor(durationMinFraction * pointCount);
            int durationMaxSamples = ceil(durationMaxFraction * pointCount * correctionFactor);

            durationsMin[periodIndex] = durationMinSamples;
            durationsMax[periodIndex] = durationMaxSamples;
        }
    }

    /**
     * Optimized duration boolean array generation
     */
    __global__ void durationBool(int* durationsMax, int* durationsMin, const int* durationSize,
                                const int* periodSize, const int* durations, bool* durationBoolArray) {
        int periodIndex = blockDim.x * blockIdx.x + threadIdx.x;
        int durationIndex = blockDim.y * blockIdx.y + threadIdx.y;

        if (periodIndex < (*periodSize) && durationIndex < (*durationSize)) {
            int durationMin = durationsMin[periodIndex];
            int durationMax = durationsMax[periodIndex];
            int currentDuration = durations[durationIndex];

            durationBoolArray[durationIndex + periodIndex * (*durationSize)] =
                (currentDuration >= durationMin && currentDuration <= durationMax);
        }
    }

    /**
     * Phase-folded light curve, extended by its first points to handle transits at the phase
     * edge, in the form used by searchChi2: the residual r = 1 - flux and (w * r, w), with
     * w = 1 / dy^2.
     */
    __global__ void patchWeighted(const int* sortIndex, const float* residual, const float* dy,
                                  float* patchedResidual, float2* patchedWeighted,
                                  int tSize, int patchedSize) {
        int patchedIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (patchedIndex >= patchedSize) {
            return;
        }
        int sourceIndex = sortIndex[(long long)periodIndex * tSize + patchedIndex % tSize];
        float pointResidual = residual[sourceIndex];
        float variance = dy[sourceIndex] * dy[sourceIndex];
        // Prevent division by zero
        if (variance < 1e-10f) {
            variance = 1e-10f;
        }
        float weight = 1.0f / variance;
        long long outputIndex = (long long)periodIndex * patchedSize + patchedIndex;
        patchedResidual[outputIndex] = pointResidual;
        patchedWeighted[outputIndex] = make_float2(weight * pointResidual, weight);
    }

    /**
     * Inclusive prefix sum of each row of the patched residuals (one block per row), in fixed
     * point: every residual is rounded to a multiple of 1 / scale (scale = 2^E, chosen by
     * GPUSearch so the row sums fit in 63 bits), then summed exactly in int64. Window sums are
     * exact differences of two prefix sums, whatever the row length and the residual offset.
     */
    __global__ void scanResidual(const float* patchedResidual, long long* cumResidual, float scale,
                                 int patchedSize) {
        long long rowOffset = (long long)blockIdx.x * patchedSize;
        long long carry = 0;
        for (int tileStart = 0; tileStart < patchedSize; tileStart += blockDim.x) {
            int patchedIndex = tileStart + threadIdx.x;
            long long fixedResidual =
                patchedIndex < patchedSize ? __float2ll_rn(patchedResidual[rowOffset + patchedIndex] * scale) : 0;
            long long tileTotal;
            long long inclusiveSum = blockInclusiveScan(fixedResidual, &tileTotal);
            if (patchedIndex < patchedSize) {
                cumResidual[rowOffset + patchedIndex] = carry + inclusiveSum;
            }
            carry += tileTotal;
        }
    }

}  // extern "C"

// Order-preserving map of a float onto an unsigned int: a < b <=> key(a) < key(b)
__device__ __forceinline__ unsigned int orderedKey(float value) {
    unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/**
 * Lowest chi2 of every period over the searched durations yBegin <= selectedIndex < yEnd and
 * their start points, fused into one pass (no per-start-point arrays). For a window of L points
 * starting at `windowStart` with template depth profile s_i and amplitude
 * k = mean_depth * overshoot / SIGNAL_DEPTH, the chi2
 *     sum_{all points} w r^2 - sum_{window} w r^2 + sum_{window} w (r - k s)^2
 * is evaluated as base + delta, delta = k^2 A_2 - 2 k A_R with A_R = sum s_i (w r)_i (crossTerm),
 * A_2 = sum s_i^2 w_i (squareTerm) (the algebraic expansion of TLS 2.0). base = sum_{all points}
 * w r^2 is the same for every window and period, so the windows are ranked by delta alone, in
 * float32 (decode_keys adds base). The mean depth is the exact fixed-point window sum of r (see
 * scanResidual), converted and divided by L with float32 rounding: no float64 per window.
 *
 * Work items of a period are the start points of all searched durations, laid out by
 * `offsets`; start points of selected duration selectedIndex are multiples of
 * strides[selectedIndex]. The searched durations are the union over the batch; a period only
 * tests those of its own row of periodMask (physically plausible durations, see durationBool),
 * so results do not depend on the batch. best[periodIndex] receives min over work items of
 * (orderedKey(delta) << 32 | selectedIndex * tSize + windowStart), so ties resolve to the first
 * duration, then the first start point.
 *
 * G consecutive threads share one window (G divides 32): they read G consecutive points per
 * step and sum A_R, A_2 with warp shuffles. With one thread per window, neighbouring threads
 * read points strides[selectedIndex] apart, which wastes memory transactions and evicts the
 * overlap of neighbouring windows from L1 once the stride is large; G is chosen per duration
 * from the stride (see core.lanes_per_window and bench/window_lanes on branch perf/window-lanes).
 */
template <int G>
__device__ __forceinline__ void searchChi2Lanes(
        unsigned long long* best,
        const long long* cumResidual,  // (periods, patchedSize), inclusive, fixed point
        float invScale,                // 1 / scale of cumResidual (a power of 2)
        const float2* patchedWeighted, // (periods, patchedSize)
        const float* signal,           // (durations, maxWidth)
        const float* overshoot,        // (durations)
        const int* widths,             // (durations)
        const int* selected,           // (nSelected) searched durations
        const int* offsets,            // (nSelected + 1)
        const int* strides,            // (nSelected)
        const bool* periodMask,        // (periods, nDurations)
        int yBegin, int yEnd, int nDurations, int tSize, int patchedSize,
        int maxWidth, float transitDepthMin,
        const float* lowerBounds = nullptr, const unsigned long long* upperKeys = nullptr,
        int nBounds = 0) {
    int periodIndex = blockIdx.y;
    int laneInWindow = threadIdx.x % G;
    int workItem = offsets[yBegin] + (blockIdx.x * blockDim.x + threadIdx.x) / G;
    unsigned long long key = 0xffffffffffffffffull;
    float crossTerm = 0.0f;
    float squareTerm = 0.0f;
    float meanDepth = 0.0f;
    bool active = false;
    int selectedIndex = yBegin;
    int windowStart = 0;
    int durationIndex = 0;

    if (workItem < offsets[yEnd]) {
        while (offsets[selectedIndex + 1] <= workItem) {
            selectedIndex++;
        }
        windowStart = (workItem - offsets[selectedIndex]) * strides[selectedIndex];
        durationIndex = selected[selectedIndex];
        int windowWidth = widths[durationIndex];
        if (periodMask[(long long)periodIndex * nDurations + durationIndex]) {
            const long long* rowCumResidual = cumResidual + (long long)periodIndex * patchedSize;
            long long windowSum = rowCumResidual[windowStart + windowWidth - 1]
                                  - (windowStart > 0 ? rowCumResidual[windowStart - 1] : 0);
            meanDepth = __fdiv_rn(__ll2float_rn(windowSum) * invScale, (float)windowWidth);
            active = meanDepth > transitDepthMin;  // the same for all G lanes of the window
            if (active && lowerBounds) {
                unsigned int high = (unsigned int)(upperKeys[periodIndex] >> 32);
                float upper = __uint_as_float(high >= 0x80000000u ? high ^ 0x80000000u : ~high);
                // Strict comparison retains every exact tie, including its original location.
                active = !(lowerBounds[(long long)periodIndex * nBounds + workItem] > upper);
            }
        }
        if (active) {
            const float2* windowWeighted = patchedWeighted + (long long)periodIndex * patchedSize + windowStart;
            const float* durationSignal = signal + (long long)durationIndex * maxWidth;
            for (int pointInWindow = laneInWindow; pointInWindow < windowWidth; pointInWindow += G) {
                float signalValue = durationSignal[pointInWindow];
                float2 weighted = windowWeighted[pointInWindow];
                crossTerm = fmaf(signalValue, weighted.x, crossTerm);
                squareTerm = fmaf(signalValue * signalValue, weighted.y, squareTerm);
            }
        }
    }

    // Sum over the G lanes of each window (every lane takes part in the shuffles)
    #pragma unroll
    for (int offset = G / 2; offset > 0; offset >>= 1) {
        crossTerm += __shfl_xor_sync(0xffffffff, crossTerm, offset);
        squareTerm += __shfl_xor_sync(0xffffffff, squareTerm, offset);
    }
    if (active && laneInWindow == 0) {
        float amplitude = meanDepth * overshoot[durationIndex] * 2.0f;  // 2 = 1 / SIGNAL_DEPTH
        float delta = amplitude * (amplitude * squareTerm - 2.0f * crossTerm);
        key = ((unsigned long long)orderedKey(delta) << 32)
              | (unsigned int)(selectedIndex * tSize + windowStart);
    }

    // Block-wide minimum, then one atomic per block
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long neighborKey = __shfl_down_sync(0xffffffff, key, offset);
        key = neighborKey < key ? neighborKey : key;
    }
    __shared__ unsigned long long warpMin[32];
    int laneIndex = threadIdx.x & 31;
    int warpIndex = threadIdx.x >> 5;
    if (laneIndex == 0) {
        warpMin[warpIndex] = key;
    }
    __syncthreads();
    if (warpIndex == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        key = laneIndex < nWarps ? warpMin[laneIndex] : 0xffffffffffffffffull;
        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned long long neighborKey = __shfl_down_sync(0xffffffff, key, offset);
            key = neighborKey < key ? neighborKey : key;
        }
        if (laneIndex == 0 && key != 0xffffffffffffffffull) {
            atomicMin(best + periodIndex, key);
        }
    }
}

// searchChi2G1 ... searchChi2G32: G threads per window
#define SEARCH_CHI2(G)                                                                            \
    extern "C" __global__ void searchChi2G##G(                                                    \
            unsigned long long* best, const long long* cumResidual, float invScale,               \
            const float2* patchedWeighted, const float* signal, const float* overshoot, const int* widths,   \
            const int* selected, const int* offsets, const int* strides, const bool* periodMask,  \
            int yBegin, int yEnd, int nDurations, int tSize, int patchedSize, int maxWidth,       \
            float transitDepthMin) {                                                              \
        searchChi2Lanes<G>(best, cumResidual, invScale, patchedWeighted, signal, overshoot, widths,   \
                           selected, offsets, strides, periodMask, yBegin, yEnd, nDurations,      \
                           tSize, patchedSize, maxWidth, transitDepthMin);                        \
    }
SEARCH_CHI2(1)
SEARCH_CHI2(2)
SEARCH_CHI2(4)
SEARCH_CHI2(8)
SEARCH_CHI2(16)
SEARCH_CHI2(32)

/**
 * Q adjacent windows share one stream of (w*r, w). Their overlapping interior
 * loads each input point once, while retaining both sums for arbitrary weights.
 * Each physical lane accumulates one residue class modulo G in every window.
 * Rotating those accumulators back to the original lanes before the XOR sum
 * preserves the original per-lane FMA order and reduction tree, including ties.
 * offsets counts groups of Q windows separately for each selected duration.
 */
__device__ __forceinline__ void storeAdjacentBest(unsigned long long key,
                                                   unsigned long long* best, int period) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffff, key, offset);
        key = other < key ? other : key;
    }
    __shared__ unsigned long long warpMin[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warpMin[warp] = key;
    __syncthreads();
    if (warp == 0) {
        key = lane < blockDim.x / 32 ? warpMin[lane] : 0xffffffffffffffffull;
        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned long long other = __shfl_down_sync(0xffffffff, key, offset);
            key = other < key ? other : key;
        }
        if (lane == 0 && key != 0xffffffffffffffffull) atomicMin(best + period, key);
    }
}

#define REUSE_ARGS unsigned long long* best, const long long* cumResidual, float invScale, \
    const float2* patchedWeighted, const float* signal, const float* overshoot, const int* widths, \
    const int* selected, const int* offsets, const int* strides, const bool* periodMask, \
    int yBegin, int yEnd, int nDurations, int tSize, int patchedSize, int maxWidth, float transitDepthMin
#define REUSE_PASS best, cumResidual, invScale, patchedWeighted, signal, overshoot, widths, \
    selected, offsets, strides, periodMask, yBegin, yEnd, nDurations, tSize, patchedSize, \
    maxWidth, transitDepthMin

template<int G, int Q>
__device__ __forceinline__ void searchAdjacentWindows(REUSE_ARGS) {
    int period = blockIdx.y;
    int lane = threadIdx.x % G;
    int group = offsets[yBegin] + (blockIdx.x * blockDim.x + threadIdx.x) / G;
    float cross[Q] = {}, square[Q] = {}, depth[Q] = {};
    bool active[Q] = {};
    int selectedIndex = yBegin, start = 0, duration = 0, stride = 0, width = 0;
    bool any = false;
    if (group < offsets[yEnd]) {
        while (offsets[selectedIndex + 1] <= group) ++selectedIndex;
        duration = selected[selectedIndex];
        stride = strides[selectedIndex];
        width = widths[duration];
        start = (group - offsets[selectedIndex]) * Q * stride;
        if (periodMask[(long long)period * nDurations + duration]) {
            const long long* row = cumResidual + (long long)period * patchedSize;
            #pragma unroll
            for (int q = 0; q < Q; ++q) {
                int ws = start + q * stride;
                if (ws < tSize) {
                    long long sum = row[ws + width - 1] - (ws > 0 ? row[ws - 1] : 0);
                    depth[q] = __fdiv_rn(__ll2float_rn(sum) * invScale, (float)width);
                    active[q] = depth[q] > transitDepthMin;
                    any |= active[q];
                }
            }
        }
        if (any) {
            const float* h = signal + (long long)duration * maxWidth;
            const float2* data = patchedWeighted + (long long)period * patchedSize + start;
            bool all = true;
            #pragma unroll
            for (int q = 0; q < Q; ++q) all &= active[q];
            if (all) {
                // The common interior has no per-point activity or bounds branches.
                int j = lane;
                for (; j < min(width, (Q - 1) * stride); j += G) {
                    float2 v = start + j < patchedSize ? data[j] : make_float2(0, 0);
                    #pragma unroll
                    for (int q = 0; q < Q; ++q) {
                        int k = j - q * stride;
                        if (active[q] && k >= 0 && k < width) {
                            float s = h[k];
                            cross[q] = fmaf(s, v.x, cross[q]);
                            square[q] = fmaf(s * s, v.y, square[q]);
                        }
                    }
                }
                for (; j < width; j += G) {
                    float2 v = data[j];
                    #pragma unroll
                    for (int q = 0; q < Q; ++q) {
                        float s = h[j - q * stride];
                        cross[q] = fmaf(s, v.x, cross[q]);
                        square[q] = fmaf(s * s, v.y, square[q]);
                    }
                }
                for (; j < width + (Q - 1) * stride; j += G) {
                    float2 v = start + j < patchedSize ? data[j] : make_float2(0, 0);
                    #pragma unroll
                    for (int q = 0; q < Q; ++q) {
                        int k = j - q * stride;
                        if (active[q] && k >= 0 && k < width) {
                            float s = h[k];
                            cross[q] = fmaf(s, v.x, cross[q]);
                            square[q] = fmaf(s * s, v.y, square[q]);
                        }
                    }
                }
            } else {
                // Stream the union once. For window q, lane ownership rotates by q*S.
                // Undo that rotation before the original XOR reduction, preserving FP32 order.
                for (int j = lane; j < width + (Q - 1) * stride; j += G) {
                    bool needed = false;
                    #pragma unroll
                    for (int q = 0; q < Q; ++q)
                        needed |= active[q] && j >= q * stride && j < width + q * stride;
                    if (needed) {
                        float2 v = data[j];
                        #pragma unroll
                        for (int q = 0; q < Q; ++q) {
                            int k = j - q * stride;
                            if (active[q] && k >= 0 && k < width) {
                                float s = h[k];
                                cross[q] = fmaf(s, v.x, cross[q]);
                                square[q] = fmaf(s * s, v.y, square[q]);
                            }
                        }
                    }
                }
            }
        }
    }
    unsigned long long key = 0xffffffffffffffffull;
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        cross[q] = __shfl_sync(0xffffffff, cross[q], (lane + q * stride) % G, G);
        square[q] = __shfl_sync(0xffffffff, square[q], (lane + q * stride) % G, G);
        #pragma unroll
        for (int offset = G / 2; offset > 0; offset >>= 1) {
            cross[q] += __shfl_xor_sync(0xffffffff, cross[q], offset);
            square[q] += __shfl_xor_sync(0xffffffff, square[q], offset);
        }
        if (active[q] && lane == 0) {
            float amplitude = depth[q] * overshoot[duration] * 2.0f;
            float delta = amplitude * (amplitude * square[q] - 2.0f * cross[q]);
            unsigned long long candidate = ((unsigned long long)orderedKey(delta) << 32)
                | (unsigned int)(selectedIndex * tSize + start + q * stride);
            key = candidate < key ? candidate : key;
        }
    }
    storeAdjacentBest(key, best, period);
}


#define SEARCH_ADJACENT(G,Q) \
    extern "C" __global__ void searchChi2AdjacentG##G(REUSE_ARGS) { \
        searchAdjacentWindows<G,Q>(REUSE_PASS); \
    }
SEARCH_ADJACENT(8,2)
SEARCH_ADJACENT(16,2)
SEARCH_ADJACENT(32,4)
#undef SEARCH_ADJACENT
#undef REUSE_ARGS
#undef REUSE_PASS

// Inclusive first/double prefixes of the actual stored float2 data, in FP64.
// At index i: (D_z[i+2], D_w[i+2], S_abs_z[i+1], S_w[i+1]).
extern "C" __global__ void scanScreening(const float2* weighted, double4* prefix, int size) {
    long long row = (long long)blockIdx.x * size;
    double zCarry = 0, wCarry = 0, absCarry = 0, dzCarry = 0, dwCarry = 0;
    for (int tile = 0; tile < size; tile += blockDim.x) {
        int i = tile + threadIdx.x;
        float2 v = i < size ? weighted[row + i] : make_float2(0, 0);
        double zTotal, wTotal, absTotal, dzTotal, dwTotal;
        double z = zCarry + blockInclusiveScan((double)v.x, &zTotal);
        double w = wCarry + blockInclusiveScan((double)v.y, &wTotal);
        double az = absCarry + blockInclusiveScan(fabs((double)v.x), &absTotal);
        double dz = dzCarry + blockInclusiveScan(z, &dzTotal);
        double dw = dwCarry + blockInclusiveScan(w, &dwTotal);
        if (i < size) prefix[row + i] = make_double4(dz, dw, az, w);
        zCarry += zTotal;
        wCarry += wTotal;
        absCarry += absTotal;
        dzCarry += dzTotal;
        dwCarry += dwTotal;
    }
}

extern "C" __global__ void screenCandidates(
        float* lower, unsigned long long* upperKeys, const double4* prefix,
        const long long* residual, float invScale, const float* overshoot,
        const int* widths, const int* selected, const int* offsets, const int* strides,
        const int* lanes, const bool* mask, const int* positions, const double2* coefficients,
        const double* metadata, double absMassError, double weightMassError,
        int nSelected, int nDurations, int tSize, int size, float depthMin) {
    int period = blockIdx.y;
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    int count = offsets[nSelected];
#if GTLS_CHEAP_SCREEN_BOUNDS
    // A valid upper bound can only decrease. Sample it once per block so a cheap
    // absolute-mass bound can skip the sparse template dot products. An empty
    // key decodes to NaN and disables this shortcut for the first wave of blocks.
    __shared__ unsigned long long earlierUpperKey;
    if (threadIdx.x == 0) earlierUpperKey = atomicAdd(upperKeys + period, 0ull);
    __syncthreads();
    unsigned int earlierHigh = (unsigned int)(earlierUpperKey >> 32);
    float earlierUpper = __uint_as_float(earlierHigh >= 0x80000000u
                                         ? earlierHigh ^ 0x80000000u : ~earlierHigh);
#endif
    unsigned long long key = 0xffffffffffffffffull;
    if (item < count) {
        float lo = -__int_as_float(0x7f800000);
        int d = 0;
        while (offsets[d + 1] <= item) ++d;
        int duration = selected[d], width = widths[duration];
        int start = (item - offsets[d]) * strides[d];
        if (!mask[(long long)period*nDurations + duration]) {
            lo = __int_as_float(0x7f800000);
        } else {
            const long long* r = residual + (long long)period*size;
            long long sum = r[start+width-1] - (start ? r[start-1] : 0);
            float depth = __fdiv_rn(__ll2float_rn(sum)*invScale, (float)width);
            if (!(depth > depthMin)) {
                lo = __int_as_float(0x7f800000);
            } else {
                const double* m = metadata + duration*8;
                int begin = (int)m[0], terms = (int)m[1];
                if (terms) {
                    const double4* p = prefix + (long long)period*size;
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    double ar = 0, a2 = 0;
                    for (int j = begin; j < begin+terms; ++j) {
                        int k = start + positions[j] - 2;
                        double4 v = k >= 0 ? p[k] : make_double4(0,0,0,0);
                        double2 c = coefficients[j];
                        ar = fma(c.x, v.x, ar);
                        a2 = fma(c.y, v.y, a2);
                    }
#endif
                    double4 end = p[start+width-1];
                    double4 first = start ? p[start-1] : make_double4(0,0,0,0);
                    double az = fmax(0.0, end.z-first.z) + absMassError;
                    double w = fmax(0.0, end.w-first.w) + weightMassError;
                    int g = lanes[d];
                    int steps = (width+g-1)/g + (__ffs(g)-1) + 2;
                    double gu = steps*0x1p-24;
                    double gamma = gu < 0.5 ? gu/(1-gu) : __int_as_float(0x7f800000);
                    // The original FMA chains + XOR tree, including subnormal rounding.
                    double tiny = (double)width*(steps+8)*0x1p-149;
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    double eAr = m[2]*az + m[6] + gamma*m[4]*az + tiny;
                    double eA2 = m[3]*w + m[7] + gamma*m[5]*w + tiny;
#endif
                    float amplitude = depth*overshoot[duration]*2.0f;
                    double a = amplitude, aa = fabs(a);
                    double arBound = (1+gamma)*m[4]*az + tiny;
                    double a2Bound = (1+gamma)*m[5]*w + tiny;
                    double finalRound = 0x1p-21*(a*a*a2Bound + 2*aa*arBound) + tiny;
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    double proxy = a*(a*a2 - 2*ar);
                    double error = (a*a*eA2 + 2*aa*eAr + finalRound)*1.0000000001 + 0x1p-140;
                    double hi = proxy+error;
                    double low = proxy-error;
#endif
                    double innerBound = aa*a2Bound + 2*arBound;
                    bool noOverflow = arBound <= 0x1.fffffep+126 && a2Bound <= 0x1.fffffep+127
                        && innerBound <= 0x1.fffffep+127 && aa*innerBound <= 0x1.fffffep+127;
#if GTLS_CHEAP_SCREEN_BOUNDS
                    // A2 is nonnegative for arbitrary positive weights. Discard
                    // that term and bound |AR| with the same original-FP32 error
                    // allowance. Outward slack also covers the FP64 operations.
                    double cheap = (-2*aa*arBound-finalRound)*1.0000000001-0x1p-140;
                    float cheapLower = __double2float_rd(cheap);
                    if (noOverflow && isfinite(amplitude) && cheapLower > earlierUpper) {
                        lo = cheapLower;
                    } else {
                        double ar = 0, a2 = 0;
                        for (int j = begin; j < begin+terms; ++j) {
                            int k = start + positions[j] - 2;
                            double4 v = k >= 0 ? p[k] : make_double4(0,0,0,0);
                            double2 c = coefficients[j];
                            ar = fma(c.x, v.x, ar);
                            a2 = fma(c.y, v.y, a2);
                        }
                        double eAr = m[2]*az + m[6] + gamma*m[4]*az + tiny;
                        double eA2 = m[3]*w + m[7] + gamma*m[5]*w + tiny;
                        double proxy = a*(a*a2 - 2*ar);
                        double error = (a*a*eA2 + 2*aa*eAr + finalRound)*1.0000000001 + 0x1p-140;
                        double hi = proxy+error;
                        double low = proxy-error;
                        // Unbounded/nonfinite certificates keep the original exact path.
                        if (noOverflow && isfinite(hi) && isfinite(low) && isfinite(amplitude)) {
                            lo = __double2float_rd(low);
                            float upper = __double2float_ru(hi);
                            key = ((unsigned long long)orderedKey(upper) << 32) | (unsigned int)item;
                        }
#else
                    // Unbounded/nonfinite certificates keep the original exact path.
                    if (noOverflow && isfinite(hi) && isfinite(low) && isfinite(amplitude)) {
                        lo = __double2float_rd(low);
                        float upper = __double2float_ru(hi);
                        key = ((unsigned long long)orderedKey(upper) << 32) | (unsigned int)item;
#endif
                    }
                }
            }
        }
        lower[(long long)period*count+item] = lo;
    }
    storeAdjacentBest(key, upperKeys, period);
}

#define SCREENED_EXACT(G) \
    extern "C" __global__ void searchChi2ScreenedG##G( \
        unsigned long long* best, const long long* cumResidual, float invScale, \
        const float2* patchedWeighted, const float* signal, const float* overshoot, const int* widths, \
        const int* selected, const int* offsets, const int* strides, const bool* periodMask, \
        int yBegin, int yEnd, int nDurations, int tSize, int patchedSize, int maxWidth, float transitDepthMin, \
        const float* lowerBounds, const unsigned long long* upperKeys, int nBounds) { \
        searchChi2Lanes<G>(best, cumResidual, invScale, patchedWeighted, signal, overshoot, widths, \
            selected, offsets, strides, periodMask, yBegin, yEnd, nDurations, tSize, patchedSize, \
            maxWidth, transitDepthMin, lowerBounds, upperKeys, nBounds); \
    }
SCREENED_EXACT(1)
SCREENED_EXACT(2)
SCREENED_EXACT(4)
SCREENED_EXACT(8)
SCREENED_EXACT(16)
SCREENED_EXACT(32)
#undef SCREENED_EXACT
