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
        int maxWidth, float transitDepthMin) {
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
