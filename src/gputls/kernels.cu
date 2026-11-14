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

// Folded phases are nonnegative finite doubles. Their complete IEEE bit strings
// have the same order as the values, so integer comparisons lose no precision.
// Keep small buckets entirely in registers to avoid repeated phase gathers and
// intermediate index writes during insertion sort.
__device__ __forceinline__ void sortSmallBucket(int* points, const double* phase, int count) {
    constexpr int N = 8;
    unsigned long long keys[N];
    int indices[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        indices[i] = i < count ? points[i] : 0x7fffffff;
        keys[i] = i < count ? (unsigned long long)__double_as_longlong(phase[indices[i]])
                            : 0xffffffffffffffffull;
    }
    #pragma unroll
    for (int k = 2; k <= N; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                int other = i ^ j;
                if (other > i) {
                    bool greater = keys[i] > keys[other]
                        || (keys[i] == keys[other] && indices[i] > indices[other]);
                    if ((i & k) == 0 ? greater : !greater) {
                        unsigned long long key = keys[i];
                        keys[i] = keys[other]; keys[other] = key;
                        int index = indices[i];
                        indices[i] = indices[other]; indices[other] = index;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) if (i < count) points[i] = indices[i];
}

extern "C"{
    #define SECONDS_PER_DAY 86400              // Seconds in a day

    /**
     * Phase sort, stage 1: fold and count the points of each bucket floor(phase * nBuckets)
     * (about three points per bucket). phase = time / period - floor(time / period), with the
     * division done as a multiplication by the correctly rounded 1 / period (float64 division
     * is a long instruction sequence, multiplication a single instruction).
     */
    __global__ void foldBuckets(const double* time, const double* inversePeriods, double* phase,
                                int* counts, int tSize, int nBuckets) {
        int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (pointIndex >= tSize) {
            return;
        }
        double cycles = time[pointIndex] * inversePeriods[periodIndex];
        double pointPhase = cycles - (int)(cycles);
        long long rowOffset = (long long)periodIndex * tSize;
        phase[rowOffset + pointIndex] = pointPhase;
        int bucket = min((int)(pointPhase * nBuckets), nBuckets - 1);
        atomicAdd(counts + (long long)periodIndex*nBuckets + bucket, 1);
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
    __global__ void bucketScatter(const double* phase, int* cursor, int* sortIndex, int tSize, int nBuckets) {
        int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (pointIndex >= tSize) {
            return;
        }
        long long rowOffset = (long long)periodIndex * tSize;
        int bucket = min((int)(phase[rowOffset + pointIndex] * nBuckets), nBuckets - 1);
        sortIndex[rowOffset + atomicAdd(cursor + (long long)periodIndex*nBuckets + bucket, 1)] = pointIndex;
    }

    /**
     * Phase sort, stage 3: insertion sort of each bucket by (phase, index). Buckets increase
     * with phase, so the result equals a stable argsort of the phases.
     */
    __global__ void sortBuckets(const double* phase, const int* counts, const int* bucketEnds,
                                int* sortIndex, int tSize, int nBuckets) {
        int bucket = blockIdx.x * blockDim.x + threadIdx.x;
        int periodIndex = blockIdx.y;
        if (bucket >= nBuckets) {
            return;
        }
        long long rowOffset = (long long)periodIndex * tSize;
        int bucketSize = counts[(long long)periodIndex*nBuckets + bucket];
        if (bucketSize < 2) {
            return;
        }
        int* bucketPoints = sortIndex + rowOffset + bucketEnds[(long long)periodIndex*nBuckets + bucket] - bucketSize;
        const double* rowPhase = phase + rowOffset;
        if (bucketSize <= 8) {
            sortSmallBucket(bucketPoints, rowPhase, bucketSize);
            return;
        }
        for (int next = 1; next < bucketSize; next++) {
            int keyPoint = bucketPoints[next];
            unsigned long long keyPhase = __double_as_longlong(rowPhase[keyPoint]);
            int slot = next - 1;
            while (slot >= 0) {
                int point = bucketPoints[slot];
                unsigned long long value = __double_as_longlong(rowPhase[point]);
                if (value < keyPhase || (value == keyPhase && point <= keyPoint)) break;
                bucketPoints[slot + 1] = point;
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

    // Input-dependent weights are rebuilt once for each search. Interleaving r and
    // w makes the later permutation a single float2 gather per input point.
    __global__ void prepareWeights(const float* residual, const float* dy,
                                   float2* prepared, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;
        float variance = dy[i] * dy[i];
        if (variance < 1e-10f) variance = 1e-10f;
        prepared[i] = make_float2(residual[i], 1.0f / variance);
    }

    /**
     * Gather the folded/extended points, write (w*r,w), and form exact fixed-point
     * residual prefixes in one pass. Integer addition is associative and the scale
     * bounds every partial sum within int64, so two consecutive samples per thread
     * preserve the original scan bit for bit while halving its synchronizations.
     */
    __global__ void patchScan(const int* sortIndex, const float2* prepared,
                             float2* weighted, long long* cumulative, float scale,
                             int tSize, int patchedSize) {
        long long rowOffset = (long long)blockIdx.x * patchedSize;
        long long indexOffset = (long long)blockIdx.x * tSize;
        long long carry = 0;
        for (int tile = 0; tile < patchedSize; tile += 2 * blockDim.x) {
            long long local[2], total = 0;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int i = tile + 2 * threadIdx.x + j;
                if (i < patchedSize) {
                    int source = i;
                    if (source >= tSize) {
                        source -= tSize;
                        if (source >= tSize) source %= tSize;
                    }
                    float2 point = prepared[sortIndex[indexOffset + source]];
                    weighted[rowOffset + i] = make_float2(point.y * point.x, point.y);
                    total += __float2ll_rn(point.x * scale);
                }
                local[j] = total;
            }
            long long tileTotal;
            long long before = blockInclusiveScan(total, &tileTotal) - total;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int i = tile + 2 * threadIdx.x + j;
                if (i < patchedSize) cumulative[rowOffset + i] = carry + before + local[j];
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
 * patchScan), converted and divided by L with float32 rounding: no float64 per window.
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
template <int G, bool Compacted = false>
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
        int nBounds = 0, const int* candidates = nullptr, const int* counts = nullptr,
        int listStride = 0) {
    int periodIndex = blockIdx.y;
    int laneInWindow = threadIdx.x % G;
    int workItem = offsets[yBegin] + (blockIdx.x * blockDim.x + threadIdx.x) / G;
    if (Compacted) {
        int task = (blockIdx.x * blockDim.x + threadIdx.x) / G;
        workItem = task < counts[periodIndex]
            ? candidates[(long long)periodIndex*listStride+task] : offsets[yEnd];
    }
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
// All FP64 (D_z, D_w) rows, followed by the high32 bits of (S_abs_z, S_w).
// Mass storage roundoff is bounded when loading each window; the dot-product
// prefixes and the final statistic retain their existing precision.
// One-warp specialization with exactly the additions of blockInclusiveScan.
__device__ __forceinline__ double warpInclusiveScan(double value, double* total) {
    int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        double neighbor = __shfl_up_sync(0xffffffff, value, offset);
        if (lane >= offset) value += neighbor;
    }
    *total = __shfl_sync(0xffffffff, value, 31);
    return value;
}

extern "C" __global__ void scanScreening(const float2* weighted, double2* prefix, int size) {
    constexpr int ITEMS = 16;
    long long row = (long long)blockIdx.x * size;
    double zCarry = 0, wCarry = 0, absCarry = 0, dzCarry = 0, dwCarry = 0;
    for (int tile = 0; tile < size; tile += ITEMS*blockDim.x) {
        int begin = tile + ITEMS*threadIdx.x;
        double z[ITEMS], w[ITEMS], az[ITEMS], dz[ITEMS], dw[ITEMS];
        double sz = 0, sw = 0, sa = 0, sdz = 0, sdw = 0;
        #pragma unroll
        for (int q = 0; q < ITEMS; ++q) {
            float2 v = begin+q < size ? weighted[row+begin+q] : make_float2(0,0);
            sz += (double)v.x; sw += (double)v.y; sa += fabs((double)v.x);
            sdz += sz; sdw += sw;
            z[q] = sz; w[q] = sw; az[q] = sa; dz[q] = sdz; dw[q] = sdw;
        }
        double zTotal, wTotal, absTotal, dzTotal, dwTotal;
        double beforeZ = zCarry + (warpInclusiveScan(sz, &zTotal)-sz);
        double beforeW = wCarry + (warpInclusiveScan(sw, &wTotal)-sw);
        double beforeA = absCarry + (warpInclusiveScan(sa, &absTotal)-sa);
        double blockDZ = sdz + ITEMS*beforeZ;
        double blockDW = sdw + ITEMS*beforeW;
        double beforeDZ = dzCarry + (warpInclusiveScan(blockDZ, &dzTotal)-blockDZ);
        double beforeDW = dwCarry + (warpInclusiveScan(blockDW, &dwTotal)-blockDW);
        #pragma unroll
        for (int q = 0; q < ITEMS; ++q) {
            if (begin+q < size) {
                prefix[row+begin+q] = make_double2(beforeDZ + (dz[q]+(q+1)*beforeZ),
                                                    beforeDW + (dw[q]+(q+1)*beforeW));
                uint2* mass = (uint2*)(prefix+(long long)gridDim.x*size);
                mass[row+begin+q] = make_uint2(__double2hiint(beforeA+az[q]),
                                             __double2hiint(beforeW+w[q]));
            }
        }
        zCarry += zTotal; wCarry += wTotal; absCarry += absTotal;
        dzCarry += dzTotal; dwCarry += dwTotal;
    }
}

// Equal weights make the template-square statistic independent of the window.
// Only the weighted residual and its absolute mass need prefix storage here.
extern "C" __global__ void scanScreeningUniform(const float2* weighted, double* prefix, int size) {
    constexpr int ITEMS = 16;
    long long row = (long long)blockIdx.x*size;
    double zCarry = 0, absCarry = 0, dzCarry = 0;
    for (int tile = 0; tile < size; tile += ITEMS*blockDim.x) {
        int begin = tile+ITEMS*threadIdx.x;
        double az[ITEMS], dz[ITEMS], sz = 0, sa = 0, sdz = 0;
        #pragma unroll
        for (int q = 0; q < ITEMS; ++q) {
            float v = begin+q < size ? weighted[row+begin+q].x : 0;
            sz += (double)v; sa += fabs((double)v); sdz += sz;
            az[q] = sa; dz[q] = sdz;
        }
        double zTotal, absTotal, dzTotal;
        double beforeZ = zCarry+(warpInclusiveScan(sz, &zTotal)-sz);
        double beforeA = absCarry+(warpInclusiveScan(sa, &absTotal)-sa);
        double blockDZ = sdz+ITEMS*beforeZ;
        double beforeDZ = dzCarry+(warpInclusiveScan(blockDZ, &dzTotal)-blockDZ);
        #pragma unroll
        for (int q = 0; q < ITEMS; ++q) {
            if (begin+q < size) {
                prefix[row+begin+q] = beforeDZ+(dz[q]+(q+1)*beforeZ);
                ((unsigned int*)(prefix+(long long)gridDim.x*size))[row+begin+q]
                    = __double2hiint(beforeA+az[q]);
            }
        }
        zCarry += zTotal; absCarry += absTotal; dzCarry += dzTotal;
    }
}

// Preserve searchChi2Lanes' FP32 FMA sequence and XOR reduction for each G.
extern "C" __global__ void prepareUniformSquare(const float* signal, const int* widths,
    const int* selected, const int* lanes, int maxWidth, float weight, float* squares) {
    int duration = selected[blockIdx.x], g = lanes[blockIdx.x], lane = threadIdx.x;
    float sum = 0;
    if (lane < g) {
        for (int i = lane; i < widths[duration]; i += g) {
            float v = signal[(long long)duration*maxWidth+i];
            sum = fmaf(v*v, weight, sum);
        }
    }
    for (int offset = g/2; offset; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane == 0) squares[duration] = sum;
}

#define SCREEN_ARGS \
    float* lower, unsigned long long* upperKeys, const double2* prefix, \
    const long long* residual, float invScale, const float* overshoot, \
    const int* widths, const int* selected, const int* offsets, const int* strides, \
    const int* lanes, const bool* mask, const int* positions, const double2* coefficients, \
    const double* metadata, double absMassError, double weightMassError, \
    int nSelected, int nDurations, int tSize, int size, float depthMin

template <bool Uniform>
__device__ __forceinline__ double2 screenSparseDot(const double2* p, int start,
    const int* positions, const double2* coefficients, int begin, int terms) {
    double ar = 0, a2 = 0;
    // Prefetch independent prefixes before the dependent FMA chain.
    // Accumulate in the original order, including the scalar tail.
#if __CUDA_ARCH__ >= 700
    constexpr int PREFETCH = 13;
#else
    constexpr int PREFETCH = 4;
#endif
    int j = begin;
    for (; j+PREFETCH <= begin+terms; j += PREFETCH) {
        double2 values[PREFETCH], weights[PREFETCH];
        #pragma unroll
        for (int q=0;q<PREFETCH;++q) {
            int k=start+positions[j+q]-2;
            values[q]=k>=0?(Uniform?make_double2(((const double*)p)[k],0):p[k]):make_double2(0,0);
            weights[q]=coefficients[j+q];
        }
#if __CUDA_ARCH__ >= 700
        // Keep the independent reads ahead of the FMA chain.
        // This mask names only the currently executing lanes;
        // no value or reduction is shared between lanes here.
        __syncwarp(__activemask());
#endif
        #pragma unroll
        for (int q=0;q<PREFETCH;++q) {
            ar=fma(weights[q].x,values[q].x,ar);
            if (!Uniform) a2=fma(weights[q].y,values[q].y,a2);
        }
    }
    for (;j<begin+terms;++j) {
        int k=start+positions[j]-2;
        double2 v=k>=0?(Uniform?make_double2(((const double*)p)[k],0):p[k]):make_double2(0,0);
        double2 c=coefficients[j];
        ar=fma(c.x,v.x,ar);if (!Uniform) a2=fma(c.y,v.y,a2);
    }
    return make_double2(ar, a2);
}

// Per-template rounding terms, evaluated on the GPU with the same arithmetic
// as the candidate path. All hierarchy levels share the same template peaks.
__device__ __forceinline__ double4 screenRoundingBudget(int width, int g) {
    int steps = (width+g-1)/g + (__ffs(g)-1) + 2;
    double gu = steps*0x1p-24;
    double gamma = gu < .5 ? gu/(1-gu) : __int_as_float(0x7f800000);
    double tiny = (double)width*(steps+8)*0x1p-149;
    float gu32 = __fmul_ru(__int2float_ru(steps), 0x1p-24f);
    float gamma32 = gu32 < .5f ? __fdiv_ru(gu32, __fsub_rd(1.0f, gu32)) : __int_as_float(0x7f800000);
    float tiny32 = __fmul_ru(__fmul_ru(__int2float_ru(width), __int2float_ru(steps+8)), 0x1p-149f);
    return make_double4(gamma, tiny, (double)gamma32, (double)tiny32);
}

__device__ __forceinline__ double4 screenFactors(const double* m, double4 r) {
    return make_double4((1+r.x)*m[4], (1+r.x)*m[5], r.x*m[4], r.x*m[5]);
}

__device__ __forceinline__ double2 cheapScreenFactors(const double* m, double4 r) {
    float factor = __fadd_ru(1.0f, (float)r.z);
    return make_double2((double)__fmul_ru(factor, __double2float_ru(m[4])),
                       (double)__fmul_ru(factor, __double2float_ru(m[5])));
}

extern "C" __global__ void prepareScreenBudgets(const int* widths, const int* selected,
    const int* lanes, const double* metadata, double4* budgets, int nSelected, int nDurations) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nSelected) {
        int d = selected[i];
        double4 r = screenRoundingBudget(widths[d], lanes[i]);
        // Three planes: rounding terms, FP64 factors, outward-rounded FP32 factors.
        budgets[d] = r;
        budgets[nDurations+d] = screenFactors(metadata+8*d, r);
        double2 cheap = cheapScreenFactors(metadata+8*d, r);
        budgets[2*nDurations+d] = make_double4(cheap.x, cheap.y, 0, 0);
    }
}

// Drop the nonnegative A2 term for a cheap lower bound. All nonnegative
// quantities are rounded upward, then negated. If FP32 intermediates cannot
// certify that the original statistic is finite, retain the full bound path.
__device__ __forceinline__ float cheapScreenLower32(float amplitude, double az, double w,
                                                   float arFactor, float a2Factor, float tiny) {
    float a = fabsf(amplitude);
    float ar=__fadd_ru(__fmul_ru(arFactor,__double2float_ru(az)),tiny);
    float a2=__fadd_ru(__fmul_ru(a2Factor,__double2float_ru(w)),tiny);
    float inner = __fadd_ru(__fmul_ru(a, a2), __fmul_ru(2.0f, ar));
    float rounding = __fadd_ru(__fmul_ru(0x1p-21f, __fmul_ru(a, inner)), tiny);
    float low = -__fadd_ru(__fmul_ru(__fadd_ru(__fmul_ru(__fmul_ru(2.0f, a), ar), rounding),
                                    0x1.000004p0f), 0x1p-140f);
    bool finite = ar <= 0x1.fffffep+126f && a2 <= 0x1.fffffep+127f
        && inner <= 0x1.fffffep+127f && __fmul_ru(a, inner) <= 0x1.fffffep+127f;
    return finite && isfinite(a) ? low : -__int_as_float(0x7f800000);
}

template <bool Refine, bool Mapped = false, bool Uniform = false>
__device__ __forceinline__ void screenCandidatesImpl(SCREEN_ARGS, const int2* blockMap = nullptr, const double4* budgets = nullptr,
    const float* uniformSquare = nullptr, float uniformWeight = 0) {
    int period = blockIdx.y;
    int d = 0;
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (Mapped) {
        int2 mapped = blockMap[blockIdx.x];
        d = mapped.x;
        item = mapped.y + threadIdx.x;
    }
    int count = offsets[nSelected];
    // Upper bounds only decrease. A stale atomic snapshot can retain extra work,
    // but can never discard a winner. The first pass starts with an empty key.
    __shared__ unsigned long long earlierUpperKey;
    if (threadIdx.x == 0) earlierUpperKey = atomicAdd(upperKeys + period, 0ull);
    __syncthreads();
    unsigned int high = (unsigned int)(earlierUpperKey >> 32);
    float earlierUpper = __uint_as_float(high >= 0x80000000u ? high ^ 0x80000000u : ~high);
    unsigned long long key = 0xffffffffffffffffull;
    int end = Mapped ? offsets[d+1] : count;
    if (item < end && (!Refine || !(lower[(long long)period*count+item] > earlierUpper))) {
        float lo = -__int_as_float(0x7f800000);
        if (!Mapped) while (offsets[d + 1] <= item) ++d;
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
                    double4 rounding = Mapped ? budgets[duration] : screenRoundingBudget(width,lanes[d]);
                    const double2* p = Uniform
                        ? (const double2*)((const double*)prefix+(long long)period*size)
                        : prefix+(long long)period*size;
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    double ar = 0, a2 = 0;
                    if (Refine) {
                        double2 dot = screenSparseDot<Uniform>(p, start, positions, coefficients, begin, terms);
                        ar = dot.x; a2 = Uniform ? (double)uniformSquare[duration] : dot.y;
                    }
#endif
                    double az, w;
                    if (Uniform) {
                        const unsigned int* mass = (const unsigned int*)((const double*)prefix
                            +(long long)gridDim.y*size)+(long long)period*size;
                        double end = __hiloint2double(mass[start+width-1], 0);
                        double first = start ? __hiloint2double(mass[start-1], 0) : 0;
                        double storageError = 0x1p-19*(fabs(end)+fabs(first))+0x1p-1020;
                        az = fmax(0.0, end-first)+absMassError+storageError;
                        w = __dmul_ru((double)uniformWeight, (double)width);
                    } else {
                        const uint2* mass = (const uint2*)(prefix+(long long)gridDim.y*size)+(long long)period*size;
                        uint2 endBits=mass[start+width-1], firstBits=start?mass[start-1]:make_uint2(0,0);
                        double2 end=make_double2(__hiloint2double(endBits.x,0), __hiloint2double(endBits.y,0));
                        double2 first=make_double2(__hiloint2double(firstBits.x,0), __hiloint2double(firstBits.y,0));
                        // Dropping the low32 bits keeps the full FP64 exponent and
                        // avoids slow FP64-to-FP32 conversions. For nonnegative stored
                        // h, truncation error is <= 2^-20*h+2^-1042. Twice that relative
                        // allowance for both endpoints covers FP64 arithmetic too.
                        // Nonfinite inputs still fall back to the original exact path.
                        double absStorageError=0x1p-19*(fabs(end.x)+fabs(first.x))+0x1p-1020;
                        double weightStorageError=0x1p-19*(fabs(end.y)+fabs(first.y))+0x1p-1020;
                        az = fmax(0.0, end.x-first.x) + absMassError + absStorageError;
                        w = fmax(0.0, end.y-first.y) + weightMassError + weightStorageError;
                    }
                    bool discarded = false;
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    // The first pass benefits from the cheap FP32 shortcut on
                    // slow-FP64 devices. Later passes already visit few candidates.
                    if (!Refine) {
                        double2 cheapFactors;
                        if (Mapped) {
                            double4 v = budgets[2*nDurations+duration];
                            cheapFactors = make_double2(v.x, v.y);
                        } else cheapFactors = cheapScreenFactors(m, rounding);
                        float cheap = cheapScreenLower32(depth*overshoot[duration]*2.0f,
                                                         az, w, (float)cheapFactors.x, (float)cheapFactors.y, (float)rounding.w);
                        if (cheap > earlierUpper) { lo = cheap; discarded = true; }
                    }
#endif
                    if (!discarded) {
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    if (!Refine) {
                        double2 dot = screenSparseDot<Uniform>(p, start, positions, coefficients, begin, terms);
                        ar = dot.x; a2 = Uniform ? (double)uniformSquare[duration] : dot.y;
                    }
#endif
                    double tiny = rounding.y;
                    double4 factors = Mapped ? budgets[nDurations+duration] : screenFactors(m, rounding);
#if !GTLS_CHEAP_SCREEN_BOUNDS
                    double eAr = m[2]*az + m[6] + factors.z*az + tiny;
                    double eA2 = Uniform ? 0 : m[3]*w + m[7] + factors.w*w + tiny;
#endif
                    float amplitude = depth*overshoot[duration]*2.0f;
                    double a = amplitude, aa = fabs(a);
                    double arBound = factors.x*az + tiny;
                    double a2Bound = factors.y*w + tiny;
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
                        // Fetch four independent prefixes before the dependent FMA chain.
                        // Accumulate in the original order, including the scalar tail.
                        int j = begin;
                        for (; j+4 <= begin+terms; j += 4) {
                            double2 values[4], weights[4];
                            #pragma unroll
                            for (int q=0;q<4;++q) {
                                int k=start+positions[j+q]-2;
                                values[q]=k>=0?(Uniform?make_double2(((const double*)p)[k],0):p[k]):make_double2(0,0);
                                weights[q]=coefficients[j+q];
                            }
                            #pragma unroll
                            for (int q=0;q<4;++q) {
                                ar=fma(weights[q].x,values[q].x,ar);
                                if (!Uniform) a2=fma(weights[q].y,values[q].y,a2);
                            }
                        }
                        for (;j<begin+terms;++j) {
                            int k=start+positions[j]-2;
                            double2 v=k>=0?(Uniform?make_double2(((const double*)p)[k],0):p[k]):make_double2(0,0);
                            double2 c=coefficients[j];
                            ar=fma(c.x,v.x,ar);if (!Uniform) a2=fma(c.y,v.y,a2);
                        }
                        if (Uniform) a2 = (double)uniformSquare[duration];
                        double eAr = m[2]*az + m[6] + factors.z*az + tiny;
                        double eA2 = Uniform ? 0 : m[3]*w + m[7] + factors.w*w + tiny;
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
        }
        // Every pass is independently conservative; retain the tighter lower bound.
        long long address = (long long)period*count+item;
        lower[address] = Refine ? fmaxf(lower[address], lo) : lo;
    }
    storeAdjacentBest(key, upperKeys, period);
}

extern "C" __global__ void screenCandidates(SCREEN_ARGS) {
    screenCandidatesImpl<false>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin);
}

extern "C" __global__ void refineCandidates(SCREEN_ARGS) {
    screenCandidatesImpl<true>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin);
}

extern "C" __global__ void screenCandidatesMapped(SCREEN_ARGS, const int2* blockMap, const double4* budgets) {
    screenCandidatesImpl<false, true>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin, blockMap, budgets);
}

extern "C" __global__ void refineCandidatesMapped(SCREEN_ARGS, const int2* blockMap, const double4* budgets) {
    screenCandidatesImpl<true, true>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin, blockMap, budgets);
}
extern "C" __global__ void screenCandidatesUniform(SCREEN_ARGS, const int2* blockMap,
    const double4* budgets, const float* uniformSquare, float uniformWeight) {
    screenCandidatesImpl<false, true, true>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin,
        blockMap, budgets, uniformSquare, uniformWeight);
}

extern "C" __global__ void refineCandidatesUniform(SCREEN_ARGS, const int2* blockMap,
    const double4* budgets, const float* uniformSquare, float uniformWeight) {
    screenCandidatesImpl<true, true, true>(lower, upperKeys, prefix, residual, invScale, overshoot, widths,
        selected, offsets, strides, lanes, mask, positions, coefficients, metadata,
        absMassError, weightMassError, nSelected, nDurations, tSize, size, depthMin,
        blockMap, budgets, uniformSquare, uniformWeight);
}
#undef SCREEN_ARGS

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

// Keep the strict original bound comparison, including NaN/overflow fallbacks.
// One atomic reserves each surviving warp's output slice; ordering within the
// compact list does not affect packed-key minima or exact per-window arithmetic.
extern "C" __global__ void compactCandidates(
        const float* lower, const unsigned long long* upperKeys, int* candidates, int* counts,
        int begin, int end, int nBounds, int listStride) {
    int item = begin + blockIdx.x*blockDim.x + threadIdx.x;
    int period = blockIdx.y;
    unsigned int high = upperKeys[period] >> 32;
    float upper = __uint_as_float(high >= 0x80000000u ? high ^ 0x80000000u : ~high);
    bool keep = item < end && !(lower[(long long)period*nBounds+item] > upper);
    unsigned int kept = __ballot_sync(0xffffffff, keep);
    int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == 0 && kept) base = atomicAdd(counts+period, __popc(kept));
    base = __shfl_sync(0xffffffff, base, 0);
    if (keep) candidates[(long long)period*listStride+base+__popc(kept & ((1u << lane)-1))] = item;
}

#define COMPACTED_EXACT(G) \
    extern "C" __global__ void searchChi2CompactedG##G( \
        unsigned long long* best, const long long* cumResidual, float invScale, \
        const float2* patchedWeighted, const float* signal, const float* overshoot, const int* widths, \
        const int* selected, const int* offsets, const int* strides, const bool* periodMask, \
        int yBegin, int yEnd, int nDurations, int tSize, int patchedSize, int maxWidth, float transitDepthMin, \
        const float* lowerBounds, const unsigned long long* upperKeys, int nBounds, \
        const int* candidates, const int* counts, int listStride) { \
        searchChi2Lanes<G, true>(best, cumResidual, invScale, patchedWeighted, signal, overshoot, widths, \
            selected, offsets, strides, periodMask, yBegin, yEnd, nDurations, tSize, patchedSize, \
            maxWidth, transitDepthMin, lowerBounds, upperKeys, nBounds, candidates, counts, listStride); \
    }
COMPACTED_EXACT(1)
COMPACTED_EXACT(2)
COMPACTED_EXACT(4)
COMPACTED_EXACT(8)
COMPACTED_EXACT(16)
COMPACTED_EXACT(32)
#undef COMPACTED_EXACT

// NumPy-compatible FP64 standard deviation for windows of 1..256 points.
// Eight independent accumulators and 128-point recursive splits reproduce
// numpy/_core/src/umath/loops_utils.h.src. Explicit RN operations prevent FMA
// contraction; the outer chunk sums remain on the CPU in their original order.
template<bool Squared>
__device__ __forceinline__ double windowStdValue(const double *x, int i, double mean) {
    double a = x[i];
    if (Squared) {
        a = __dsub_rn(a, mean);
        a = __dmul_rn(a, a);
    }
    return a;
}
template<bool Squared>
__device__ __forceinline__ double windowStdLeaf(const double *x, int n, double mean) {
    if (n < 8) {
        double result = -0.;
        for (int i=0; i<n; ++i) result = __dadd_rn(result, windowStdValue<Squared>(x,i,mean));
        return result;
    }
    double r[8];
    #pragma unroll
    for (int j=0;j<8;++j) r[j]=windowStdValue<Squared>(x,j,mean);
    int i=8;
    for (; i<n-(n%8);i+=8) {
        #pragma unroll
        for (int j=0;j<8;++j) r[j]=__dadd_rn(r[j],windowStdValue<Squared>(x,i+j,mean));
    }
    double result = __dadd_rn(__dadd_rn(__dadd_rn(r[0],r[1]),__dadd_rn(r[2],r[3])),__dadd_rn(__dadd_rn(r[4],r[5]),__dadd_rn(r[6],r[7])));
    for (;i<n;++i) result=__dadd_rn(result,windowStdValue<Squared>(x,i,mean));
    return result;
}
template<bool Squared>
__device__ __forceinline__ double windowStdMiddle(const double *x, int n, double mean) {
    if (n<=128) return windowStdLeaf<Squared>(x,n,mean);
    int half=(n/2)&~7;
    return __dadd_rn(windowStdLeaf<Squared>(x,half,mean),windowStdLeaf<Squared>(x+half,n-half,mean));
}
template<bool Squared>
__device__ __forceinline__ double windowStdPairwise(const double *x, int n, double mean) {
    if (n<=128) return windowStdLeaf<Squared>(x,n,mean);
    int half=(n/2)&~7;
    return __dadd_rn(windowStdMiddle<Squared>(x,half,mean),windowStdMiddle<Squared>(x+half,n-half,mean));
}
extern "C" __global__ void windowStd(const double *x,double *out,int count,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    double mean=__ddiv_rn(windowStdPairwise<false>(x+i,width,0.),double(width));
    double variance=__ddiv_rn(windowStdPairwise<true>(x+i,width,mean),double(width));
    out[i]=__dsqrt_rn(variance);
}
