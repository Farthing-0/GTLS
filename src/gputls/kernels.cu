// Block-wide inclusive prefix sum of one value per thread (blockDim.x a multiple of 32).
// Returns the thread's inclusive sum and stores the block total in *total.
template <typename T>
__device__ __forceinline__ T blockInclusiveScan(T v, T* total) {
    __shared__ T warpSums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nWarps = blockDim.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        T other = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) {
            v += other;
        }
    }
    if (lane == 31) {
        warpSums[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        T w = lane < nWarps ? warpSums[lane] : (T)0;
        for (int o = 1; o < 32; o <<= 1) {
            T other = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) {
                w += other;
            }
        }
        warpSums[lane] = w;
    }
    __syncthreads();
    if (warp > 0) {
        v += warpSums[warp - 1];
    }
    *total = warpSums[nWarps - 1];
    __syncthreads();  // warpSums is reused by the next call
    return v;
}

extern "C"{
    #define SECONDS_PER_DAY 86400              // Seconds in a day

    /**
     * Phase sort, stage 1: fold and count the points of each bucket floor(phase * tSize)
     * (about one point per bucket). phase = time / period - floor(time / period).
     */
    __global__ void foldBuckets(const double* time, const double* periods, double* phase,
                                int* counts, int tSize) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // point index
        int z = blockIdx.y;                             // period index
        if (i >= tSize) {
            return;
        }
        double phaseRaw = time[i] / periods[z];
        double p = phaseRaw - (int)(phaseRaw);
        long long o = (long long)z * tSize;
        phase[o + i] = p;
        int bucket = min((int)(p * tSize), tSize - 1);
        atomicAdd(counts + o + bucket, 1);
    }

    /**
     * Exclusive prefix sum of each row of counts (one block per row): the first position of
     * every bucket in the sorted order.
     */
    __global__ void scanCounts(const int* counts, int* cursor, int tSize) {
        long long o = (long long)blockIdx.x * tSize;
        int carry = 0;
        for (int tile = 0; tile < tSize; tile += blockDim.x) {
            int q = tile + threadIdx.x;
            int c = q < tSize ? counts[o + q] : 0;
            int total;
            int inclusive = blockInclusiveScan(c, &total);
            if (q < tSize) {
                cursor[o + q] = carry + inclusive - c;
            }
            carry += total;
        }
    }

    /**
     * Phase sort, stage 2: scatter the point indices into their buckets. cursor holds the
     * first position of each bucket on entry and one past its last position on exit.
     */
    __global__ void bucketScatter(const double* phase, int* cursor, int* sortIndex, int tSize) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int z = blockIdx.y;
        if (i >= tSize) {
            return;
        }
        long long o = (long long)z * tSize;
        int bucket = min((int)(phase[o + i] * tSize), tSize - 1);
        sortIndex[o + atomicAdd(cursor + o + bucket, 1)] = i;
    }

    /**
     * Phase sort, stage 3: insertion sort of each bucket by (phase, index). Buckets increase
     * with phase, so the result equals a stable argsort of the phases.
     */
    __global__ void sortBuckets(const double* phase, const int* counts, const int* ends,
                                int* sortIndex, int tSize) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;  // bucket index
        int z = blockIdx.y;
        if (b >= tSize) {
            return;
        }
        long long o = (long long)z * tSize;
        int n = counts[o + b];
        if (n < 2) {
            return;
        }
        int* seg = sortIndex + o + ends[o + b] - n;
        const double* ph = phase + o;
        for (int a = 1; a < n; a++) {
            int v = seg[a];
            double pv = ph[v];
            int c = a - 1;
            while (c >= 0 && (ph[seg[c]] > pv || (ph[seg[c]] == pv && seg[c] > v))) {
                seg[c + 1] = seg[c];
                c--;
            }
            seg[c + 1] = v;
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
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < (*periodSize)) {
            double length = *tLength;
            int size = *tSize;

            // A transit can be split over the start and end of the data, so the window of
            // the longest duration is widened by (transits + 1) / transits
            double period_days = periods[tid];
            double no_of_transits_naive = length / period_days;
            double correction_factor = (no_of_transits_naive + 1.0) / no_of_transits_naive;

            double period_seconds = period_days * SECONDS_PER_DAY;
            double durationMin = rShort * pow((4.0 * period_seconds) / gmShort, 1.0 / 3.0) / period_seconds;
            double durationMax = rLong * pow((4.0 * period_seconds) / gmLong, 1.0 / 3.0) / period_seconds;
            durationMin = (durationMin > upperLimit) ? upperLimit : durationMin;
            durationMax = (durationMax > upperLimit) ? upperLimit : durationMax;

            int duration_min_in_samples = floor(durationMin * size);
            int duration_max_in_samples = ceil(durationMax * size * correction_factor);
            
            durationsMin[tid] = duration_min_in_samples;
            durationsMax[tid] = duration_max_in_samples;
        }
    }

    /**
     * Optimized duration boolean array generation
     */
    __global__ void durationBool(int* durationsMax, int* durationsMin, const int* durationSize,
                                const int* periodSize, const int* durations, bool* durationBoolArray) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x; // period index
        int y = blockDim.y * blockIdx.y + threadIdx.y;   // duration index

        if (tid < (*periodSize) && y < (*durationSize)) {
            int duration_min = durationsMin[tid];
            int duration_max = durationsMax[tid];
            int current_duration = durations[y];
            
            // Optimized boolean assignment
            durationBoolArray[y + tid * (*durationSize)] = 
                (current_duration >= duration_min && current_duration <= duration_max);
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
        int q = blockIdx.x * blockDim.x + threadIdx.x;  // patched point index
        int z = blockIdx.y;                             // period index
        if (q >= patchedSize) {
            return;
        }
        int src = sortIndex[(long long)z * tSize + q % tSize];
        float r = residual[src];
        float dy2 = dy[src] * dy[src];
        // Prevent division by zero
        if (dy2 < 1e-10f) {
            dy2 = 1e-10f;
        }
        float w = 1.0f / dy2;
        long long o = (long long)z * patchedSize + q;
        patchedResidual[o] = r;
        patchedWeighted[o] = make_float2(w * r, w);
    }

    /**
     * Inclusive float64 prefix sum of each row of the patched residuals (one block per row),
     * and base = sum of w r^2 over the first tSize (not patched) points of the row.
     */
    __global__ void scanResidual(const float* patchedResidual, const float2* patchedWeighted,
                                 double* cumResidual, double* base, int tSize, int patchedSize) {
        long long o = (long long)blockIdx.x * patchedSize;
        double carry = 0.0;
        double chi2 = 0.0;
        for (int tile = 0; tile < patchedSize; tile += blockDim.x) {
            int q = tile + threadIdx.x;
            double r = q < patchedSize ? (double)patchedResidual[o + q] : 0.0;
            if (q < tSize) {
                chi2 += (double)patchedWeighted[o + q].x * r;
            }
            double total;
            double inclusive = blockInclusiveScan(r, &total);
            if (q < patchedSize) {
                cumResidual[o + q] = carry + inclusive;
            }
            carry += total;
        }
        double total;
        blockInclusiveScan(chi2, &total);
        if (threadIdx.x == 0) {
            base[blockIdx.x] = total;
        }
    }

    // Order-preserving map of a float onto an unsigned int: a < b <=> key(a) < key(b)
    __device__ __forceinline__ unsigned int orderedKey(float f) {
        unsigned int u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }

    /**
     * Lowest chi2 of every period over all searched durations and start points, fused into one
     * pass (no per-start-point arrays). For a window of L points starting at `start` with
     * template depth profile s_i and amplitude k = mean_depth * overshoot / SIGNAL_DEPTH, the chi2
     *     sum_{all points} w r^2 - sum_{window} w r^2 + sum_{window} w (r - k s)^2
     * is evaluated as base - 2 k A_R + k^2 A_2 with A_R = sum s_i (w r)_i, A_2 = sum s_i^2 w_i
     * (the algebraic expansion of TLS 2.0). The mean depth comes from a float64 prefix sum of r.
     *
     * Work items of a period are the start points of all searched durations, laid out by
     * `offsets`; start points of duration y are multiples of strides[y]. The searched durations
     * are the union over the batch; a period only tests those of its own row of periodMask
     * (physically plausible durations, see durationBool), so results do not depend on the batch.
     * best[z] receives min over work items of (orderedKey(chi2) << 32 | y * tSize + start), so
     * ties resolve to the first duration, then the first start point.
     */
    __global__ void searchChi2(unsigned long long* best,
                               const double* cumResidual,     // (periods, patchedSize), inclusive
                               const float2* patchedWeighted, // (periods, patchedSize)
                               const double* base,            // (periods)
                               const float* signal,           // (durations, maxWidth)
                               const float* overshoot,        // (durations)
                               const int* widths,             // (durations)
                               const int* selected,           // (nSelected) searched durations
                               const int* offsets,            // (nSelected + 1)
                               const int* strides,            // (nSelected)
                               const bool* periodMask,        // (periods, nDurations)
                               int nSelected, int nDurations, int tSize, int patchedSize,
                               int maxWidth, float transitDepthMin) {
        int z = blockIdx.y;
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long key = 0xffffffffffffffffull;

        if (g < offsets[nSelected]) {
            int y = 0;
            while (offsets[y + 1] <= g) {
                y++;
            }
            int start = (g - offsets[y]) * strides[y];
            int d = selected[y];
            int L = widths[d];
            const double* C = cumResidual + (long long)z * patchedSize;
            double windowSum = C[start + L - 1] - (start > 0 ? C[start - 1] : 0.0);
            float meanDepth = (float)(windowSum / L);

            if (periodMask[(long long)z * nDurations + d] && meanDepth > transitDepthMin) {
                float k = meanDepth * overshoot[d] * 2.0f;  // 2 = 1 / SIGNAL_DEPTH
                const float2* x = patchedWeighted + (long long)z * patchedSize + start;
                const float* s = signal + (long long)d * maxWidth;
                float ar = 0.0f;
                float a2 = 0.0f;
                for (int i = 0; i < L; i++) {
                    float si = s[i];
                    float2 v = x[i];
                    ar = fmaf(si, v.x, ar);
                    a2 = fmaf(si * si, v.y, a2);
                }
                float chi2 = (float)(base[z] + (double)(k * (k * a2 - 2.0f * ar)));
                key = ((unsigned long long)orderedKey(chi2) << 32) | (unsigned int)(y * tSize + start);
            }
        }

        // Block-wide minimum, then one atomic per block
        for (int o = 16; o > 0; o >>= 1) {
            unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
            key = other < key ? other : key;
        }
        __shared__ unsigned long long warpMin[32];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warpMin[warp] = key;
        }
        __syncthreads();
        if (warp == 0) {
            int nWarps = (blockDim.x + 31) >> 5;
            key = lane < nWarps ? warpMin[lane] : 0xffffffffffffffffull;
            for (int o = 16; o > 0; o >>= 1) {
                unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
                key = other < key ? other : key;
            }
            if (lane == 0 && key != 0xffffffffffffffffull) {
                atomicMin(best + z, key);
            }
        }
    }
}
