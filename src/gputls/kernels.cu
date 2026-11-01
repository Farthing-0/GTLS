extern "C"{
    // Physical constants - optimized with appropriate values
    #define R_STAR_MIN 0.05                    // Minimum stellar radius (solar radii) - updated boundary
    #define R_STAR_MAX 4.0                     // Maximum stellar radius (solar radii) - updated boundary
    #define SECONDS_PER_DAY 86400              // Seconds in a day
    #define R_SUN 695508000                    // Radius of the Sun [m]
    #define R_JUP 69911000                     // Radius of Jupiter [m]
    #define FRACTIONAL_TRANSIT_DURATION_MAX 0.15  // Maximum fractional transit duration - updated value

    // Derived constants for duration calculations - optimized values
    #define PI_GM_MAX 416970                   // Simplified pi*G*M_max for duration calc
    #define PI_GM_MIN 20848                    // Simplified pi*G*M_min for duration calc - updated boundary
    #define RS_MIN (R_SUN * R_STAR_MIN)        // Minimum stellar radius in meters
    #define RS_MAX (R_SUN * R_STAR_MAX)        // Maximum stellar radius in meters

    // Transit fitting constants
    #define SCALE_FACTOR 1000000000000000.0    // Scale factor for duration calculations

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
     * Optimized duration grid calculation kernel
     * Based on stellar and planetary parameters with improved performance
     */
    __global__ void durationsGrid(const double* periods, int* durationsMax, int* durationsMin,
                                 const float* tLength, const int* tSize, const int* periodSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (tid < (*periodSize)) {
            float length = *tLength;
            int size = *tSize;
            
            // Calculate transit statistics with optimized operations
            double period_days = periods[tid];
            double no_of_transits_naive = length / period_days;
            double correction_factor = (no_of_transits_naive + 1.0) / no_of_transits_naive;

            double period_seconds = period_days * SECONDS_PER_DAY;
            
            // Pre-calculate common factors for efficiency
            double period_factor_min = (4.0 * period_seconds) / (PI_GM_MIN * SCALE_FACTOR);
            double period_factor_max = (4.0 * period_seconds) / (PI_GM_MAX * SCALE_FACTOR);
            
            // Calculate minimum and maximum transit durations
            double T14Min = RS_MIN * pow(period_factor_min, 1.0 / 3.0);
            double T14Max = (RS_MAX + R_JUP * 2.0) * pow(period_factor_max, 1.0 / 3.0);
            
            double durationMin = T14Min / period_seconds;
            double durationMax = T14Max / period_seconds;
            
            // Apply maximum duration constraints efficiently
            durationMin = (durationMin > FRACTIONAL_TRANSIT_DURATION_MAX) ? 
                         FRACTIONAL_TRANSIT_DURATION_MAX : durationMin;
            durationMax = (durationMax > FRACTIONAL_TRANSIT_DURATION_MAX) ? 
                         FRACTIONAL_TRANSIT_DURATION_MAX : durationMax;
            
            // Convert to sample indices with optimized rounding
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
     * `offsets`; start points of duration y are multiples of strides[y].
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
                               int nSelected, int tSize, int patchedSize, int maxWidth,
                               float transitDepthMin) {
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

            if (meanDepth > transitDepthMin) {
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
