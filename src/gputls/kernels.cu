extern "C"{
    // Configuration constants
    #define SKIP_POINT 8
    
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
    #define FLOAT_INFINITY __int_as_float(0x7f800000) // IEEE-754 float infinity
    #define SCALE_FACTOR 1000000000000000.0    // Scale factor for duration calculations

    /**
     * Fast phase folding kernel with optimized memory access
     * Calculates phase = (time / period) - floor(time / period) for each time point
     */
    __global__ void foldFast(const double* time, const double* periods, double* phase, int* timeSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (tid < (*timeSize)) {
            double time_val = time[tid];
            double period = periods[y];
            double phase_raw = time_val / period;
            phase[tid + y * (*timeSize)] = phase_raw - (int)(phase_raw);
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
     * Optimized data patching kernel with improved memory coalescing
     */
    __global__ void patchData(float *in_patchedData, float *in_patchedDys,
                             int *patchedDataSize, int *in_sortIndex, int *maxDuration,
                             float *flux, float *dy, int *tSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // patchedData index
        int y = blockIdx.y * blockDim.y + threadIdx.y;   // period index

        float *patchedData = in_patchedData + y * (*patchedDataSize);
        float *patchedDys = in_patchedDys + y * (*patchedDataSize);
        int *sortIndex = in_sortIndex + y * (*tSize);

        if (tid < (*tSize)) {
            int src_idx = sortIndex[tid];
            patchedData[tid] = flux[src_idx];
            patchedDys[tid] = dy[src_idx];
        } else if (tid < (*tSize + *maxDuration)) {
            int src_idx = sortIndex[tid - (*tSize)];
            patchedData[tid] = flux[src_idx];
            patchedDys[tid] = dy[src_idx];
        }
    }

    /**
     * Optimized inverse squared calculation
     */
    __global__ void calcInverseSquaredPatchedDy(float *out, float *patched_dys, int *patched_data_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (tid < *patched_data_size) {
            float dy_val = patched_dys[tid + y * (*patched_data_size)];
            // Prevent division by zero: use minimum threshold
            float dy_squared = dy_val * dy_val;
            if (dy_squared < 1e-10f) {
                dy_squared = 1e-10f;
            }
            out[tid + y * (*patched_data_size)] = 1.0f / dy_squared;
        }
    }

    /**
     * Optimized edge effect correction calculation
     */
    __global__ void calcEdgeEffectCorrections(float *out, float *patch_data,
                                             float* inverse_squared_patched_dys, int *patched_data_size,
                                             int* maxDuration, int* period_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid >= *period_size) {
            return;
        }

        float* patched_data = patch_data + tid * (*patched_data_size);
        float* inverse_squared_patched_dy = inverse_squared_patched_dys + tid * (*patched_data_size);

        double edgeEffect = 0.0;
        int start_idx = (*patched_data_size) - (*maxDuration);
        
        for (int j = start_idx; j < (*patched_data_size); j++) {
            double patchDataJ = (double)(patched_data[j]);
            double patchDataDyJ = (double)(inverse_squared_patched_dy[j]);
            edgeEffect += (1.0 + patchDataJ * patchDataJ - 2.0 * patchDataJ) * patchDataDyJ;
        }
        out[tid] = edgeEffect;
    }

    /**
     * Optimized full sum calculation with better memory access patterns
     */
    __global__ void calcAllFullSum(float* fullsums, float *in_patched_data,
                                  float *in_inverse_squared_patched_dy, int *patched_data_size,
                                  int *in_duration, int *duration_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // duration index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // period index

        float *patched_data = in_patched_data + y * (*patched_data_size);
        float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y * (*patched_data_size);

        // Calculate full sum once
        float fullsum = 0.0f;
        for (int i = 0; i < *patched_data_size; i++) {
            float diff = 1.0f - patched_data[i];
            fullsum += diff * diff * inverse_squared_patched_dy[i];
        }

        if (tid < (*duration_size)) {
            int window = in_duration[tid];
            float window_sum = 0.0f;
            
            // Calculate window sum efficiently
            for (int i = 0; i < window; i++) {
                float diff = 1.0f - patched_data[i];
                window_sum += diff * diff * inverse_squared_patched_dy[i];
            }
            
            fullsums[tid + y * (*duration_size)] = fullsum - window_sum;
        }
    }
    
    /**
     * Optimized out-of-transit residuals calculation - Step 1
     */
    __global__ void calcAllOutOfTransitResiduals_step1_2GPU(float *temp_ootr,
                                                            float *in_patched_data, int *in_duration, int *duration_size,
                                                            float *in_inverse_squared_patched_dy, int *patched_data_size,
                                                            int *resultArrayXAxisSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        if (tid >= *resultArrayXAxisSize || y >= *duration_size) {
            return;
        }

        float *patched_data = in_patched_data + (long long)z * (*patched_data_size);
        float *inverse_squared_patched = in_inverse_squared_patched_dy + (long long)z * (*patched_data_size);
        int window = in_duration[y];
        
        int becomes_visible = tid;
        int becomes_invisible = tid + window;
        
        float weight = 0.0f;
        if (becomes_visible < *patched_data_size && becomes_invisible < *patched_data_size) {
            float diff_visible = 1.0f - patched_data[becomes_visible];
            float diff_invisible = 1.0f - patched_data[becomes_invisible];
            
            float add_visible_left = diff_visible * diff_visible * inverse_squared_patched[becomes_visible];
            float remove_invisible_right = diff_invisible * diff_invisible * inverse_squared_patched[becomes_invisible];
            weight = add_visible_left - remove_invisible_right;
        }
        
        temp_ootr[(long long)tid + (long long)y * (*resultArrayXAxisSize) + (long long)z * (*resultArrayXAxisSize) * (*duration_size)] = weight;
    }

    /**
     * Optimized out-of-transit residuals calculation - Step 2
     */
    __global__ void calcAllOutOfTransitResiduals_step2_2GPU(float *in_ootr, int *duration_size,
                                                            int *resultArrayXAxisSize, float *in_fullsum) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;      // point index
        int tid = blockIdx.y * blockDim.y + threadIdx.y;    // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        if (i >= (*resultArrayXAxisSize) || tid >= (*duration_size)) {
            return;
        }

        float *fullsum = in_fullsum + (long long)z * (*duration_size);
        float *ootr = in_ootr + (long long)z * (*resultArrayXAxisSize) * (*duration_size);
        
        ootr[(long long)i + (long long)tid * (*resultArrayXAxisSize)] = fullsum[tid] + ootr[(long long)i + (long long)tid * (*resultArrayXAxisSize)];
    }

    /**
     * Out-of-transit residuals, stage 1: per-point squared error (1 - flux)^2 / dy^2.
     * Its prefix sum (computed on the host with cupy.cumsum) feeds calculate_final_ootr.
     */
    __global__ void calculate_base_error(
        float* out_base_error, // Shape: (num_periods, patched_data_size)
        const float* in_patched_data,
        const float* in_inverse_squared_patched_dy,
        int patched_data_size,
        int num_periods
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // point index
        int z = blockIdx.y; // period index

        if (z >= num_periods || tid >= patched_data_size) {
            return;
        }

        const float* patched_data_period = in_patched_data + z * patched_data_size;
        const float* inverse_squared_patched_period = in_inverse_squared_patched_dy + z * patched_data_size;

        float diff = 1.0f - patched_data_period[tid];
        out_base_error[tid + z * patched_data_size] = diff * diff * inverse_squared_patched_period[tid];
    }

    /**
     * Out-of-transit residuals, stage 2: out-of-transit residual for every
     * (period, duration, start point) from the error prefix sum.
     */
    __global__ void calculate_final_ootr(
        float* out_ootr,
        const float* error_prefix_sum, // PRE-COMPUTED. Shape: (num_periods, patchedDatasSize)
        const float* in_fullsum,
        const int* in_duration,
        int tSize,                 // The analysis length (<= patchedDatasSize)
        int patchedDatasSize,      // The full data length
        int duration_size,
        int num_periods
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x; // point index
        int d = blockIdx.y;                           // duration index
        int z = blockIdx.z;                           // period index

        // Only compute for the required analysis length 'tSize'
        if (p >= tSize || d >= duration_size || z >= num_periods) {
            return;
        }

        // Pointer to the current period's prefix sum array (length is patchedDatasSize)
        const float* P_E = error_prefix_sum + (long long)z * patchedDatasSize; 

        int window = in_duration[d];

        // --- Core Telescoping Sum Logic with FAITHFUL BOUNDARY CHECKS ---
        
        // P_E[p] is always in bounds due to the check above
        float p_e_p = P_E[p];

        // Check bounds for P_E[p + window]
        float p_e_p_plus_window = (p + window < patchedDatasSize) ? P_E[p + window] : 0.0f;

        // Check bounds for P_E[window - 1]
        float p_e_window_minus_1 = (window > 0 && window - 1 < patchedDatasSize) ? P_E[window - 1] : 0.0f;

        float cumsum_weight = p_e_p - (p_e_p_plus_window - p_e_window_minus_1);
        
        // --- Finalization Step ---
        float fullsum_val = in_fullsum[(long long)d + (long long)z * duration_size];

        long long final_idx = (long long)p + (long long)d * tSize + (long long)z * tSize * duration_size;
        out_ootr[final_idx] = fullsum_val + cumsum_weight;
    }

    /**
     * Optimized device function for cumsum average calculation
     */
    __device__ float calcAverageFromCumsum(float *inPatchedDataCumsum, int duration, 
                                          int *patched_data_size, int tid, int z) {
        if (tid == 0) {
            return 1.0f - inPatchedDataCumsum[z * (*patched_data_size) + duration - 1] / duration;
        } else {
            float end_val = inPatchedDataCumsum[z * (*patched_data_size) + tid + duration - 1];
            float start_val = inPatchedDataCumsum[z * (*patched_data_size) + tid - 1];
            return 1.0f - (end_val - start_val) / duration;
        }
    }

    /**
     * Coarse search: chi2 for every (period, duration, start point), evaluating
     * only every SKIP_POINT-th start point (the others are set to infinity).
     */
    __global__ void calcAllLowestResidualsSkip(
        float *out, int *resultArrayXAxisSize,
        float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
        float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
        float *in_overshoot, float *in_ootr, float *in_fullsum,
        float *in_summed_edge_effect_correction, float *cumsumGPU,
        float *in_transit_depth_min
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y;
        int z = blockIdx.z;

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        int duration = in_duration[y];
        int skipPoint = (duration > SKIP_POINT) ? (duration / SKIP_POINT) : 1;
        float transit_depth_min = *in_transit_depth_min;
        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, z);

        float current_stat = FLOAT_INFINITY;

        if (calc_mean > transit_depth_min && tid % skipPoint == 0) {
            float ootr = (tid == 0) ? 
                in_fullsum[(long long)z * (*in_duration_size) + y] :
                in_ootr[(long long)y * (*resultArrayXAxisSize) + (long long)z * (*resultArrayXAxisSize) * (*in_duration_size) + tid - 1];

            float *data = in_patched_datas + (long long)z * (*in_patched_datas_size) + tid;
            float *dy = in_inverse_squared_patched_dys + (long long)z * (*in_patched_datas_size) + tid;
            float *signal = in_signal + (long long)y * (*in_max_signal_x_size);
            
            float reverse_scale = calc_mean * in_overshoot[y] * 2.0f;
            float summed_edge_effect_correction = in_summed_edge_effect_correction[z];
            float intransit_residual = 0.0f;

            for (int i = 0; i < duration; i++) {
                float sigi = signal[i] * reverse_scale;
                float loss = data[i] - (1.0f - sigi);
                intransit_residual += loss * loss * dy[i];
            }

            current_stat = intransit_residual + ootr - summed_edge_effect_correction;
        }

        out[(long long)tid + (long long)y * (*resultArrayXAxisSize) + (long long)z * (*resultArrayXAxisSize) * (*in_duration_size)] = current_stat;
    }

    /**
     * Refinement search: chi2 for every (period, duration, start point) without skipping.
     */
    __global__ void calcAllLowestResidualsNoSkip(float *out, int *resultArrayXAxisSize,
                                                         float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
                                                         float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
                                                         float *in_overshoot, float *in_ootr, float *in_fullsum,
                                                         float *in_summed_edge_effect_correction, float *cumsumGPU,
                                                         float *in_transit_depth_min) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        int durationIndex = y;
        int periodIndex = z;
        int duration = in_duration[durationIndex];

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        float transit_depth_min = *in_transit_depth_min;

        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, periodIndex);
        float overshoot = in_overshoot[durationIndex];

        if (calc_mean > transit_depth_min) {
            // Optimized OOTR calculation
            float ootr = (tid == 0) ? 
                in_fullsum[periodIndex * (*in_duration_size) + durationIndex] :
                in_ootr[durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size) + tid - 1];

            // Pre-calculate pointers for better memory access
            float *data = in_patched_datas + periodIndex * (*in_patched_datas_size) + tid;
            float *signal = in_signal + durationIndex * (*in_max_signal_x_size);
            float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + periodIndex * (*in_patched_datas_size);
            float summed_edge_effect_correction = in_summed_edge_effect_correction[periodIndex];
            float *dy = inverse_squared_patched_dy_arr + tid;

            // Optimized scale calculation with proper float type
            float reverse_scale = calc_mean * overshoot * 2.0f;  // 2.0f for SIGNAL_DEPTH = 0.5

            float intransit_residual = 0.0f;

            // Optimized inner loop with reduced variable declarations
            for (int i = 0; i < duration; i++) {
                float sigi = signal[i] * reverse_scale;
                float loss = data[i] - (1.0f - sigi);
                intransit_residual += loss * loss * dy[i];
            }

            float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
            out[tid + durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size)] = current_stat;
        } else {
            out[tid + durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size)] = FLOAT_INFINITY;
        }
    }

}
