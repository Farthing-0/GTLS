extern "C"{

    __global__ void foldFast(const double* time,const double* periods, double* phase,int* periodSize,int* timeSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if(tid < (*timeSize)){
            phase[tid + y*(*timeSize)] = ((time[tid]) / (periods[y])) - (int)((time[tid]) / (periods[y]));
        }
    }

    __global__ void durationsGrid(const double* periods,int* durationsMax,int* durationsMin,const float* tLength,const int* tSize,const int* periodSize){
        const double R_STAR_MIN = 0.13;
        const double R_STAR_MAX = 3.5;
        const double SECONDS_PER_DAY = 86400;
        const double R_sun = 695508000;// radius of the Sun [m]
        const double R_jup = 69911000;// radius of Jupiter [m]
        const double FRACTIONAL_TRANSIT_DURATION_MAX = 0.12;

        const double RsMin = R_sun * R_STAR_MIN;
        const double RsMax = R_sun * R_STAR_MAX;
        // const double piGMMax = 416970932280661395000; //pi * 6.673e-11 * 1.989 * 10 ** 30 (pi*G*M)
        // const double piGMMin = 41697093228066139500;  //pi * 6.673e-11 * 1.989 * 10 ** 30 * 0.1 (pi*G*M)
        const double piGMMax = 416970;
        const double piGMMin = 41697;

        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if(tid < (*periodSize)){
            float length = *tLength;
            int size = *tSize;
            double no_of_transits_naive = length / periods[tid];
            double no_of_transits_worst = no_of_transits_naive + 1;
            double correction_factor = no_of_transits_worst / no_of_transits_naive;

            double period = periods[tid] * SECONDS_PER_DAY;
            double T14Min = RsMin * pow((4 * period) / piGMMin / 1000000000000000, 1.0 / 3.0);

            double T14Max = (RsMax + R_jup*2) * pow((4 * period) / piGMMax / 1000000000000000, 1.0 / 3.0);
            double durationMin = T14Min / period;
            double durationMax = T14Max / period;
            if(durationMin > FRACTIONAL_TRANSIT_DURATION_MAX){
                durationMin = FRACTIONAL_TRANSIT_DURATION_MAX;
            }
            if(durationMax > FRACTIONAL_TRANSIT_DURATION_MAX){
                durationMax = FRACTIONAL_TRANSIT_DURATION_MAX;
            }
            int duration_min_in_samples = floor(durationMin * size);
            int duration_max_in_samples = ceil(durationMax * size * correction_factor);
            durationsMin[tid] = duration_min_in_samples;
            durationsMax[tid] = duration_max_in_samples;
        }
    }

    __global__ void patchData(float *in_patchedData,float *in_patchedDys,
    int *patchedDataSize,int *in_sortIndex,int *maxWidthInSamples,
    float *flux,float *dy,int *tSize,int *periodSize){
        int tid = blockIdx.x * blockDim.x + threadIdx.x; //patchedData index
        int y = blockIdx.y * blockDim.y + threadIdx.y; //period index

        float *patchedData = in_patchedData + y*(*patchedDataSize);
        float *patchedDys = in_patchedDys + y*(*patchedDataSize);
        int *sortIndex = in_sortIndex + y*(*tSize);

        if(tid < (*tSize)){
            patchedData[tid] = flux[sortIndex[tid]];
            patchedDys[tid] = dy[sortIndex[tid]];
        }
        else if(tid < (*tSize + *maxWidthInSamples)){
            patchedData[tid] = flux[sortIndex[tid - (*tSize)]];
            patchedDys[tid] = dy[sortIndex[tid - (*tSize)]];
        }
    }

    __global__ void calcInverseSquaredPatchedDy(float *out,
    float *patched_dys,int *patched_data_size){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(tid < *patched_data_size){
            out[tid + y*(*patched_data_size)] = 1 / (patched_dys[tid + y*(*patched_data_size)] * patched_dys[tid + y*(*patched_data_size)]);
        }
    }

    __global__ void calcEdgeEffectCorrections(float *out,float *patch_data,
    float* inverse_squared_patched_dys,int *patched_data_size,int* maxwidth_in_samples,int* period_size)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        float* patched_data = patch_data + tid*(*patched_data_size);
        float* inverse_squared_patched_dy = inverse_squared_patched_dys + tid*(*patched_data_size);

        double regular = 0;
        double patched = 0;
        if(tid < *period_size){
            for (int j = 0; j < (*patched_data_size - *maxwidth_in_samples); j++) {
                regular = regular + (1+(double)(patched_data[j])*(double)(patched_data[j])-2*(double)(patched_data[j])) * (double)(inverse_squared_patched_dy[j]);

            }
            for (int j = 0; j < (*patched_data_size); j++) {
                patched = patched + (1-patched_data[j]) *(1-patched_data[j]) * (double)(inverse_squared_patched_dy[j]);
            }
        out[tid] = patched - regular;
        }
    }

    __global__ void calcAllFullSum(float* fullsums,float *in_patched_data,
    float *in_inverse_squared_patched_dy,int *patched_data_size,int *in_duration,int *duration_size,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // tid:durations
        int y = blockIdx.y * blockDim.y + threadIdx.y;    // y:periods

        if(y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){// && tid < (*single_calc_periods_arr_gpu)){
            int y_input = (y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

            float *patched_data = in_patched_data + y_input*(*patched_data_size);
            float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y_input*(*patched_data_size);

            float fullsum = 0;
            for (int i = 0; i < *patched_data_size; i++) {
                fullsum = fullsum + ((1 - patched_data[i]) * (1 - patched_data[i])) * inverse_squared_patched_dy[i];
            }

            if(tid < (*duration_size)){
                int window = in_duration[tid];
                float window_sum = 0;
                for (int i = 0; i < window; i++) {
                    window_sum = window_sum + ((1 - patched_data[i]) * (1 - patched_data[i])) * inverse_squared_patched_dy[i];
                }
                fullsums[tid + y*(*duration_size)] = fullsum - window_sum;
            }
        }
    }
    
    __global__ void calcAllOutOfTransitResiduals_step1_2GPU(float *temp_ootr,
    float *in_patched_data, int *in_duration,int *duration_size,
    float *in_inverse_squared_patched_dy, int *patched_data_size,int *mean_x_size,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;  //z is period index

        if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
            int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

            float *patched_data = in_patched_data + z_input*(*patched_data_size);
            float *inverse_squared_patched = in_inverse_squared_patched_dy + z_input*(*patched_data_size);
            int window = in_duration[y];
            
            if(tid < *mean_x_size){
                if(tid < *patched_data_size - window){
                    int becomes_visible = tid;
                    int becomes_invisible = tid + window;
                    float add_visible_left = (1 - patched_data[becomes_visible]) * (1 - patched_data[becomes_visible]) * inverse_squared_patched[becomes_visible];
                    float remove_invisible_right = (1 - patched_data[becomes_invisible]) * (1 - patched_data[becomes_invisible]) * inverse_squared_patched[becomes_invisible];
                    float weight = add_visible_left - remove_invisible_right;
                    temp_ootr[tid + y*(*mean_x_size)+z*(*mean_x_size)*(*duration_size)] = weight;
                }
                else{
                    temp_ootr[tid + y*(*mean_x_size)+z*(*mean_x_size)*(*duration_size)] = 0;
                }
            }
        }
    }

    __global__ void calcAllOutOfTransitResiduals_step2_2GPU(float *in_ootr,
    int *duration_size,int *patched_data_size,int *in_mean_size,
    int *mean_x_size,float *in_fullsum,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;//i is point index
        int tid = blockIdx.y * blockDim.y + threadIdx.y;    //tid is duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      //z is period index

        int mean_size = in_mean_size[tid];
        if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
            float *fullsum = in_fullsum + z*(*duration_size);
            float *ootr = in_ootr + z*(*mean_x_size)*(*duration_size);
            float start = fullsum[tid];
            if(i < mean_size){
                ootr[i+tid*(*mean_x_size)] = start + ootr[i+tid*(*mean_x_size)];
            }
            else if (i<*mean_x_size){
                ootr[i+tid*(*mean_x_size)] = 0;
            }

        }
    }

    __device__ float calcAverageFromCumsum(float *inPatchedDataCumsum,
    int *duration,int *duration_size,int *patched_data_size,int *mean_x_size,
    int tid,int y,int z,int z_input){
        int window = duration[y];
        if(tid == 0){
            return 1 - inPatchedDataCumsum[z*(*patched_data_size) + window - 1] / window;
        }
        else{
            return 1 - (inPatchedDataCumsum[z*(*patched_data_size) + tid + window - 1] - inPatchedDataCumsum[z*(*patched_data_size) + tid - 1]) / window;
        }
    }

    __global__ void calcAllLowestResidualsGPU(float *out,
    float *depths, int *in_mean_size,
    int *mean_x_size,float *in_patched_datas,
    int *in_patched_datas_size,int *in_duration,int *in_duration_size,
    float *in_signal,int *in_max_signal_x_size,
    float *in_inverse_squared_patched_dys,
    float *in_overshoot, float *in_ootr,float *in_fullsum,
    float *in_summed_edge_effect_correction,int *in_datapoints,float *cumsumGPU,
    int *durationsMax,int *durationsMin, float *in_transit_depth_min,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
        int y = blockIdx.y * blockDim.y + threadIdx.y;      //y is the duration
        int z = blockIdx.z * blockDim.z + threadIdx.z;      //z is the period

        int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
        float transit_depth_min = *in_transit_depth_min;

        int mean_size = in_mean_size[y];
        int datapoints = *in_datapoints;
        if(z_input < (*period_size_gpu)){
            int durationMax = durationsMax[z_input];
            int durationMin = durationsMin[z_input];
            int duration = in_duration[y];

            if(duration >= durationMin && duration <= durationMax &&( tid %100 == 0) ){
            // if(duration >= durationMin && duration <= durationMax ){
                float calc_mean = calcAverageFromCumsum(cumsumGPU,in_duration,in_duration_size,in_patched_datas_size,mean_x_size,tid,y,z,z_input);
                float overshoot = in_overshoot[y];
                if(tid < *mean_x_size){
                    out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = datapoints;
                    depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = 0.0;
                }
                if(tid < mean_size && calc_mean > transit_depth_min){
                    float ootr = 0;
                    if(tid == 0){
                        ootr = in_fullsum[z*(*in_duration_size) + y];
                    }
                    else{
                        ootr = *(in_ootr+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size) + tid - 1);                    
                    }

                    float *patched_data = in_patched_datas + z_input*(*in_patched_datas_size);
                    // int signal_x_size = in_signal_x_size[y];
                    int signal_x_size = duration;
                    float *signal = in_signal+y*(*in_max_signal_x_size);
                    
                    float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + z_input*(*in_patched_datas_size);
                    float summed_edge_effect_correction = in_summed_edge_effect_correction[z_input];
                    int best_row = 0;
                    int best_depth = 0;
                    float SIGNAL_DEPTH = 0.5;

                    float *data = patched_data + tid;
                    float *dy = inverse_squared_patched_dy_arr + tid;
                    float target_depth = calc_mean * overshoot;
                    float reverse_scale = target_depth / SIGNAL_DEPTH;

                    float intransit_residual = 0;
                    float sigi = 0;
                    for (int i = 0; i < signal_x_size; i++) {
                        sigi = (1 - signal[i]) * reverse_scale;
                        intransit_residual = intransit_residual + ((data[i] - (1 - sigi)) * (data[i] - (1 - sigi))) * dy[i];
                    }

                    float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
                    out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = current_stat;
                    depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = target_depth;
                }
            }else{
                if(tid < *mean_x_size){
                    //0x7f800000 => infinity in float, according to IEEE-754
                    out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = 0x7f800000;
                    depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = 0.0;
                }
            }
        }
    }

}