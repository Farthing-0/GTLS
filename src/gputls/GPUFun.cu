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
    int *patchedDataSize,int *in_sortIndex,int *maxDuration,
    float *flux,float *dy,int *tSize){
        int tid = blockIdx.x * blockDim.x + threadIdx.x; //patchedData index
        int y = blockIdx.y * blockDim.y + threadIdx.y; //period index

        float *patchedData = in_patchedData + y*(*patchedDataSize);
        float *patchedDys = in_patchedDys + y*(*patchedDataSize);
        int *sortIndex = in_sortIndex + y*(*tSize);

        if(tid < (*tSize)){
            patchedData[tid] = flux[sortIndex[tid]];
            patchedDys[tid] = dy[sortIndex[tid]];
        }
        else if(tid < (*tSize + *maxDuration)){
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
    float* inverse_squared_patched_dys,int *patched_data_size,int* maxDuration,int* period_size)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        float* patched_data = patch_data + tid*(*patched_data_size);
        float* inverse_squared_patched_dy = inverse_squared_patched_dys + tid*(*patched_data_size);

        double regular = 0;
        double patched = 0;
        if(tid < *period_size){
            for (int j = 0; j < (*patched_data_size - *maxDuration); j++) {
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
    int *period_size_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // tid:durations
        int y = blockIdx.y * blockDim.y + threadIdx.y;    // y:periods

        // if(y < (*period_size_gpu)){// && tid < (*single_calc_periods_arr_gpu)){
        //     int y_input = (y );

            // float *patched_data = in_patched_data + y_input*(*patched_data_size);
            // float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y_input*(*patched_data_size);
        float *patched_data = in_patched_data + y*(*patched_data_size);
        float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y*(*patched_data_size);

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
        // }
    }
    
    __global__ void calcAllOutOfTransitResiduals_step1_2GPU(float *temp_ootr,
    float *in_patched_data, int *in_duration,int *duration_size,
    float *in_inverse_squared_patched_dy, int *patched_data_size,int *resultArrayXAxisSize)//,
    // int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;  //z is period index

        // if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
        //     int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

        float *patched_data = in_patched_data + z*(*patched_data_size);
        float *inverse_squared_patched = in_inverse_squared_patched_dy + z*(*patched_data_size);
        int window = in_duration[y];
        
        if(tid < *resultArrayXAxisSize){
            if(tid < *patched_data_size - window){
                int becomes_visible = tid;
                int becomes_invisible = tid + window;
                float add_visible_left = (1 - patched_data[becomes_visible]) * (1 - patched_data[becomes_visible]) * inverse_squared_patched[becomes_visible];
                float remove_invisible_right = (1 - patched_data[becomes_invisible]) * (1 - patched_data[becomes_invisible]) * inverse_squared_patched[becomes_invisible];
                float weight = add_visible_left - remove_invisible_right;
                temp_ootr[tid + y*(*resultArrayXAxisSize)+z*(*resultArrayXAxisSize)*(*duration_size)] = weight;
            }
            else{
                temp_ootr[tid + y*(*resultArrayXAxisSize)+z*(*resultArrayXAxisSize)*(*duration_size)] = 0;
            }
        }
        // }
    }

    __global__ void calcAllOutOfTransitResiduals_step2_2GPU(float *in_ootr,
    int *duration_size,int *patched_data_size,int *in_duration,
    int *resultArrayXAxisSize,float *in_fullsum)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;//i is point index
        int tid = blockIdx.y * blockDim.y + threadIdx.y;    //tid is duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      //z is period index

        // int mean_size = in_mean_size[tid];
        int duration = in_duration[tid];

        float *fullsum = in_fullsum + z*(*duration_size);
        float *ootr = in_ootr + z*(*resultArrayXAxisSize)*(*duration_size);
        float start = fullsum[tid];
        if(i < (*patched_data_size) - duration + 1){
            ootr[i+tid*(*resultArrayXAxisSize)] = start + ootr[i+tid*(*resultArrayXAxisSize)];
        }
        else if (i<*resultArrayXAxisSize){
            ootr[i+tid*(*resultArrayXAxisSize)] = 0;
        }
    }

    __device__ float calcAverageFromCumsum(float *inPatchedDataCumsum,
    int duration, int *patched_data_size, int tid, int z){
        if(tid == 0){
            return 1 - inPatchedDataCumsum[z*(*patched_data_size) + duration - 1] / duration;
        }
        else{
            return 1 - (inPatchedDataCumsum[z*(*patched_data_size) + tid + duration - 1] - inPatchedDataCumsum[z*(*patched_data_size) + tid - 1]) / duration;
        }
    }

    // __global__ void calcAllLowestResidualsAtomGPU(int signal_x_size,float* signal,float* data,float* dy,float reverse_scale,float* intransit_residuals){
    //     // for (int i = 0; i < signal_x_size; i++) {
    //     //     sigi = (1 - signal[i]) * reverse_scale;
    //     //     intransit_residual = intransit_residual + ((data[i] - (1 - sigi)) * (data[i] - (1 - sigi))) * dy[i];
    //     // }
    //     int tid = blockIdx.x * blockDim.x + threadIdx.x; //tid is point index, tid < signal_x_size

    //     if(tid < signal_x_size){
    //         // float sigi = (1 - signal[tid]) * reverse_scale;
    //         // intransit_residuals[tid] = ((data[tid] - (1 - sigi)) * (data[tid] - (1 - sigi))) * dy[tid];
    //         intransit_residuals[tid] = 0;
    //     }
    // }

    // __global__ void calcAllLowestResidualsGPU(
    // float *out,//float *depths,
    // int *in_mean_size,int *resultArrayXAxisSize,
    // float *in_patched_datas,
    // int *in_patched_datas_size,int *in_duration,int *in_duration_size,
    // float *in_signal,float *in_signal_grazing,float *in_signal_box,
    // int *in_max_signal_x_size,
    // float *in_inverse_squared_patched_dys,
    // float *in_overshoot, float *in_ootr,float *in_fullsum,
    // float *in_summed_edge_effect_correction,int *in_datapoints,float *cumsumGPU,
    // int *durationsMax,int *durationsMin, float *in_transit_depth_min,
    // int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu
    // )
    // {
    //     int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
    //     int y = blockIdx.y * blockDim.y + threadIdx.y;      //y is the duration
    //     int z = blockIdx.z * blockDim.z + threadIdx.z;      //z is the period

    //     int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
    //     float transit_depth_min = *in_transit_depth_min;

    //     int mean_size = in_mean_size[y];
    //     int datapoints = *in_datapoints;
    //     if(z_input < (*period_size_gpu)){
    //         int durationMax = durationsMax[z_input];
    //         int durationMin = durationsMin[z_input];
    //         int duration = in_duration[y];

    //         // if(duration >= durationMin && duration <= durationMax &&( tid %100 == 0) ){
    //         if(duration >= durationMin && duration <= durationMax ){
    //             float calc_mean = calcAverageFromCumsum(cumsumGPU,duration,in_patched_datas_size,tid,z);
    //             float overshoot = in_overshoot[y];
    //             if(tid < *resultArrayXAxisSize){
    //                 out[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = datapoints;
    //                 //depths[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 0.0;
    //                 // outType[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 3;
    //             }
    //             if(tid < mean_size && calc_mean > transit_depth_min){
    //                 float ootr = 0;
    //                 if(tid == 0){
    //                     ootr = in_fullsum[z*(*in_duration_size) + y];
    //                 }
    //                 else{
    //                     ootr = *(in_ootr+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size) + tid - 1);                    
    //                 }

    //                 float *data = in_patched_datas + z_input*(*in_patched_datas_size) + tid;
    //                 // int signal_x_size = in_signal_x_size[y];
    //                 int signal_x_size = duration;
    //                 float *signal = in_signal+y*(*in_max_signal_x_size);
    //                 float *signal_grazing = in_signal_grazing+y*(*in_max_signal_x_size);
    //                 float *signal_box = in_signal_box+y*(*in_max_signal_x_size);
                    
    //                 float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + z_input*(*in_patched_datas_size);
    //                 float summed_edge_effect_correction = in_summed_edge_effect_correction[z_input];
    //                 float SIGNAL_DEPTH = 0.5;

    //                 float *dy = inverse_squared_patched_dy_arr + tid;
    //                 float target_depth = calc_mean * overshoot;
    //                 float reverse_scale = target_depth / SIGNAL_DEPTH;

    //                 float sigi = 0;
    //                 // float intransit_residual = 0;
    //                 // float intransit_residuals[(signal_x_size)];
    //                 // float* intransit_residuals = (float*)cudaMalloc(signal_x_size * sizeof(float));
    //                 float* intransit_residuals ;
    //                 cudaMalloc((void**)&intransit_residuals,1 * sizeof(float));
    //                 // int signal_x_size,float* signal,float* data,float* dy,float reverse_scale,float* intransit_residuals){
    //                 // calcAllLowestResidualsAtomGPU<<<1,signal_x_size>>>(signal_x_size,signal,data,dy,reverse_scale,intransit_residuals);
    //                 // calcAllLowestResidualsAtomGPU<<<1,1>>>(signal_x_size,signal,data,dy,reverse_scale,intransit_residuals);
    //                 // cudaDeviceSynchronize();
    //                 float intransit_residual = 0;
    //                 // for(int i = 0; i < signal_x_size; i++){
    //                 //     intransit_residual = intransit_residual + intransit_residuals[i];
    //                 // }
    //                 // free(intransit_residuals);
    //                 // for (int i = 0; i < signal_x_size; i++) {
    //                 //     sigi = (1 - signal[i]) * reverse_scale;
    //                 //     intransit_residual = intransit_residual + ((data[i] - (1 - sigi)) * (data[i] - (1 - sigi))) * dy[i];
    //                 // }

    //                 float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
    //                 // float current_stat_grazing = intransit_residual_grazing + ootr - summed_edge_effect_correction;
    //                 // float current_stat_box = intransit_residual_box + ootr - summed_edge_effect_correction;
    //                 cudaFree(intransit_residuals);
    //                 out[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 0;
    //                 // out[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = current_stat_box;
    //                 // out[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = min(current_stat, min(current_stat_grazing, current_stat_box));
    //                 //depths[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = target_depth;
    //             }
    //         }else{
    //             if(tid < *resultArrayXAxisSize){
    //                 //0x7f800000 => infinity in float, according to IEEE-754
    //                 out[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 0x7f800000;
    //                 //depths[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 0.0;
    //                 // outType[tid+y*(*resultArrayXAxisSize) + z*(*resultArrayXAxisSize)*(*in_duration_size)] = 3;
    //             }
    //         }
    //     }
    // }

    __global__ void calcAllLowestResidualsGPU(
    float *out,//float *depths,
    int *resultArrayXAxisSize,
    float *in_patched_datas,
    int *in_patched_datas_size,int *in_duration,int *in_duration_size,
    float *in_signal,int *in_signal_size,//float *in_signal_grazing,float *in_signal_box,
    int *in_max_signal_x_size,
    float *in_inverse_squared_patched_dys,
    float *in_overshoot, float *in_ootr,float *in_fullsum,
    float *in_summed_edge_effect_correction,int *in_datapoints,float *cumsumGPU,
    int *durationsMax,int *durationsMin, float *in_transit_depth_min,
    int iter_flag_gpu
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
        int y = blockIdx.y * blockDim.y + threadIdx.y;      //y is the period

        float transit_depth_min = *in_transit_depth_min;

        int datapoints = *in_datapoints;
        int durationMax = durationsMax[y];
        int durationMin = durationsMin[y];

        for (int durationIndex = 0; durationIndex < *in_duration_size; durationIndex++){

            int duration = in_duration[durationIndex];
            int mean_size = *in_patched_datas_size - duration + 1;

            if(duration >= durationMin && duration <= durationMax ){
                float calc_mean = calcAverageFromCumsum(cumsumGPU,duration,in_patched_datas_size,tid,y);
                float overshoot = in_overshoot[durationIndex];
                if(tid < *resultArrayXAxisSize){
                    out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = datapoints;
                }
                if(tid < mean_size && calc_mean > transit_depth_min){
                    float ootr = 0;
                    if(tid == 0){
                        ootr = in_fullsum[y*(*in_duration_size) + durationIndex];
                    }
                    else{
                        ootr = *(in_ootr+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size) + tid - 1);                    
                    }

                    float *data = in_patched_datas + y*(*in_patched_datas_size) + tid;
                    float *signal = in_signal+durationIndex*(*in_max_signal_x_size);
                    // float signal = 0.1;

                    float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + y*(*in_patched_datas_size);
                    float summed_edge_effect_correction = in_summed_edge_effect_correction[y];

                    float *dy = inverse_squared_patched_dy_arr + tid;
                    float reverse_scale = calc_mean * overshoot * 2;  // "*2" means SIGNAL_DEPTH is 0.5,as "/SIGANL_DEPTH"

                    float sigi = 0;
                    float intransit_residual = 0;
                    float loss = 0;
                    for (int i = 0; i < duration; i++) {
                        sigi = (signal[i]) * reverse_scale;
                        // sigi = (signal) * reverse_scale;
                        loss = (data[i] - (1 - sigi));
                        intransit_residual = intransit_residual + loss * loss * dy[i];
                        // intransit_residual = intransit_residual + loss * loss;
                    }
                    float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
                    out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = current_stat;
                }
            }
            else{
                if(tid < *resultArrayXAxisSize){
                    //0x7f800000 => infinity in float, according to IEEE-754
                    out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = 0x7f800000;
                }
            }
        }
    }

    __global__ void calcAllLowestResidualsCompatibleGPU(
    float *out,//float *depths,
    int *in_mean_size,int *resultArrayXAxisSize,
    float *in_patched_datas,
    int *in_patched_datas_size,int *in_duration,int *in_duration_size,
    float *in_signal,//float *in_signal_grazing,float *in_signal_box,
    int *in_max_signal_x_size,
    float *in_inverse_squared_patched_dys,
    float *in_overshoot, float *in_ootr,float *in_fullsum,
    float *in_summed_edge_effect_correction,int *in_datapoints,float *cumsumGPU,
    int *durationsMax,int *durationsMin, float *in_transit_depth_min,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
        int y = blockIdx.y * blockDim.y + threadIdx.y;      //y is the period

        int y_input = (y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
        float transit_depth_min = *in_transit_depth_min;

        int datapoints = *in_datapoints;
        if(y_input < (*period_size_gpu)){
            int durationMax = durationsMax[y_input];
            int durationMin = durationsMin[y_input];

            for (int durationIndex = 0; durationIndex < *in_duration_size; durationIndex++){
                int mean_size = in_mean_size[durationIndex];
                int duration = in_duration[durationIndex];

                int skipGap = int(duration/100);
                if(duration >= durationMin && duration <= durationMax && tid %skipGap == 0){
                // if(duration >= durationMin && duration <= durationMax ){

                    float calc_mean = calcAverageFromCumsum(cumsumGPU,duration,in_patched_datas_size,tid,y);
                    float overshoot = in_overshoot[durationIndex];
                    if(tid < *resultArrayXAxisSize){
                        out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = datapoints;
                    }
                    if(tid < mean_size && calc_mean > transit_depth_min){
                        float ootr = 0;
                        if(tid == 0){
                            ootr = in_fullsum[y*(*in_duration_size) + durationIndex];
                        }
                        else{
                            ootr = *(in_ootr+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size) + tid - 1);                    
                        }

                        float *data = in_patched_datas + y_input*(*in_patched_datas_size) + tid;
                        float *signal = in_signal+durationIndex*(*in_max_signal_x_size);
                        // float *signal = in_signal;

                        float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + y_input*(*in_patched_datas_size);
                        float summed_edge_effect_correction = in_summed_edge_effect_correction[y_input];
                        // float SIGNAL_DEPTH = 0.5;

                        float *dy = inverse_squared_patched_dy_arr + tid;
                        // float target_depth = calc_mean * overshoot;
                        float reverse_scale = calc_mean * overshoot * 2;  // "*2" means SIGNAL_DEPTH is 0.5,as "/SIGANL_DEPTH"

                        float sigi = 0;
                        float intransit_residual = 0;
                        float loss = 0;
                        for (int i = 0; i < duration; i++) {
                            // sigi = (1 - signal[i]) * reverse_scale;
                            sigi = (signal[i]) * reverse_scale;
                            // sigi = 1;
                            loss = (data[i] - (1 - sigi));
                            intransit_residual = intransit_residual + loss * loss * dy[i];
                        }

                        float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
                        out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = current_stat;
                    }
                }else{
                    if(tid < *resultArrayXAxisSize){
                        //0x7f800000 => infinity in float, according to IEEE-754
                        out[tid+durationIndex*(*resultArrayXAxisSize) + y*(*resultArrayXAxisSize)*(*in_duration_size)] = 0x7f800000;
                    }
                }
            }
        }
    }

    // This function is used after the best period and duration are found, to calculate the SNR and some other metrics.
    __global__ void trapezoidFit(float *results,
    float *inData, float *inInverseSquaredDys,
    int duration, int inT0Index,
    float *transitMean, int tidMax ){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < tidMax) {
            float *result = results + tid*(duration);
            //TrapezoidDepth can not change since I use other fixed values in the kernel
            float TrapezoidDepth = ((float)tidMax * (*transitMean) - 0.5*(float)tid)/((float)tidMax - 0.5*(float)tid);
            float meanSignal = (tid*0.75 + (tidMax - tid)*TrapezoidDepth)/tidMax;
            
            float trapezoidQ = ((float)tid/(float)tidMax)*(float(duration)/2);
            float signal;
            float sigi = 0;
            float intransitResidual = 0;

            float *data = inData + inT0Index;
            float *dy = inInverseSquaredDys + inT0Index;

            //Warining: Might be a bug especially when duration is small
            for (int i = 0; i < duration; i++) {
                if (i == 0 && tid != 0) {
                    signal = 1 - ((1-TrapezoidDepth)/(trapezoidQ));
                }
                else if(i < trapezoidQ){
                    signal = (1 - ((1-TrapezoidDepth)/(trapezoidQ))*i);
                }
                else if(i >= trapezoidQ && i < duration - trapezoidQ){
                    signal = TrapezoidDepth;
                }
                else{
                    signal = (1-((1-TrapezoidDepth)/(trapezoidQ))*(duration-i));
                }
                result[i] = (data[i] - signal) * (data[i] - signal) * dy[i];
            }
        }
    }

    // This function is used after the best period and duration are found, to calculate the SNR and some other metrics.
    // "Atom" means that it is a single thread, and it is used to calculate the a single point loss of trapezoid fit.
    __global__ void trapezoidFitAtom(float *results,
    float *inData, float *inInverseSquaredDys,
    int duration, int inT0Index,float *transitMean, int trapezoidFitSize){
        int tid = blockIdx.x * blockDim.x + threadIdx.x; //tid is the index of the point in the trapezoid fit, tid < duration
        int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is the index of different types of trapezoid fit, y < trapezoidFitSize

        // For now, z is not used, but it is reserved for future use. 
        // int z = blockIdx.z * blockDim.z + threadIdx.z;  //z is the index of different periods and durations (period and duration are one to one mapping)
        printf("data[inT0Index]:%f\\n",inData[inT0Index]);
        if (y < trapezoidFitSize && tid < duration) {
            printf("tid:%d,duration:%d\\n",tid,duration);
            float *result = results + y*(duration) +tid;

            //TrapezoidDepth can not change since I use other fixed values in the kernel
            float TrapezoidDepth = ((float)trapezoidFitSize * (*transitMean) - 0.5*(float)y)/((float)trapezoidFitSize - 0.5*(float)y);
            float meanSignal = (y*0.75 + (trapezoidFitSize - y)*TrapezoidDepth)/trapezoidFitSize;
            
            float trapezoidQ = ((float)y/(float)trapezoidFitSize)*(float(duration)/2);
            float signal;
            float sigi = 0;
            float intransitResidual = 0;

            float *data = inData + inT0Index + tid;
            float *dy = inInverseSquaredDys + inT0Index + tid;
            // float *data = inData;
            // float *dy = inInverseSquaredDys;

            //Warining: Might be a bug especially when duration is small
            // for (int i = 0; i < duration; i++) {
            if (tid == 0 && y != 0) {
                signal = 1 - ((1-TrapezoidDepth)/(trapezoidQ));
            }
            else if(tid < trapezoidQ){
                signal = (1 - ((1-TrapezoidDepth)/(trapezoidQ))*tid);
            }
            else if(tid >= trapezoidQ && tid < duration - trapezoidQ){
                signal = TrapezoidDepth;
            }
            else{
                signal = (1-((1-TrapezoidDepth)/(trapezoidQ))*(duration-tid));
            }
            // result[tid] = (data[tid] - signal) * (data[tid] - signal) * dy[tid];
            *result = (*data - signal) * (*data - signal);
            // result[tid] = 1;
        }
    }

    __global__ void generateTrapezoidFit(float *results,
    int bestFitTid, int duration, int trapezoidFitSize, float TrapezoidDepth){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < duration){
            float trapezoidQ = ((float)bestFitTid/(float)trapezoidFitSize)*(float(duration)/2);
            // printf("trapezoidQ: %d", bestFitTid);
            float signal;
            // float TrapezoidDepth = *TrapezoidDepthArray;

            if (tid == 0 && bestFitTid != 0) {
                signal = 1 - ((1-TrapezoidDepth)/(trapezoidQ));
            }
            else if(tid < trapezoidQ){
                signal = (1 - ((1-TrapezoidDepth)/(trapezoidQ))*tid);
            }
            else if(tid >= trapezoidQ && tid < duration - trapezoidQ){
                signal = TrapezoidDepth;
            }
            else{
                signal = (1-((1-TrapezoidDepth)/(trapezoidQ))*(duration-tid));
            }
            results[tid] = signal;
        }
    }

    // This function is used after the best period and duration are found, to calculate the "new" SNR, which will delete trapzoid fit form the light curve, 
    // The difference is every single point will subtract the trapezoid fit to calculate the standard deviation of the residual.
    __global__ void trapezoidSNRloss(float *results,int resultSize,float *inData, float *inInverseSquaredDys,
    int duration, float* trapezoidFit){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < resultSize){
            float *data = inData + tid;
            float *dy = inInverseSquaredDys + tid;
            // float *result = results + tid; 

            float tempResidual = 0;
            for (int i = 0; i < duration; i++) {
                tempResidual = tempResidual + (data[i] - trapezoidFit[i]) * (data[i] - trapezoidFit[i]) * dy[i];
            }
            results[tid] = tempResidual;
        }
    }

    // For debug
    __global__ void trapezoidSNRlossAtom(float *results,int resultSize,float *inData, float *inInverseSquaredDys,
    int duration, float* trapezoidFit){
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // every point's lost in a duration, tid < duration
        int y = blockIdx.y * blockDim.y + threadIdx.y;  // every first point, y < len(t), which is resultSize

        if(y < resultSize){
            float *data = inData + y + tid;
            float *dy = inInverseSquaredDys + y + tid;
            float *result = results + y*duration + tid; 

            *result  = (*data - trapezoidFit[tid]) * (*data - trapezoidFit[tid]) * (*dy);
        }
    }

    __global__ void postTransitFitAtom(float *results,
    float *inData, float *idealTransit, float *inInverseSquaredDys,
    int duration, int inDataSize, int idealTransitSize,float *pointResult){
        int x = blockIdx.x * blockDim.x + threadIdx.x; //x is the index of the point in the folded curve, x < len(t)
        int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is the index of different types of idealTransit fit, y < idealTransitSize

        float *result = results + y*inDataSize + x;
        float tempResult = 0;
        float tempTotalResult = 0;
        // float pointResult[duration];


        float data;
        float dy;
        float idealTransitValue;
        if (x < inDataSize){
            for(int i = 0; i < duration; i++){
                data = inData[x + i];
                dy = inInverseSquaredDys[x + i];
                idealTransitValue = idealTransit[y*duration + i];

                tempResult = (data - idealTransitValue) * (data - idealTransitValue) * dy;
                pointResult[i] = tempResult;
                tempTotalResult = tempTotalResult + tempResult;
            }
        }
        // *result = tempTotalResult;
        // result = std(pointResult)
        float mean = tempTotalResult/duration;
        float std = 0;
        for(int i = 0; i < duration; i++){
            std = std + (pointResult[i] - mean) * (pointResult[i] - mean);
        }
        std = sqrt(std/duration);
        *result = std;
    }

}