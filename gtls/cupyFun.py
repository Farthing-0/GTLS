def getCuPyFun():
    cupyFun = (r'''
extern "C"{
    __global__ void foldFast(const float* time,const float* periods, float* phase,int* periodSize,int* timeSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if(tid < (*timeSize)){
            phase[tid + y*(*timeSize)] = time[tid] / periods[y] - (int)(time[tid] / periods[y]);
        }
    }

    __global__ void patchData(float *in_patchedData,float* in_patchedDys,
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

    __global__ void calcEdgeEffectCorrections(float *out,float *patched_dys,
    float* inverse_squared_patched_dys,int *patched_data_size,int* maxwidth_in_samples,int* period_size)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        float* patched_dy = patched_dys + tid*(*patched_data_size);
        float* inverse_squared_patched_dy = inverse_squared_patched_dys + tid*(*patched_data_size);

        float regular = 0;
        float patched = 0;
        if(tid < *period_size){
            for (int j = 0; j < (*patched_data_size - *maxwidth_in_samples); j++) {
                regular = regular + ((1 - patched_dy[j]) * (1 - patched_dy[j])) * (inverse_squared_patched_dy[tid]);
            }
            for (int j = 0; j < (*patched_data_size); j++) {
                patched = patched + ((1 - patched_dy[j]) * (1 - patched_dy[j])) * (inverse_squared_patched_dy[tid]);
            }
        out[tid] = patched - regular;
        }
    }

    __device__ float calcFullSum(int tid,float *in_patched_data,
    float *in_inverse_squared_patched_dy,int *patched_data_size,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int tid_input = (tid + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
        float *patched_data = in_patched_data + tid_input*(*patched_data_size);
        float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + tid_input*(*patched_data_size);
        float fullsum = 0;
        for (int i = 0; i < *patched_data_size; i++) {
            fullsum = fullsum + ((1 - patched_data[i]) * (1 - patched_data[i])) * inverse_squared_patched_dy[i];
        }
        return fullsum;
    }

    __global__ void calcAllFullSum(float* fullsums,float *in_patched_data,
    float *in_inverse_squared_patched_dy,int *patched_data_size,int *in_duration,int *duration_size,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // tid:durations
        int y = blockIdx.y * blockDim.y + threadIdx.y;    // y:periods

        if(y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){// && tid < (*single_calc_periods_arr_gpu)){
            int y_input = (y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

            // fullsum = numpy.sum(((1 - patched_data) ** 2) * inverse_squared_patched_dy)

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
            int window = in_duration[y];
            float *inverse_squared_patched = in_inverse_squared_patched_dy + z_input*(*patched_data_size);
            
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
    float *in_patched_data,
    float *in_inverse_squared_patched_dy,int *in_duration,
    int *duration_size,int *patched_data_size,int *in_mean_size,
    int *mean_x_size,float *in_fullsum,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;//i is point index
        int tid = blockIdx.y * blockDim.y + threadIdx.y;    //tid is duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      //z is period index

        int mean_size = in_mean_size[tid];

        if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
            if(tid < *duration_size &&  i < mean_size){
                int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
                int window = in_duration[tid];
                float *patched_data = in_patched_data + z_input*(*patched_data_size);
                float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + z_input*(*patched_data_size);
                float *fullsum = in_fullsum + z*(*duration_size);
                float *ootr = in_ootr + z*(*mean_x_size)*(*duration_size);
                float start = fullsum[tid];
                ootr[i+tid*(*mean_x_size)] = start + ootr[i+tid*(*mean_x_size)];
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
    float *in_signal,int *in_signal_x_size,
    int *in_max_signal_x_size,float *in_inverse_squared_patched_dys,
    float *in_overshoot, float *in_ootr,float *in_summed_edge_effect_correction,int *in_datapoints,float *cumsumGPU,
    int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
        int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is the duration
        int z = blockIdx.z * blockDim.z + threadIdx.z;  //z is the period

        int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
        float transit_depth_min = 0.00001;

        int mean_size = in_mean_size[y];
        int datapoints = *in_datapoints;
        if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
            float calc_mean = calcAverageFromCumsum(cumsumGPU,in_duration,in_duration_size,in_patched_datas_size,mean_x_size,tid,y,z,z_input);
            float transit_depth_min = 0.00001;
            float overshoot = in_overshoot[y];
            if(tid < *mean_x_size){
                out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = datapoints;
                depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = 0.0;
                }
            if(tid < mean_size && calc_mean > transit_depth_min){
                int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
                
                float ootr = 0;
                if(tid == 0){
                    ootr = *(in_ootr+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size) + tid);
                }
                else{
                    ootr = *(in_ootr+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size) + tid - 1);                    
                }

                float *patched_data = in_patched_datas + z_input*(*in_patched_datas_size);
                int duration = in_duration[y];
                int signal_x_size = in_signal_x_size[y];
                float *signal = in_signal+y*(*in_max_signal_x_size);
                
                float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + z_input*(*in_patched_datas_size);
                float summed_edge_effect_correction = in_summed_edge_effect_correction[z_input];
                int best_row = 0;
                int best_depth = 0;
                int T0_fit_margin = 100;
                float SIGNAL_DEPTH = 0.5;

                float *data = patched_data + tid;
                float *dy = inverse_squared_patched_dy_arr + tid;
                int xth_point = 0;
                xth_point = duration / T0_fit_margin;
                if(xth_point < 1){
                    xth_point = 1;
                }

                float target_depth = calc_mean * overshoot;
                float scale = SIGNAL_DEPTH / target_depth;
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
        }
    }
}
    ''')
    return cupyFun