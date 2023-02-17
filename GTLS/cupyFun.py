import cupy as cp

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

        __global__ void test_multiply(const float* x1, const float* x2, float* y, unsigned int N){
            unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < N)
            {
                y[tid] = x1[tid] * x2[tid];
            }
        }

        __global__ void calcInverseSquaredPatchedDy(float *out,
        float *patched_dys,int *patched_data_size){
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if(tid < *patched_data_size){
                out[tid + y*(*patched_data_size)] = 1 / (patched_dys[tid + y*(*patched_data_size)] * patched_dys[tid + y*(*patched_data_size)]);
                //printf("out:%f\\n",1 / (patched_dys[tid + y*(*patched_data_size)] * patched_dys[tid + y*(*patched_data_size)]));
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

        __global__ void calcAllAverage(float *out,float *in_patched_data,
        int *duration,int *duration_size,int *patched_data_size,int *mean_x_size,
        int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu
        )
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
                
                int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
                int tid_start = tid + 1;
                int tid_end = tid_start + duration[y];
                int raw_array_size = *patched_data_size + 1;
                float *raw_array = in_patched_data + z_input*(*patched_data_size);
                

                //printf("y:%d,duration[y]:%d\\n",y,duration[y]);
                float mean = 0;
                for (int i = tid_start; i < tid_end; i++) {
                    //i == 0:raw_array[0] == 0,but this raw_array is faked by patched_data, so skip it.
                    if(i > 0){
                        mean += raw_array[i - 1];
                    }
                    ;
                }

                if(tid < mean_x_size[0]){
                    out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*duration_size)] =  mean / duration[y];
                    //out[tid+y*(*mean_x_size)] =  mean / duration[y];
                    if(tid_end > raw_array_size){
                        out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*duration_size)] = 0;
                    }
                }
            }
        }

        __global__ void calcAllFullSum(float* fullsums,float *in_patched_data,
        float *in_inverse_squared_patched_dy,int *patched_data_size,
        int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if(tid + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
                int tid_input = (tid + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

                // fullsum = numpy.sum(((1 - patched_data) ** 2) * inverse_squared_patched_dy)

                float *patched_data = in_patched_data + tid_input*(*patched_data_size);
                float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + tid_input*(*patched_data_size);

                fullsums[tid] = 0;
                for (int i = 0; i < *patched_data_size; i++) {
                    fullsums[tid] = fullsums[tid] + ((1 - patched_data[i]) * (1 - patched_data[i])) * inverse_squared_patched_dy[i];
                }
            }
        }

        __global__ void calcAllOutOfTransitResiduals_step1_2GPU(float *temp_ootr,
        float *in_patched_data, int *in_duration,int *duration_size,
        float *in_inverse_squared_patched_dy, int *patched_data_size,int *mean_x_size,
        int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int z = blockIdx.z * blockDim.z + threadIdx.z;

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
                /*else{
                        temp_ootr[tid + y*(*mean_x_size)+z*(*mean_x_size)*(*duration_size)] = 0;
                    }*/
            }
        }

        __global__ void calcAllOutOfTransitResiduals_step2_2GPU(float *in_ootr,
        float* in_fullsum,float *in_patched_data,
        float *in_inverse_squared_patched_dy,
        int *in_duration,int *duration_size,int *patched_data_size,int *mean_x_size,
        int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if(y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
                if(tid < *duration_size){
                    int y_input = (y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

                    int window = in_duration[tid];
                    float *patched_data = in_patched_data + y_input*(*patched_data_size);
                    float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y_input*(*patched_data_size);

                    float fullsum = in_fullsum[y];            
                    //float *ootr =out_ootr + y*(*mean_x_size)*(*duration_size);
                    float *ootr = in_ootr + y*(*mean_x_size)*(*duration_size);

                    //float window_sum = numpy.sum(((1 - data[:width_signal]) ** 2) * dy[:width_signal])
                    float window_sum = 0;
                    /*for (int i = 0; i < window; i++) {
                        window_sum = window_sum + ((1 - patched_data[i]) * (1 - patched_data[i])) * inverse_squared_patched_dy[i];
                    }*/
                    //printf("tid: %d, y: %d, window:%d,fullsum: %f, window_sum:,mean_x_size: %d, duration_size: %d\n",tid,y,window,fullsum,*mean_x_size,*duration_size);
                    //ootr[tid*(*mean_x_size)] = fullsum - window_sum;
                    float start = fullsum - window_sum;
                    
                    for (int i = *patched_data_size - window; i > 0 ; i--) {
                        ootr[i+tid*(*mean_x_size)] = ootr[i - 1 + tid*(*mean_x_size)] + start;
                        //ootr[i+tid*(*mean_x_size)] = start;
                        /*if(y == 0 && tid == 0 && i<10){
                        printf("tid:%d,i:%d,ootr: %f,start: %f\n",tid,i,ootr[i + tid*(*mean_x_size)],start);               
                        }*/
                    }
                    ootr[tid*(*mean_x_size)] = start;
                }
            }
        }

        __device__ float calcAverage(float *in_patched_data,
        int *duration,int *duration_size,int *patched_data_size,int *mean_x_size,
        int tid,int y,int z,int z_input)
        {
                int tid_start = tid + 1;
                int tid_end = tid_start + duration[y];
                int raw_array_size = *patched_data_size + 1;
                float *raw_array = in_patched_data + z_input*(*patched_data_size);

                float mean = 0;
                for (int i = tid_start; i < tid_end; i++) {
                    //i == 0:raw_array[0] == 0,but this raw_array is faked by patched_data, so skip it.
                    if(i > 0){
                        mean += raw_array[i - 1];
                    }
                    ;
                }
                float average = 0;
                if(tid < mean_x_size[0]){
                    average =  mean / duration[y];
                    /*if(tid_end > raw_array_size){
                        average = 0;
                    }*/
                }
                return 1 - average;
        }

        __global__ void calcAllLowestResiduals(float *out,
        float *depths, int *in_mean_size,
        int *mean_x_size,float *in_patched_datas,
        int *in_patched_datas_size,int *in_duration,int *in_duration_size,
        float *in_signal,int *in_signal_x_size,
        int *in_max_signal_x_size,float *in_inverse_squared_patched_dys,
        float *in_overshoot, float *in_ootr,float *in_summed_edge_effect_correction,int *in_datapoints,
        int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu

        //Constants
        // float *in_transit_depth_min,
        //    float *in_T0_fit_margin
        )
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;    //tid is each point
            int y = blockIdx.y * blockDim.y + threadIdx.y;  //y is the duration
            int z = blockIdx.z * blockDim.z + threadIdx.z;  //z is the period


            if(z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
                if(tid < mean_size){
                    int z_input = (z + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));
                    float calc_mean = calcAverage(in_patched_datas,in_duration,in_duration_size,in_patched_datas_size,mean_x_size,tid,y,z,z_input);

                    float transit_depth_min = 0.00001;
                    int mean_size = in_mean_size[y];
                    //int ootr_size = mean_size;

                    //float* mean = in_mean+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size);
                    float* ootr = in_ootr+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size);

                    float *patched_data = in_patched_datas + z_input*(*in_patched_datas_size);
                    int duration = in_duration[y];
                    int signal_x_size = in_signal_x_size[y];
                    float *signal = in_signal+y*(*in_max_signal_x_size);
                    
                    float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + z_input*(*in_patched_datas_size);
                    float overshoot = in_overshoot[y];
                    //TODO:To be check.!!!
                    float summed_edge_effect_correction = in_summed_edge_effect_correction[z_input];
                    int datapoints = *in_datapoints;

                    //int summed_residual_in_rows = datapoints[y];
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
                    //float calc_mean = 1 - mean[tid];

                    if(calc_mean > transit_depth_min){
                        /*if(y == 13){
                        printf("y:%d, tid:%d, calc_mean:%f, overshoot:%f\n",y,tid,calc_mean,overshoot);
                        }*/
                        float target_depth = calc_mean * overshoot;
                        float scale = SIGNAL_DEPTH / target_depth;
                        float reverse_scale = target_depth / SIGNAL_DEPTH;

                        float intransit_residual = 0;
                        float sigi = 0;
                        for (int i = 0; i < signal_x_size; i++) {
                            sigi = (1 - signal[i]) * reverse_scale;
                            intransit_residual = intransit_residual + ((data[i] - (1 - sigi)) * (data[i] - (1 - sigi))) * dy[i];
                        }
                        float current_stat = intransit_residual + ootr[tid] - summed_edge_effect_correction;
                        out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = current_stat;
                        depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = target_depth;
                        // if(tid == 12274 && y == 13){
                        /*if(y == 13){
                            //printf("intransit_residual:%f , ootr: %f, summed_edge_effect_correction:%f , current_stat:, target_depth: %f,z:%d\n",
                            //intransit_residual,ootr[tid],summed_edge_effect_correction,target_depth,z);
                            //printf("tid:%d\n",tid);
                            //printf("intransit_residual: %f, ootr: , summed_edge_effect_correction:, current_stat:, target_depth:%f ,z:%d\n",
                            //intransit_residual,target_depth,z);
                        }*/
                    }
                    else{
                        if(tid < *mean_x_size){
                        out[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = datapoints;
                        depths[tid+y*(*mean_x_size) + z*(*mean_x_size)*(*in_duration_size)] = 0.0;
                        }
                    }
                }
            }
        }

        __global__ void findBestFit(
            float *temp_result_index,
            float *temp_result_Residual,
            float *temp_result_depth,
            float *lowest_residuals,
            float *depths,
            int *mean_x_size,
            int *mean_size,
            int *duration_size,
            int *datapoints,
            int *iter_flag_gpu,int *single_calc_periods_arr_gpu,int *period_size_gpu

        ){
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            //int y = threadIdx.y + blockIdx.y * blockDim.y;

            if(tid + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu) < (*period_size_gpu)){
                if(tid < *single_calc_periods_arr_gpu){
                    int tid_output = (tid + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu));

                    float *lowest_residual ;//= lowest_residuals + tid*(*mean_x_size) + y*(*duration_size)*(*mean_x_size);
                    float *depth;// = depths + tid*(*mean_x_size) + y*(*duration_size)*(*mean_x_size);
                    float theLowestResidual = (*datapoints);
                    float theIndex = 0;
                    float theDepth = 0;
                    for (int i = 0; i < (*duration_size) - 1; i++){
                        lowest_residual = lowest_residuals + tid*(*duration_size)*(*mean_x_size) + i*(*mean_x_size);
                        depth = depths + tid*(*duration_size)*(*mean_x_size) + i*(*mean_x_size);
                        //printf("tid:%d,i:%d,lowest_residual:%f,depth:%f\n",tid,i,*(lowest_residual),*(depth));
                        for (int j = 0; j < mean_size[i]; j++){
                            if (lowest_residual[j] < theLowestResidual){
                                theLowestResidual = lowest_residual[j];
                                theDepth = depth[j];
                                theIndex = i;
                            }
                        }
                    }
                    //printf("tid:%d,tid_output:%d,mean_size:%d\n",tid,tid_output,mean_size[(*duration_size) - 1]);

                    temp_result_index[tid_output] = theIndex;
                    temp_result_Residual[tid_output] = theLowestResidual;
                    temp_result_depth[tid_output] = theDepth;
                    /*if(tid == 0 && *iter_flag_gpu == 1){
                        printf("y_output:%d,y:%d, tid:%d, theLowestResidual:%f, theIndex:%f, theDepth:%f\\n",y_output,y,tid,theLowestResidual,theIndex,theDepth);
                    }*/
                }
            }
        }

        __global__ void updateIterFlag(int *iter_flag_gpu){
        *iter_flag_gpu = *iter_flag_gpu + 1;
        //printf("iter_flag_gpu:%d\\n",*iter_flag_gpu);
        }
    }
    ''')
    return cupyFun

if "__main__" == __name__:
    x1 = cp.arange(1, dtype=cp.float32)
    x2 = cp.arange(1, dtype=cp.float32)
    y = cp.zeros((1), dtype=cp.float32)
    module = cp.RawModule(code=getCuPyFun())
    ker_fold = module.get_function('test_multiply')
    ker_fold((1,), (1024,), (x1, x2, y,1000))
    print(type(x1),type(x2),type(y))
