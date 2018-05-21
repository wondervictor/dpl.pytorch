// SPM Max Pooling CUDA
// Author: Vic Chan
// Date: 2018/5/21

#ifdef __cplusplus
extern "C" {
#endif

int spmmax_pooling_forward_kernel(int batch_size, int num_grids, int feature_size, int num_rois, float* x_data,
                                  float* shapes_data, float* rois_data, float* output_data, int64_t* max_ids_data);

int spmmax_pooling_backward_kernel(int batch_size, int num_grids, int feature_size, int num_rois, float* grad_input_data,
                                   float* grad_output_data, int64_t* max_ids_data);

#ifdef __cplusplus
}
#endif