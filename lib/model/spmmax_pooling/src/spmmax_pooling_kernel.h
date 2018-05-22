// SPM Max Pooling CUDA
// Author: Vic Chan
// Date: 2018/5/21

#ifdef __cplusplus
extern "C" {
#endif

int spmmax_pooling_forward_kernel(const int batch_size, const int num_grids, const int feature_size, const int num_rois,
                                  const float* x_data,const float* shapes_data, const float* rois_data, float* output_data, int* max_ids_data, cudaStream_t stream);

int spmmax_pooling_backward_kernel(const int batch_size, const int num_grids, const int feature_size,
                                   const int num_rois, const float* grad_input_data, float* grad_output_data, int* max_ids_data, cudaStream_t stream);

#ifdef __cplusplus
}
#endif