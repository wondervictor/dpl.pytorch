// SPM Max Pooling CUDA
// Author: Vic Chan
// Date: 2018/5/21

#ifdef __cplusplus
extern "C" {
#endif

#include <THC/THC.h>
#include <cxxabi.h>

extern THCState *state;
#include "spmmax_pooling_cuda.h"
#include "spmmax_pooling_kernel.h"

int spmmax_pooling_forward_cuda(THCudaTensor *x, THCudaTensor *shapes, THCudaTensor *rois, THCudaTensor *output,
                                THCudaIntTensor *max_ids) {

  // x:       num_rois * feature_size
  // shapes:  batch_size * 2
  // rois:    num_rois * 5
  // output:  batch_size * num_grids * feature_size
  // max_ids: batch_size * num_grids * feature_size

  cudaStream_t stream = THCState_getCurrentStream(state);
  int num_rois = (int)THCudaTensor_size(state, x, 0);
  int feature_size = (int)THCudaTensor_size(state, x, 1);
  int batch_size = (int)THCudaTensor_size(state, shapes, 0);
  int num_grids = 8;

  float* x_data = THCudaTensor_data(state, x);
  float* shapes_data = THCudaTensor_data(state, shapes);
  float* rois_data = THCudaTensor_data(state, rois);
  float* output_data = THCudaTensor_data(state, output);
  int64_t* max_ids_data = THCudaIntTensor_data(state, max_ids);

  spmmax_pooling_forward_kernel(batch_size, num_grids, feature_size, num_rois, x_data, shapes_data,
                                rois_data, output_data, max_ids_data);

  return 1;
}

int spmmax_pooling_backward_cuda(THCudaTensor *grad_input, THCudaIntTensor *max_ids, THCudaTensor *grad_output) {
  // grad_input:  batch_size * num_grids * feature_size
  // max_ids:     batch_size * num_grids * feature_size
  // grad_output: num_rois x feature_size
  cudaStream_t stream = THCState_getCurrentStream(state);
  int num_rois = (int)THCudaTensor_size(state, grad_output, 0);
  int feature_size = (int)THCudaTensor_size(state, grad_output, 1);
  int batch_size = (int)THCudaTensor_size(state, grad_input, 0);
  int  num_grids = 8;

  float* grad_output_data = THCudaTensor_data(state, grad_output);
  float* grad_input_data = THCudaTensor_data(state, grad_input);
  int64_t* max_ids_data = THCudaIntTensor_data(state, max_ids);

  spmmax_pooling_backward_kernel(batch_size, num_grids, feature_size, num_rois, grad_input_data,
                                 grad_output_data, max_ids_data);

  return 1;

}

#ifdef __cplusplus
}
#endif