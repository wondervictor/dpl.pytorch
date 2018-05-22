// SPM Max Pooling CUDA
// Author: Vic Chan
// Date: 2018/5/21
//#ifdef __cplusplus
//extern "C" {
//#endif

#include <stdio.h>
#include "spmmax_pooling_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

__global__ void spmmax_pooling_forward(const int num_threads, const int batch_size, const int num_grids, const int feature_size,
                                       const int num_rois, const float* x_data, const float* shapes_data,
                                       const float* rois_data, float* output_data, int* max_ids_data){

  __shared__ float spm[32] = {0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5, 0.5,
                   1, 0.5, 1, 0, 1, 0, 0.33, 0, 1, 0.33, 0.67, 0, 1, 0.67, 1};

  CUDA_1D_KERNEL_LOOP(thread_idx, num_threads) {
    // int thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int roi_id = thread_idx/(num_grids * feature_size);
    int grid_id = (thread_idx - roi_id * num_grids * feature_size)/feature_size;
    int feature_id = thread_idx - roi_id * num_grids * feature_size - grid_id * feature_size;

    int batch_id = (int)rois_data[roi_id*5];
    float center_x = (rois_data[roi_id*5+1] + rois_data[roi_id*5+3])/(2*shapes_data[batch_id*2+0]);
    float center_y = (rois_data[roi_id*5+2] + rois_data[roi_id*5+4])/(2*shapes_data[batch_id*2+1]);

    if (center_x >= spm[grid_id*4+0] && center_x < spm[grid_id*4+1]
        && center_y >= spm[grid_id*4+2] && center_y < spm[grid_id*4+3]) {
      int idx = batch_id*num_grids*feature_size + grid_id * feature_size + feature_id;
      if (x_data[roi_id*feature_size + feature_id] > output_data[idx]) {
        atomicExch(output_data+idx, x_data[roi_id*feature_size + feature_id]);
        atomicExch(max_ids_data+idx, roi_id);
      }
    }

  }
}

__global__ void spmmax_pooling_backward(const int num_threads, const int batch_size, const int num_grids, const int feature_size,
                                        const int num_rois, const float* grad_input_data, float* grad_output_data, const int* max_ids_data) {
  CUDA_1D_KERNEL_LOOP(thread_idx, num_threads) {
    // int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_id = thread_idx / (num_grids * feature_size);
    int grid_id = (thread_idx - (num_grids * feature_size * batch_id)) / feature_size;
    int feature_id = thread_idx - num_grids * feature_size * batch_id - feature_size * grid_id;

    int idx = batch_id * num_grids * feature_size + grid_id * feature_size + feature_id;
    if (max_ids_data[idx] == -1) {
      atomicAdd(grad_output_data + max_ids_data[idx] * feature_size + feature_id, grad_input_data[idx]);
    }

  }
}

int spmmax_pooling_forward_kernel(const int batch_size, const int num_grids, const int feature_size, const int num_rois,
                                  const float* x_data,const float* shapes_data, const float* rois_data, float* output_data, int* max_ids_data,
                                  cudaStream_t stream) {
  int output_size = num_rois * num_grids * feature_size;
  cudaError_t err;

  float spm[32] = {0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5, 0.5,
                   1, 0.5, 1, 0, 1, 0, 0.33, 0, 1, 0.33, 0.67, 0, 1, 0.67, 1};

  const int kThreadsPerBlock = 1024;
  dim3 threads(kThreadsPerBlock);
  int block = (output_size + kThreadsPerBlock - 1)/kThreadsPerBlock;
  dim3 blocks(block);
  spmmax_pooling_forward<<<blocks, threads, 0, stream>>>(output_size, batch_size, num_grids, feature_size, num_rois, x_data,
      shapes_data,rois_data, output_data, max_ids_data);
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}

int spmmax_pooling_backward_kernel(const int batch_size, const int num_grids, const int feature_size,
                                   const int num_rois, const float* grad_input_data, float* grad_output_data,  int* max_ids_data, cudaStream_t stream) {

  const int kThreadsPerBlock = 1024;
  int output_size = batch_size * num_grids * feature_size;
  cudaError_t err;
  dim3 threads(kThreadsPerBlock);
  int block = (output_size + kThreadsPerBlock - 1)/kThreadsPerBlock;
  dim3 blocks(block);
  spmmax_pooling_backward<<<blocks, threads, 0, stream>>>(output_size, batch_size, num_grids, feature_size, num_rois, grad_input_data,
      grad_output_data, max_ids_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit(-1);
  }

  return 1;
}


//#ifdef __cplusplus
//}
//#endif
