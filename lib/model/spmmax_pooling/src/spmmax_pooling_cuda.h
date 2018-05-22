// SPM Max Pooling CUDA
// Author: Vic Chan
// Date: 2018/5/21

int spmmax_pooling_forward_cuda(THCudaTensor* x, THCudaTensor* shapes, THCudaTensor* rois, THCudaTensor* output,
                                THCudaIntTensor* max_ids);

int spmmax_pooling_backward_cuda(THCudaTensor* grad_input, THCudaIntTensor* max_ids, THCudaTensor* grad_output);
