// SPM Max Pooling
// Author: Vic Chan
// Date: 2018/5/21

int spm_max_pooling_forward(THFloatTensor* x, THFloatTensor* shapes, THFloatTensor* rois, THFloatTensor* result,
                            THLongTensor* max_ids);
int spm_max_pooling_backward(THFloatTensor* grad_input, THLongTensor* max_ids, THFloatTensor* grad_output);