// SPM Max Pooling
// Author: Vic Chan
// Date: 2018/5/21

#ifdef __cplusplus
extern "C" {
#endif

#include <TH/TH.h>
#include "spmmax_pooling.h"

int spm_max_pooling_forward(THFloatTensor* x, THFloatTensor* shapes, THFloatTensor* rois, THFloatTensor* output,
                            THIntTensor* max_ids) {
  float spm[32] = {0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5, 0.5,
                   1, 0.5, 1, 0, 1, 0, 0.33, 0, 1, 0.33, 0.67, 0, 1, 0.67, 1};
  int num_grids = 8;
  int num_rois = (int)THFloatTensor_size(rois, 0);
  int feature_size = (int)THFloatTensor_size(x, 1);
  int batch_size = (int)THFloatTensor_size(shapes, 0);
  int output_size = batch_size * num_grids * feature_size;

  float* x_data = THFloatTensor_data(x);
  float* shaps_data = THFloatTensor_data(shapes);
  float* rois_data = THFloatTensor_data(rois);
  float* output_data = THFloatTensor_data(output);
  int* maxid_data = THIntTensor_data(max_ids);
  // memset(output_data, 0, output_size);
  // memset(maxid_data, -1, output_size);
  for(int i = 0; i < num_rois; i ++) {
    int batch_id = (int)rois_data[i * 5 + 0];
    float center_x = (rois_data[i*5+1] + rois_data[i*5+3])/(2*shaps_data[batch_id*2+0]);
    float center_y = (rois_data[i*5+2] + rois_data[i*5+4])/(2*shaps_data[batch_id*2+1]);
    for(int j = 0; j < num_grids; j ++) {
      if (center_x >= spm[j*4+0] && center_x < spm[j*4+1] && center_y >= spm[j*4+2] && center_y < spm[j*4+3]) {
        for(int c = 0; c < feature_size; c++) {
          if (maxid_data[batch_id*num_grids*feature_size+j*feature_size+c] == -1 ||x_data[i*feature_size+c] > output_data[batch_id*num_grids*feature_size+j*feature_size+c]) {

            output_data[batch_id*num_grids*feature_size+j*feature_size+c] = x_data[i*feature_size+c];
            maxid_data[batch_id*num_grids*feature_size+j*feature_size+c] = i;
          }
        }
      }
    }
  }
  return 1;
}

int spm_max_pooling_backward(THFloatTensor* grad_input, THIntTensor* max_ids, THFloatTensor* grad_output) {
  // grad_input: batch_id x num_grids x feature_size
  // grad_output: num_rois x feature_size
  float* gradout_data = THFloatTensor_data(grad_output);
  float* gradin_data = THFloatTensor_data(grad_input);
  int* maxid_data = THIntTensor_data(max_ids);
  int feature_size = (int)THFloatTensor_size(grad_output, 1);
  int batch_size = (int)THIntTensor_size(max_ids, 0);
  int num_grids = 8;
  for(int i = 0; i < batch_size; i ++) {
    for (int j = 0; j < num_grids; j ++) {
      for (int c = 0; c < feature_size; c++) {
        int idx = i*num_grids*feature_size + j * feature_size + c;
        if (maxid_data[idx] != -1) {
          gradout_data[maxid_data[idx] * feature_size + c] += gradin_data[idx];
        }
      }
    }
  }
  return 1;
}

#ifdef __cplusplus
}
#endif
