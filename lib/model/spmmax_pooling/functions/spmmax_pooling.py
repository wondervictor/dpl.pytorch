import torch
from torch.autograd import Function
from .._ext import spmmax_pooling


class SPMMaxPoolingFunction(Function):

    def __init__(self):
        super(SPMMaxPoolingFunction, self).__init__()
        self.num_rois = 0
        self.batch_size = 0
        self.feature_size = 0
        self.num_grids = 8
        self.max_id = None

    def forward(self, x, shapes, rois):
        num_rois, feature_size = x.size()
        batch_size, _ = shapes.size()
        self.num_rois = num_rois
        self.feature_size = feature_size
        self.batch_size = batch_size

        output = x.new(batch_size, self.num_grids, feature_size).zero_()

        max_id = torch.zeros((batch_size, self.num_grids, feature_size)).int()-1  # x.new(batch_size, self.num_grids, feature_size).zero_().int()
        if output.is_cuda:
            max_id = max_id.cuda()
        if x.is_cuda:
            spmmax_pooling.spmmax_pooling_forward_cuda(
                x, shapes, rois, output, max_id
            )
        else:
            spmmax_pooling.spm_max_pooling_forward(
                x, shapes, rois, output, max_id
            )
        self.max_id = max_id
        print self.max_id
        return output

    def backward(self, grad_input):
        grad_output = grad_input.new(self.num_rois, self.feature_size).zero_()
        if grad_input.is_cuda:
            spmmax_pooling.spmmax_pooling_backward_cuda(
                grad_input, self.max_id, grad_output
            )
        else:
            spmmax_pooling.spm_max_pooling_backward(
                grad_input, self.max_id, grad_output
            )
        return grad_output, None, None
