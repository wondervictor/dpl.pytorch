from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.spmmax_pooling import SPMMaxPoolingFunction


class SPMMaxPooling(Module):
    def __init__(self):
        super(SPMMaxPooling, self).__init__()

    def forward(self, features, shapes, rois):
        return SPMMaxPoolingFunction()(features, shapes, rois)
