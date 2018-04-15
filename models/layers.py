# -*- coding: utf-8 -*-

"""
Description: Layers Definition
Author: wondervictor
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MultiSigmoidCrossEntropyLoss(nn.Module):
    """ MultiClass Sigmoid Cross Entropy Loss
    Inputs:
        - `s`: the input score
        - `y`: the target label

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)`

    """
    def __init__(self):
        super(MultiSigmoidCrossEntropyLoss, self).__init__()

    def forward(self, s, y):
        # s: batch * class
        # y: batch * class
        m = y * torch.log(F.sigmoid(s)) + (1-y)*torch.log(1-F.sigmoid(s))
        m = torch.sum(m, dim=1)
        m = torch.mean(m)
        return m

    def __repr__(self):
        return self.__class__.__name__


class ROIPooling1(nn.Module):
    """ ROI Pooling Version1
    Args:
        pool_size (int): ROI Pooling size
        scale (float): scale for input features which were downsampled by pooling layers in convolution layers
    Inputs:
        - `features`: the input features
        - `rois`: the target label

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)`

    """
    def __init__(self, pool_size, scale):
        super(ROIPooling1, self).__init__()
        self.pool_size = pool_size
        self.scale = scale

    def forward(self, features, rois):
        # features  B*C*H*W
        # rois      B*N*4 (px, py, qx, qy)
        batch_size, num_ch, height, width = features.size()
        num_rois = rois.size()[1]
        output = Variable(torch.FloatTensor(batch_size, num_rois, num_ch, self.pool_size, self.pool_size))
        for b in xrange(batch_size):
            for roindex in xrange(num_rois):
                px, py, qx, qy = np.round(rois[b, roindex].data.cpu().numpy() * self.scale).astype(int)
                roi_width = max(qx - px + 1, 1)
                roi_height = max(qy - py + 1, 1)
                # pool kernel size
                psize_w = int(np.ceil(float(roi_width) / self.pool_size))
                psize_h = int(np.ceil(float(roi_height) / self.pool_size))
                pad_top = (psize_h * self.pool_size - roi_height)/2
                pad_left = (psize_w * self.pool_size - roi_width)/2
                pad_bottom = psize_h * self.pool_size - roi_height - pad_top
                pad_right = psize_w * self.pool_size - roi_width - pad_left
                maxpool = nn.MaxPool2d((psize_h, psize_w), stride=(psize_h, psize_w))
                feature_part = features[b, :, py:qy+1, px: qx+1]
                pad = nn.ZeroPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))
                feature_part = pad(feature_part.unsqueeze(0)).squeeze(0)
                output[b, roindex] = maxpool(feature_part)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ("Pool Size:{} Scale: {}".format(self.pool_size, self.scale))


class ROIPooling(nn.Module):

    """ ROI Pooling V2 for arbitray rois
    Args:
        pool_size (int): ROI Pooling size
        scale (float): scale for input features which were downsampled by pooling layers in convolution layers
    Inputs:
        - `features`: the input features
        - `rois`: the target label

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)`

    """
    def __init__(self, pool_size, scale):
        super(ROIPooling, self).__init__()
        self.pool_size = pool_size
        self.scale = scale

    def forward(self, features, rois):
        # features: N*C*H*W
        # rois: N*5
        assert len(rois.size()) == 2 and rois.size()[1] == 5, "the shape of rois should be `Nx5`"
        batch_size, num_ch, height, width = features.size()
        num_rois = rois.size()[0]
        output = Variable(torch.FloatTensor(num_rois, num_ch, self.pool_size, self.pool_size))
        output_batch_id = np.zeros(num_rois, dtype=np.int32)
        for roiidx, roi in enumerate(rois):
            batch_id = int(roi[0].data[0])
            px, py, qx, qy = np.round(roi.data[1:].cpu().numpy() * self.scale).astype(int)
            roi_width = max(qx - px + 1, 1)
            roi_height = max(qy - py + 1, 1)
            # pool kernel size
            psize_w = int(np.ceil(float(roi_width) / self.pool_size))
            psize_h = int(np.ceil(float(roi_height) / self.pool_size))
            pad_top = (psize_h * self.pool_size - roi_height) / 2
            pad_left = (psize_w * self.pool_size - roi_width) / 2
            pad_bottom = psize_h * self.pool_size - roi_height - pad_top
            pad_right = psize_w * self.pool_size - roi_width - pad_left
            maxpool = nn.MaxPool2d((psize_h, psize_w), stride=(psize_h, psize_w))
            feature_part = features[batch_id, :, py:qy + 1, px: qx + 1]
            pad = nn.ZeroPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))
            feature_part = pad(feature_part.unsqueeze(0)).squeeze(0)
            output[roiidx] = maxpool(feature_part)
            output_batch_id[roiidx] = batch_id

        return output, output_batch_id

    def __repr__(self):
        return self.__class__.__name__ + ("Pool Size:{} Scale: {}".format(self.pool_size, self.scale))


class PatchPooling(nn.Module):

    """ PatchPooling Layer
    Args:
        batch_size (int): batchsize of the patches
    Inputs:
        - `features`: the input features
        - `rois`: the target label

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)`

    """
    def __init__(self, batch_size):
        super(PatchPooling, self).__init__()
        self.batch_size = batch_size

    def forward(self, patches, patch_ids):
        # patches: torch.FloatTensor, NxC
        # patch_ids: numpy array, Nx1
        num_patch, num_features = patches.size()
        output = Variable(torch.FloatTensor(self.batch_size, num_features))
        for i in xrange(self.batch_size):
            output[i] = torch.max(patches[np.where(patch_ids == 0), :].squeeze(0), dim=0)[0]
        return output


def __test__roi():

    roi_pooling = ROIPooling(pool_size=7, scale=0.5)
    features = Variable(torch.randn((2, 3, 40, 40)))
    # rois = Variable(torch.FloatTensor([[[2, 14, 17, 39], [4, 26, 29, 39], [1, 2, 39, 39], [1, 4, 23, 34]], [[5, 3, 30, 27], [12, 12, 30, 34], [5, 14, 39, 38], [6, 10, 22, 35]]]))
    rois = Variable(torch.FloatTensor([[1, 2, 14, 17, 39], [0, 4, 26, 29, 39], [0, 1, 2, 39, 39], [0, 1, 4, 23, 34], [0, 5, 3, 30, 27], [1, 12, 12, 30, 34], [1, 5, 14, 39, 38], [0, 6, 10, 22, 35]]))

    print("Features: {}".format(features.size()))
    print("ROIS: {}".format(rois.size()))

    output, batch_id = roi_pooling(features, rois)
    print(batch_id)
    print(output.size())
    output = output.view(-1, 147)
    patch_pool = PatchPooling(batch_size=2)
    output = patch_pool(output, batch_id)
    print(output.size())

# __test__roi()


class SPMMaxPooling(nn.Module):

    def __init__(self):
        super(SPMMaxPooling, self).__init__()

    def forward(self, x):
        pass
