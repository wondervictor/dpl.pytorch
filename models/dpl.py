# -*- coding: utf-8 -*-

"""
Description: Deep Patch Learning Model
Author: wondervictor
"""

import torch
import torch.nn as nn
import numpy as np
import layers
import basenet


class DPL(nn.Module):

    def __init__(self, batch_size, use_cuda, num_classes=20, base='vgg', use_relation=False):
        super(DPL, self).__init__()
        if base == 'vgg':
            self.cnn = basenet.VGG16()
        else:
            self.cnn = basenet.VGG16()

        self.use_cuda = use_cuda
        self.roi_align = layers.ROIAlign(out_size=7, spatial_scale=0.0625)
        self.patch_pooling = layers.PatchPooling(cuda=self.use_cuda)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Dropout(0.5)
        )

        self.out = nn.Linear(1024, num_classes)

    def forward(self, images, rois):
        # N denotes the num_rois, B denotes the batchsize
        # images: B*C*H*W
        # rois:   N*5

        features = self.cnn(images)
        # features: B*512*H*W
        batch_size = features.size()[0]
        roi_output = self.roi_align(features, rois)
        # roi_output N*512*7*7
        num_rois = rois.size()[0]
        output_batch_id = np.zeros(num_rois, dtype=np.int32)
        for roiidx, roi in enumerate(rois):
            batch_id = int(roi[0].data[0])
            output_batch_id[roiidx] = batch_id

        patch_features = roi_output.view(-1, 512*7*7)
        # patch_features: N * (512*7*7)

        patch_features = self.fcs(patch_features)
        # patch_features: N * 1024
        batch_features = self.patch_pooling(batch_size, patch_features, output_batch_id)
        # batch_features: B * 1024

        output = self.out(batch_features)
        # output: B * 20

        return output


