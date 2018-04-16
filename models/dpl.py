# -*- coding: utf-8 -*-

"""
Description: Deep Patch Learning Model
Author: wondervictor
"""

import torch
import torch.nn as nn
import layers
import basenet


class DPL(nn.Module):

    def __init__(self, batch_size, use_cuda, num_classes=20, base='vgg', use_relation=False):
        super(DPL, self).__init__()
        if base == 'vgg':
            self.cnn = basenet.VGG16()
        else:
            self.cnn = basenet.VGG16()

        self.cuda = use_cuda
        self.roi_pooling = layers.ROIPooling(pool_size=7, scale=0.0625, cuda=self.cuda)
        self.patch_pooling = layers.PatchPooling(batch_size=batch_size, cuda=self.cuda)

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

        roi_output, batch_id = self.roi_pooling(features, rois)
        # roi_output N*512*7*7

        patch_features = roi_output.view(-1, 512*7*7)
        # patch_features: N * (512*7*7)

        patch_features = self.fcs(patch_features)
        # patch_features: N * 1024

        batch_features = self.patch_pooling(patch_features)
        # batch_features: B * 1024

        output = self.out(batch_features)
        # output: B * 20

        return output


