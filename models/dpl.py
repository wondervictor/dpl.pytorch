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


class PatchHeadNetwork(nn.Module):

    def __init__(self, use_cuda, num_classes, use_relation=False):
        super(PatchHeadNetwork, self).__init__()

        self.roi_align = layers.ROIAlign(out_size=7, spatial_scale=0.0625)

        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Dropout(0.5)
        )

        self.patch_encoder = nn.Linear(4096, 256)
        self.cls_score1 = nn.Linear(256*8, num_classes)
        self.cls_score2 = nn.Linear(4096, num_classes)
        self.patch_pooling = layers.MaxPatchPooling(use_cuda)
        self.spm_pooling = layers.SPMMaxPooling()

    def forward(self, features, shapes, rois):
        # N denotes the num_rois, B denotes the batchsize
        # features: B*C*H*W
        # rois:   N*5
        # shapes: B*2

        batch_size = features.size()[0]

        roi_output = self.roi_align(features, rois)
        # N*512*7*7
        patch_features = roi_output.view(-1, 512 * 7 * 7)
        # N*25088

        num_rois = rois.size()[0]
        output_batch_id = np.zeros(num_rois, dtype=np.int32)
        for i in xrange(num_rois):
            batch_id = int(rois[i].data[0])
            output_batch_id[i] = batch_id

        patch_features = roi_output.view(-1, 512*7*7)
        # patch_features: N * (512*7*7)

        patch_features = self.fc(patch_features)
        # patch_features: N * 4096
        encoded_features = self.patch_encoder(patch_features)
        spm_features = self.spm_pooling(encoded_features, shapes, rois)
        spm_features = spm_features.view(batch_size, 256 * 8)
        cls_score1 = self.cls_score1(spm_features)
        cls_score2_features = self.cls_score2(patch_features)
        cls_score2 = self.patch_pooling(cls_score2_features, batch_size, output_batch_id)

        return cls_score1, cls_score2


class DPL(nn.Module):

    def __init__(self, use_cuda, num_classes=20, base='vgg', use_relation=False):
        super(DPL, self).__init__()
        if base == 'vgg':
            self.cnn = basenet.VGG16()
        else:
            self.cnn = basenet.VGG16()

        self.use_cuda = use_cuda
        self.head_network = PatchHeadNetwork(use_cuda=use_cuda, num_classes=num_classes, use_relation=use_relation)

    def forward(self, images, shapes, rois):
        features = self.cnn(images)
        cls_score1, cls_score2 = self.head_network(features, shapes, rois)
        return cls_score1, cls_score2

