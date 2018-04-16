# -*- coding: utf-8 -*-

"""
Description: Basic Network (ResNet, VGG, DenseNet.....)
Author: wondervictor
"""

import torch
import torchvision
import torch.nn as nn


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True)
        self.layers = nn.Sequential(*list(vgg.children())[:-1])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, images):
        return self.layers(images)


