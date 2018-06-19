# -*- coding: utf-8 -*-
"""
description: Dataset Utils
"""

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    transposed = zip(*batch)
    # imgs = default_collate(transposed[0])
    lbl = default_collate(transposed[1])
    images = transposed[0]
    num_images = len(images)
    shapes = transposed[3]

    max_shape = np.max(shapes, axis=0)
    imgs = torch.zeros((num_images, 3, max_shape, max_shape))
    for i in xrange(num_images):
        img_size = images[i].size()
        imgs[i, :, 0:img_size[1], 0:img_size[2]] = images[i]
    boxes = []
    box = transposed[2]
    for i in xrange(len(transposed[2])):
        boxes += [[i] + b.tolist() for b in box[i]]
    boxes = np.array(boxes)
    shapes = default_collate(transposed[3])
    return imgs, lbl, boxes, shapes


def test_collate_fn(batch):
    transposed = zip(*batch)
    # imgs = default_collate(transposed[0])
    images = transposed[0]
    num_images = len(images)
    shapes = transposed[2]

    max_shape = np.max(shapes, axis=0)
    imgs = torch.zeros((num_images, 3, max_shape, max_shape))
    for i in xrange(num_images):
        img_size = images[i].size()
        imgs[i, :, 0:img_size[1], 0:img_size[2]] = images[i]
    boxes = []
    box = transposed[1]
    for i in xrange(len(box)):
        boxes += [[i] + b.tolist() for b in box[i]]
    boxes = np.array(boxes)
    shapes = default_collate(shapes)
    return imgs, boxes, shapes

