# -*- coding: utf-8 -*-
"""
description: Dataset Utils
"""

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    transposed = zip(*batch)
    imgs = default_collate(transposed[0])
    lbl = default_collate(transposed[1])
    boxes = []
    box = transposed[2]
    for i in xrange(len(transposed[2])):
        boxes += [[i] + b.tolist() for b in box[i]]
    boxes = np.array(boxes)
    shapes = default_collate(transposed[3])

    return imgs, lbl, boxes, shapes


