# -*- coding: utf-8 -*-

"""
Description: Prepare dense box
"""

import os
import pickle
import numpy as np
import scipy.io as sio
import cv2

scales = 32 * np.arange(2, 9)
stepsize = 32
n_scales = len(scales)


def prepare_dense_box(dataset_dir, image_list, save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            boxes = pickle.load(f)
        return boxes
    boxes = {}
    for name in image_list:
        img = cv2.imread(os.path.join(dataset_dir, "{}.jpg".format(name)))
        print(os.path.join(dataset_dir, "{}.jpg".format(name)))
        h, w = img.shape[:2]
        rect = np.zeros((0, 4), dtype=np.float)
        for j in range(len(scales)):
            xs = range(1, w - scales[j] + 1, stepsize)
            if w % stepsize != 0:
                xs = xs + [w - scales[j] + 1]
            if len(xs) == 0:
                xs = [1]

            ys = range(1, h - scales[j] + 1, stepsize)
            if h % stepsize != 0:
                ys = ys + [h - scales[j] + 1]
            if len(ys) == 0:
                ys = [1]

            xs = np.array(xs)
            ys = np.array(ys)
            [x, y] = np.meshgrid(xs, ys)

            x = x.reshape(-1)
            y = y.reshape(-1)
            x[x < 1] = 1
            y[y < 1] = 1
            x1 = x + scales[j] - 1
            y1 = y + scales[j] - 1
            x1[x1 > w] = w
            y1[y1 > h] = h
            rect = np.vstack((rect, np.array([x, y, x1, y1]).T))
        boxes[name] = rect
    with open(save_path, 'wb') as f:
        pickle.dump(boxes, f)
    return boxes
