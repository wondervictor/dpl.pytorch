# -*- coding: utf-8 -*-

"""
Description: Test the DPL Model
Author: wondervictor
"""

import os
import sys
import heapq
import torch
import random
import cPickle
import argparse
import numpy as np
import torch.utils.data
from torch.autograd import Variable

import models.dpl as model
from datasets import pascal_voc
from datasets import utils as data_utils

sys.path.append('./lib')
from model.nms import nms

parser = argparse.ArgumentParser()
parser.add_argument('--basemodel', type=str, default='vgg', help='base cnn model:[vgg, resnet34, resnet50]')
parser.add_argument('--cuda', action='store_true', help='use GPU to train')
parser.add_argument('--dataset', type=str, default='VOC2012', help='training dataset:[VOC2012, VOC2007, COCO]')
parser.add_argument('--data_dir', type=str, required=True, help='parameters storage')
parser.add_argument('--name', type=str, required=True, help='expriment name')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--num_class', type=int, default=20, help='label classes')
parser.add_argument('--param', type=str, required=True, help='model params path')
parser.add_argument('--proposal', type=str, default='selective_search',
                    help='proposal type:[selective_search, dense_box]')


opt = parser.parse_args()
print(opt)

test_dataset = pascal_voc.PASCALVOC(
    data_dir=opt.data_dir,
    imageset='test',
    roi_path='./data/',
    roi_type=opt.proposal,
    devkit='./devkit/'
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=data_utils.collate_fn
)

batch_size = 1
dpl = model.DPL(use_cuda=opt.cuda, base=opt.basemodel, num_classes=opt.num_class)
dpl.load_state_dict(torch.load(opt.param))
dpl.eval()

print(dpl)
print("---------- DPL Model Init Finished -----------")

images = Variable(torch.FloatTensor(batch_size, 3, opt.img_size, opt.img_size))

if opt.cuda:
    dpl = dpl.cuda()
    images = images.cuda()


def load_data(v, data):
    v.data.resize_(data.size()).copy_(data)


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if len(dets) == 0:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def detect(net, im, shapes, boxes):
    _, _, box_scores = net(im, shapes, boxes)
    # box_scores: Nx20
    boxes = boxes.cpu().data.numpy()
    scores = box_scores.cpu().data.numpy()
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    return scores, pred_boxes


def test(net, output_dir):
    # output_dir = 'devkit/results/VOC2012/Main/comp2_cls_val_xxxx.txt'
    test_iter = iter(test_loader)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_images = len(test_dataset)
    num_classes = test_dataset.num_classes
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100

    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)

    thresh = -np.inf * np.ones(num_classes)

    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)

    top_scores = [[] for _ in xrange(num_classes)]

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    i = 0
    while i < len(test_loader):
        img, box, shapes = test_iter.next()
        load_data(images, img)
        boxes = Variable(torch.FloatTensor(box)).cuda()
        shapes = Variable(torch.FloatTensor(shapes)).cuda()

        scores, pred_boxes = detect(dpl, img, shapes, boxes)

        for j in xrange(0, num_classes):
            inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        print 'im_detect: {:d}/{:d}: {}'.format(i + 1, len(test_loader), test_dataset.image_index[i])
        i = i + 1

    for j in xrange(0, num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, 0.3)

    print 'Writing results'
    for i in xrange(num_images):
        for j in xrange(num_classes):
            if len(nms_dets[j][i]) == 0:
                continue
            cls_file = os.path.join(output_dir, 'comp4_det_test_'+test_dataset.classes[j]+'.txt')
            with open(cls_file, 'a') as f:
                tmp = np.argmax(nms_dets[j][i][:, 4])
                f.write(test_dataset.image_index[i]+' '+str(nms_dets[j][i][tmp, 0])
                        + ' ' + str(nms_dets[j][i][tmp, 1])
                        + ' ' + str(nms_dets[j][i][tmp, 2])
                        + ' ' + str(nms_dets[j][i][tmp, 3])+'\n')

    test_dataset.evaluate_detections(nms_dets, output_dir)


if __name__ == '__main__':

    print("----------------Start to Test-------------------")
    test(dpl, 'evaluation/VOCdevkit/results/VOC2012/Main/')
    print("----------------FinishTesting-------------------")
