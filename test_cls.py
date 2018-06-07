# -*- coding: utf-8 -*-

"""
Description: Test the DPL Model
Author: wondervictor
"""

import os
import torch
import random
import argparse
import numpy as np
import torch.utils.data
from torch.autograd import Variable

import models.dpl as model
from datasets import pascal_voc
from datasets import utils as data_utils

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

val_dataset = pascal_voc.PASCALVOC(
    img_size=opt.img_size,
    data_dir=opt.data_dir,
    imageset='test',
    roi_path='./data/',
    roi_type=opt.proposal,
    devkit='./devkit/',
    test_mode=True,
    flip=False
)


test_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
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


def test(net, output_dir):
    # output_dir = 'devkit/results/VOC2012/Main/comp2_cls_val_xxxx.txt'
    test_iter = iter(test_loader)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    while i < len(test_loader):
        img, box, shapes = test_iter.next()
        load_data(images, img)
        boxes = Variable(torch.FloatTensor(box)).cuda()
        shapes = Variable(torch.FloatTensor(shapes)).cuda()
        cls_score1, cls_score2, _ = net(images, shapes, boxes)
        cls_score = cls_score1 + cls_score2
        cls_score = cls_score.cpu().squeeze(0).data.numpy()
        for m in xrange(opt.num_class):
            cls_file = os.path.join(output_dir, 'comp2_cls_test_' + val_dataset.classes[m] + '.txt')
            with open(cls_file, 'a') as f:
                f.write(val_dataset.image_index[i] + ' ' + str(cls_score[m]) + '\n')

        print 'im_cls: {:d}/{:d}: {}'.format(i + 1, len(test_loader), val_dataset.image_index[i])
        i = i + 1


if __name__ == '__main__':

    print("----------------Start to Test-------------------")
    test(dpl, 'evaluation/VOCdevkit/results/VOC2012/Main/')
    print("----------------FinishTesting-------------------")
