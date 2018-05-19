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
import torch.optim as optim
from torch.autograd import Variable

import utils
import models.dpl as model
import models.layers as layers
from datasets import pascal_voc
from datasets import utils as data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--base model', type=str, default='vgg', help='base cnn model:[vgg, resnet, densenet]')
parser.add_argument('--cuda', action='store_true', help='use GPU to train')
parser.add_argument('--dataset', type=str, default='VOC2012', help='training dataset:[VOC2012, VOC2007, COCO]')
parser.add_argument('--data_dir', type=str, required=True, help='parameters storage')
parser.add_argument('--name', type=str, required=True, help='expriment name')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--num_class', type=int, default=20, help='label classes')
parser.add_argument('--param', type=str, required=True, help='model params path')


opt = parser.parse_args()
print(opt)


expr_dir = 'output/{}/'.format(opt.name)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


val_dataset = pascal_voc.PASCALVOC(
    img_size=opt.img_size,
    data_dir=opt.data_dir,
    imageset='val',
    roi_path='./data/',
    roi_type='dense_box', #selective_search',
    devkit='./devkit/'
)


test_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=data_utils.collate_fn
)

batch_size=1
dpl = model.DPL(batch_size=batch_size, use_cuda=opt.cuda)

criterion = layers.MultiSigmoidCrossEntropyLoss()

print(dpl)
print("---------- DPL Model Init Finished -----------")

log_dir = expr_dir+'log/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = utils.Logger(stdio=True, log_file=log_dir+"testing.log")
images = Variable(torch.FloatTensor(batch_size, 3, opt.img_size, opt.img_size))
labels = Variable(torch.FloatTensor(batch_size, opt.num_class))

if opt.cuda:
    criterion = criterion.cuda()
    dpl = dpl.cuda()
    images = images.cuda()
    labels = labels.cuda()

dpl.load_state_dict(torch.load(opt.param))

averager = utils.Averager()


def load_data(v, data):
    v.data.resize_(data.size()).copy_(data)


def test(net, criterion, output_dir):
    # output_dir = 'devkit/results/VOC2012/Main/comp2_cls_val_xxxx.txt'
    net.eval()
    test_iter = iter(test_loader)
    test_averager = utils.Averager()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    while i < len(test_loader):
        img, lbl, box = test_iter.next()
        load_data(images, img)
        load_data(labels, lbl)
        boxes = Variable(torch.FloatTensor(box)).cuda()
        output = net(images, boxes).squeeze(0)
        loss = criterion(output, labels)
        test_averager.add(loss)

        for m in xrange(opt.num_class):
            cls_file = os.path.join(output_dir, 'comp2_cls_val_' + val_dataset.classes[m] + '.txt')
            with open(cls_file, 'a') as f:
                f.write(val_dataset.image_index[i] + ' ' + str(output[m]) + '\n')

            print 'im_cls: {:d}/{:d}: {}'.format(i + 1, len(test_loader), val_dataset.image_index[i])
    print("Avg Loss: {}".format(test_averager.val()))
    # val_dataset.do_python_eval(output_dir)


if __name__ == '__main__':

    print("----------------Start to Test-------------------")
    test(dpl, criterion, 'evaluation/VOCdevkit/results/VOC2012/Main/')
    print("----------------FinishTesting-------------------")