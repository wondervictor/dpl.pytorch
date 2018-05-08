# -*- coding: utf-8 -*-

"""
Description: Train the DPL Model
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
import dataset.dataset as dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--base model', type=str, default='vgg', help='base cnn model:[vgg, resnet, densenet]')
parser.add_argument('--cuda', action='store_true', help='use GPU to train')
parser.add_argument('--dataset', type=str, default='VOC2012', help='training dataset:[VOC2012, VOC2007, COCO]')
parser.add_argument('--epoch', type=int, default=100, help='training epoches')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--data_dir', type=str, required=True, help='parameters storage')
parser.add_argument('--log_interval', type=int, default=20, help='log messages interval')
parser.add_argument('--val_interval', type=int, default=5, help='validation interval')
parser.add_argument('--save_interval', type=int, default=5, help='save model interval')
parser.add_argument('--name', type=str, required=True, help='expriment name')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--num_class', type=int, default=20, help='label classes')


opt = parser.parse_args()
print(opt)

# fix the random seed
random_seed = 2781
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if not os.path.exists('output/'):
    os.mkdir('output/')

expr_dir = 'output/{}/'.format(opt.name)

if not os.path.exists(expr_dir):
    os.mkdir(expr_dir)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


train_dataset = dataset.PASCAL(data_path=opt.data_dir, train=True)
test_dataset = dataset.PASCAL(data_path=opt.data_dir, train=False)

def adjust_lr(_optimizer, _epoch):
    lr = opt.lr * 0.9 * (_epoch/5)
    for param_group in _optimizer.param_groups:
        param_group['lr'] = lr


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.uniform_(0, 0.5)


dpl = model.DPL(batch_size=opt.batch_size, use_cuda=opt.cuda)
dpl.apply(weights_init)
dpl.train()

criterion = layers.MultiSigmoidCrossEntropyLoss()

print(dpl)
print("---------- DPL Model Init Finished -----------")

log_dir = expr_dir+'log/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = utils.Logger(stdio=True, log_file=log_dir+"training.log")
images = Variable(torch.FloatTensor(opt.batch_size, 3, opt.img_size, opt.img_size))
labels = Variable(torch.FloatTensor(opt.batch_size, opt.num_class))

if opt.cuda:
    criterion = criterion.cuda()
    dpl = dpl.cuda()
    images = images.cuda()
    labels = labels.cuda()

param_dir = expr_dir+'param/'
if not os.path.exists(param_dir):
    os.mkdir(param_dir)

optimizer = optim.Adam([{"params": dpl.fcs.parameters()},{"params": dpl.out.parameters()}], lr=opt.lr)

averager = utils.Averager()


def load_data(v, data):
    v.data.resize_(data.size()).copy_(data)


def test(net, criterion):
    net.eval()
    test_iter = iter(test_loader)
    test_averager = utils.Averager()
    pass


def train_batch(net, data, criterion, optimizer):

    img, lbl, box = data

    load_data(images, img)
    load_data(labels, lbl)
    boxes = []
    for n in range(len(box)):
        boxes += [[n]+b.tolist() for b in box[n]]
    boxes = Variable(torch.FloatTensor(boxes)).cuda()

    output = net(images, boxes)
    loss = criterion(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

logger.log('starting to train')
iter_steps = 0
for epoch in xrange(opt.epoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):

        dpl.train()
        data = train_iter.next()
        _loss = train_batch(dpl, data, criterion, optimizer=optimizer)
        averager.add(_loss)
        iter_steps += 1
        i += 1
        if (iter_steps+1) % opt.log_interval == 0:
            logger.log('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.epoch, i, len(train_loader), averager.val()))

    averager.reset()
    if (epoch+1) % opt.val_interval == 0:
        pass
    if (epoch+1) % opt.save_interval == 0:
        torch.save(dpl.state_dict(), "{}epoch_{}.pth".format(param_dir, epoch))

    if (epoch+1) % 5 == 0:
        adjust_lr(optimizer, epoch)


