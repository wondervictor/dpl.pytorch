# -*- coding: utf-8 -*-

import sys
import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from spmmax_pooling.modules import spmmax_pooling

random_seed = np.random.randint(0, 10000)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc = nn.Linear(6, 7)
        self.spm = spmmax_pooling.SPMMaxPooling()

    def forward(self, x, shapes, rois):
        x = self.fc(x)
        x = self.spm(x, shapes, rois)
        return x


weights = torch.randn((7, 6))
bias = torch.randn(7)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.copy_(weights)
        m.bias.data.copy_(bias)


test1 = TestModule()
test2 = TestModule()
test2 = test2.cuda(0)
test2.apply(weights_init)
test1.apply(weights_init)
# spm = spmmax_pooling.SPMMaxPooling()
# spm2 = SPMMaxPoolingXX(False)
x1 = Variable(torch.rand([5, 6])).float()
x2 = x1.cuda(0)
shapes1 = Variable(torch.FloatTensor([[10, 8], [15, 20], [32, 16]]))
shapes2 = shapes1.cuda(0)
rois1 = Variable(torch.FloatTensor([[0, 2, 4, 5, 6], [1, 3, 1, 6, 9], [1, 12, 8, 14, 13], [2, 3, 6, 8, 12, ], [2, 3, 4, 15, 13]]))
rois2 = rois1.cuda(0)
loss_criterion1 = nn.MSELoss()
loss_criterion2 = nn.MSELoss().cuda(0)

pred1 = test1(x1, shapes1, rois1)
pred2 = test2(x2, shapes2, rois2)
print pred1
# print pred1 - pred2.cpu()
print pred2
y = Variable(torch.rand([3, 8, 7]))
y1 = y.cuda(0)

_loss1 = loss_criterion1(pred1, y)
_loss2 = loss_criterion2(pred2, y1)
print _loss1, _loss2
import torch.optim as optim
#
optimizer1 = optim.Adam(params=test1.parameters(), lr=0.01)
optimizer2 = optim.Adam(params=test2.parameters(), lr=0.01)

optimizer1.zero_grad()
_loss1.backward()
grad1 = test1.fc.bias.grad
#for param in test1.parameters():
#   print param.grad
optimizer1.step()

optimizer2.zero_grad()
_loss2.backward()
grad2 = test2.fc.bias.grad
#for param in test2.parameters():
#    print param.grad
optimizer2.step()

print grad1
print grad2
