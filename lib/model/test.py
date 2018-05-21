# -*- coding: utf-8 -*-

import sys
import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from spmmax_pooling.modules import spmmax_pooling


class SPMMaxPoolingXX(nn.Module):
    SPM = [0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5,
           0.5, 1, 0.5, 1, 0, 1, 0, 0.33, 0, 1, 0.33, 0.67, 0, 1, 0.67, 1]

    def __init__(self, cuda):
        super(SPMMaxPoolingXX, self).__init__()
        self.num_grids = 8
        self.cuda = cuda

    def forward(self, x, shapes, rois):
        # x:        num_rois x dimension
        # shapes:   batch x 2
        # rois:     num_rois x 5
        batch_size = shapes.size()[0]
        num_rois = rois.size()[0]
        x_dim = x.size()[1]
        num_grids = self.num_grids
        tmp = Variable(torch.zeros((batch_size, num_grids, x_dim)), requires_grad=False)  # -1e-12
        output = Variable(torch.zeros((batch_size, num_grids, x_dim)))  # -1e-12
        max_id = Variable(torch.zeros((batch_size, num_grids, x_dim)), requires_grad=False).int() - 1
        if self.cuda:
            tmp = tmp.cuda()
            output = output.cuda()
            max_id = max_id.cuda()

        # TODO: Optimize its performance
        for i in xrange(num_rois):
            roi = rois[i].data
            batch_id = int(roi[0])
            center_x = float(roi[1]+roi[3])/(2*shapes.data[batch_id][0])
            center_y = float(roi[2]+roi[4])/(2*shapes.data[batch_id][1])

            for j in xrange(num_grids):
                if (center_x >= self.SPM[j*4]) \
                        and (center_x < self.SPM[j*4+1]) \
                        and (center_y >= self.SPM[j*4+2]) \
                        and (center_y < self.SPM[j*4+3]):

                    # output[batch_id, j] = torch.max(x[i], output[batch_id, j])
                    for c in xrange(x_dim):
                        if x[i, c] > tmp[batch_id, j, c]:
                            tmp[batch_id, j, c] = x[i, c]
                            max_id[batch_id, j, c] = i

        for i in xrange(batch_size):
            for j in xrange(num_grids):
                for c in xrange(x_dim):
                    if max_id[i, j, c] != -1:
                        output[i, j, c] = x[max_id[i, j, c], c]
                    else:
                        output[i, j, c] = 0
        return output


random_seed = 2781
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


class TestModule1(nn.Module):

    def __init__(self):
        super(TestModule1, self).__init__()
        self.fc = nn.Linear(6, 7)
        self.spm = SPMMaxPoolingXX(cuda=False)

    def forward(self, x, shapes, rois):
        x = self.fc(x)
        print(x.size())
        x = self.spm(x, shapes, rois)
        return x


class TestModule2(nn.Module):
    def __init__(self):
        super(TestModule2, self).__init__()
        self.fc = nn.Linear(6, 7)
        self.spm = spmmax_pooling.SPMMaxPooling()

    def forward(self, x, shapes, rois):
        x = self.fc(x)
        print(x.size())
        x = self.spm(x, shapes, rois)
        return x


weights = torch.randn((7, 6))
bias = torch.randn(7)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.copy_(weights)
        m.bias.data.copy_(bias)


test1 = TestModule1()
test2 = TestModule2()
test2.apply(weights_init)
test1.apply(weights_init)
# spm = spmmax_pooling.SPMMaxPooling()
# spm2 = SPMMaxPoolingXX(False)
x = Variable(torch.rand([5, 6]))
shapes = Variable(torch.FloatTensor([[10, 8], [15, 20], [32, 16]]))
rois = Variable(torch.FloatTensor([[0, 2, 4, 5, 6], [1, 3, 1, 6, 9], [1, 12, 8, 14, 13], [2, 3, 6, 8, 12, ], [2, 3, 4, 15, 13]]))

loss_criterion = nn.MSELoss()
pred1 = test1(x, shapes, rois)
pred2 = test2(x, shapes, rois)
print(pred1.size())
print(pred2.size())

print(pred2 - pred1)

y = Variable(torch.rand([3, 8, 7]))

_loss1 = loss_criterion(pred1, y)
_loss2 = loss_criterion(pred2, y)
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
print grad1-grad2