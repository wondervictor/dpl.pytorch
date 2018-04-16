# -*- coding: utf-8 -*-

"""
Description: Utils for training and testing
Author: wondervictor
"""

import torch
from torch.autograd import Variable


class Logger(object):

    def __init__(self, stdio=False, log_file=None):
        self.logfile = log_file
        self.stdio = stdio

    def log(self, message):

        if self.stdio:
            print(message)

        with open(self.logfile, 'a+') as f:
            f.write(message+'\n')


class Averager(object):

    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        count = 0
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.FloatTensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
