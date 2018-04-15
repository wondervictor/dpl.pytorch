# -*- coding: utf-8 -*-

"""
Description: Utils for training and testing
Author: wondervictor
"""


class Logger(object):

    def __init__(self, stdio=False, log_file=None):
        self.logfile = log_file
        self.stdio = stdio

    def log(self, message):

        if self.stdio:
            print(message)

        with open(self.logfile, 'a+') as f:
            f.write(message+'\n')
