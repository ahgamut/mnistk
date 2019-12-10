# -*- coding: utf-8 -*-
"""
    mnistk.basic
    ~~~~~~~~~~~~

    A simple linear model implementing logistic regression

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import torch
import torch.nn as nn


class Basic(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.in_shape = [784]
        self.l1 = nn.Linear(in_features=784, out_features=10, bias=True)
        self.ac = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0], *self.in_shape)
        x = self.l1(x)
        x = torch.flatten(x, 1)
        x = self.ac(x)
        return x
