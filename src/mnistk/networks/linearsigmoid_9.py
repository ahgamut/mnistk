# -*- coding: utf-8 -*-
"""
    linearsigmoid_9.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSigmoid_9(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=45, bias=False)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Linear(in_features=45, out_features=29, bias=False)
        self.f3 = nn.Sigmoid()
        self.f4 = nn.Linear(in_features=29, out_features=10, bias=False)
        self.f5 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return x
