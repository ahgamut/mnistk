# -*- coding: utf-8 -*-
"""
    linearbias_14.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearBias_14(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=120, bias=False)
        self.f1 = nn.Linear(in_features=120, out_features=13, bias=True)
        self.f2 = nn.Linear(in_features=13, out_features=11, bias=True)
        self.f3 = nn.Linear(in_features=11, out_features=10, bias=False)
        self.f4 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x
