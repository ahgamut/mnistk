# -*- coding: utf-8 -*-
"""
    linearbias_25.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearBias_25(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=57, bias=False)
        self.f1 = nn.Linear(in_features=57, out_features=55, bias=False)
        self.f2 = nn.Linear(in_features=55, out_features=17, bias=True)
        self.f3 = nn.Linear(in_features=17, out_features=10, bias=False)
        self.f4 = nn.Linear(in_features=10, out_features=10, bias=False)
        self.f5 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x
