# -*- coding: utf-8 -*-
"""
    linearbias_24.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearBias_24(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=70, bias=False)
        self.f1 = nn.Linear(in_features=70, out_features=49, bias=False)
        self.f2 = nn.Linear(in_features=49, out_features=35, bias=False)
        self.f3 = nn.Linear(in_features=35, out_features=27, bias=False)
        self.f4 = nn.Linear(in_features=27, out_features=25, bias=True)
        self.f5 = nn.Linear(in_features=25, out_features=10, bias=False)
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
