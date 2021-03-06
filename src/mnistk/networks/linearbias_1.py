# -*- coding: utf-8 -*-
"""
    linearbias_1.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearBias_1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=98, bias=False)
        self.f1 = nn.Linear(in_features=98, out_features=10, bias=True)
        self.f2 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        return x
