# -*- coding: utf-8 -*-
"""
    linearsigmoid_17.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSigmoid_17(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=75, bias=True)
        self.f1 = nn.Linear(in_features=75, out_features=50, bias=False)
        self.f2 = nn.Linear(in_features=50, out_features=10, bias=False)
        self.f3 = nn.Sigmoid()
        self.f4 = nn.Linear(in_features=10, out_features=10, bias=False)
        self.f5 = nn.Sigmoid()
        self.f6 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f7 = nn.LogSoftmax(dim=1)

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
        x = self.f7(x)
        return x
