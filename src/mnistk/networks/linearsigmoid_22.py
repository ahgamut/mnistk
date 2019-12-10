# -*- coding: utf-8 -*-
"""
    linearsigmoid_22.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSigmoid_22(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=67, bias=True)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Linear(in_features=67, out_features=63, bias=False)
        self.f3 = nn.Linear(in_features=63, out_features=48, bias=False)
        self.f4 = nn.Linear(in_features=48, out_features=10, bias=False)
        self.f5 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f6 = nn.Sigmoid()
        self.f7 = nn.Linear(in_features=10, out_features=10, bias=False)
        self.f8 = nn.LogSoftmax(dim=1)

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
        x = self.f8(x)
        return x
