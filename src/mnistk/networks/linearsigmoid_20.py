# -*- coding: utf-8 -*-
"""
    linearsigmoid_20.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSigmoid_20(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=115, bias=True)
        self.f1 = nn.Linear(in_features=115, out_features=76, bias=False)
        self.f2 = nn.Sigmoid()
        self.f3 = nn.Linear(in_features=76, out_features=53, bias=True)
        self.f4 = nn.Sigmoid()
        self.f5 = nn.Linear(in_features=53, out_features=46, bias=True)
        self.f6 = nn.Sigmoid()
        self.f7 = nn.Linear(in_features=46, out_features=10, bias=False)
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
