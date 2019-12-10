# -*- coding: utf-8 -*-
"""
    linearselu_21.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSELU_21(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=80, bias=True)
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Linear(in_features=80, out_features=55, bias=False)
        self.f3 = nn.SELU(inplace=False)
        self.f4 = nn.Linear(in_features=55, out_features=51, bias=False)
        self.f5 = nn.Linear(in_features=51, out_features=49, bias=True)
        self.f6 = nn.Linear(in_features=49, out_features=10, bias=False)
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
