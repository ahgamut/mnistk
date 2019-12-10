# -*- coding: utf-8 -*-
"""
    linearselu_17.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSELU_17(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=52, bias=True)
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Linear(in_features=52, out_features=14, bias=True)
        self.f3 = nn.SELU(inplace=False)
        self.f4 = nn.Linear(in_features=14, out_features=13, bias=True)
        self.f5 = nn.SELU(inplace=False)
        self.f6 = nn.Linear(in_features=13, out_features=12, bias=True)
        self.f7 = nn.SELU(inplace=False)
        self.f8 = nn.Linear(in_features=12, out_features=10, bias=False)
        self.f9 = nn.LogSoftmax(dim=1)

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
        x = self.f9(x)
        return x
