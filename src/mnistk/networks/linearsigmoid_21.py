# -*- coding: utf-8 -*-
"""
    linearsigmoid_21.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSigmoid_21(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=112, bias=False)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Linear(in_features=112, out_features=38, bias=False)
        self.f3 = nn.Sigmoid()
        self.f4 = nn.Linear(in_features=38, out_features=27, bias=False)
        self.f5 = nn.Linear(in_features=27, out_features=26, bias=True)
        self.f6 = nn.Sigmoid()
        self.f7 = nn.Linear(in_features=26, out_features=15, bias=False)
        self.f8 = nn.Sigmoid()
        self.f9 = nn.Linear(in_features=15, out_features=10, bias=False)
        self.f10 = nn.LogSoftmax(dim=1)

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
        x = self.f10(x)
        return x
