# -*- coding: utf-8 -*-
"""
    lineartanh_19.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearTanh_19(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=42, bias=False)
        self.f1 = nn.Tanh()
        self.f2 = nn.Linear(in_features=42, out_features=38, bias=False)
        self.f3 = nn.Tanh()
        self.f4 = nn.Linear(in_features=38, out_features=20, bias=True)
        self.f5 = nn.Linear(in_features=20, out_features=11, bias=False)
        self.f6 = nn.Tanh()
        self.f7 = nn.Linear(in_features=11, out_features=10, bias=True)
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
