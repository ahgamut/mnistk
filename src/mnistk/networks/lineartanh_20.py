# -*- coding: utf-8 -*-
"""
    lineartanh_20.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearTanh_20(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=71, bias=True)
        self.f1 = nn.Tanh()
        self.f2 = nn.Linear(in_features=71, out_features=19, bias=True)
        self.f3 = nn.Tanh()
        self.f4 = nn.Linear(in_features=19, out_features=12, bias=True)
        self.f5 = nn.Tanh()
        self.f6 = nn.Linear(in_features=12, out_features=11, bias=True)
        self.f7 = nn.Tanh()
        self.f8 = nn.Linear(in_features=11, out_features=10, bias=False)
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
