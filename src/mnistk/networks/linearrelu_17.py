# -*- coding: utf-8 -*-
"""
    linearrelu_17.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearReLU_17(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=123, bias=True)
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = nn.Linear(in_features=123, out_features=62, bias=False)
        self.f3 = nn.Linear(in_features=62, out_features=41, bias=False)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=41, out_features=34, bias=True)
        self.f6 = nn.Linear(in_features=34, out_features=10, bias=False)
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
