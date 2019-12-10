# -*- coding: utf-8 -*-
"""
    linearrelu_16.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearReLU_16(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=119, bias=False)
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = nn.Linear(in_features=119, out_features=23, bias=False)
        self.f3 = nn.Linear(in_features=23, out_features=22, bias=False)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=22, out_features=11, bias=True)
        self.f6 = nn.Linear(in_features=11, out_features=10, bias=False)
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
