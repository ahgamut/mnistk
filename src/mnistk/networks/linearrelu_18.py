# -*- coding: utf-8 -*-
"""
    linearrelu_18.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearReLU_18(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=116, bias=True)
        self.f1 = nn.Linear(in_features=116, out_features=98, bias=False)
        self.f2 = nn.Linear(in_features=98, out_features=44, bias=False)
        self.f3 = nn.Linear(in_features=44, out_features=23, bias=False)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=23, out_features=10, bias=True)
        self.f6 = nn.LogSoftmax(dim=1)

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
        return x
