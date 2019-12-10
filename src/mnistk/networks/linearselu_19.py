# -*- coding: utf-8 -*-
"""
    linearselu_19.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearSELU_19(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=44, bias=False)
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Linear(in_features=44, out_features=44, bias=False)
        self.f3 = nn.Linear(in_features=44, out_features=25, bias=False)
        self.f4 = nn.SELU(inplace=False)
        self.f5 = nn.Linear(in_features=25, out_features=17, bias=False)
        self.f6 = nn.Linear(in_features=17, out_features=10, bias=False)
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
