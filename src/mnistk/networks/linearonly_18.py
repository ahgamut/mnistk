# -*- coding: utf-8 -*-
"""
    linearonly_18.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class LinearOnly_18(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=40, bias=False)
        self.f1 = nn.Linear(in_features=40, out_features=25, bias=False)
        self.f2 = nn.Linear(in_features=25, out_features=17, bias=False)
        self.f3 = nn.Linear(in_features=17, out_features=17, bias=False)
        self.f4 = nn.Linear(in_features=17, out_features=10, bias=False)
        self.f5 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return x
