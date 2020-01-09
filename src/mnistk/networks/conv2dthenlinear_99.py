# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_99.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_99(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=47, kernel_size=(14, 14), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=10575, out_features=54, bias=False)
        self.f2 = nn.Linear(in_features=54, out_features=31, bias=False)
        self.f3 = nn.SELU(inplace=False)
        self.f4 = nn.Linear(in_features=31, out_features=22, bias=True)
        self.f5 = nn.SELU(inplace=False)
        self.f6 = nn.Linear(in_features=22, out_features=15, bias=False)
        self.f7 = nn.SELU(inplace=False)
        self.f8 = nn.Linear(in_features=15, out_features=10, bias=True)
        self.f9 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = x.view(x.shape[0],10575)
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
