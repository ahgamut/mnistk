# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_37.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_37(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=38, kernel_size=(24, 24), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Linear(in_features=950, out_features=10, bias=False)
        self.f3 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],950)
        x = self.f2(x)
        x = self.f3(x)
        return x
