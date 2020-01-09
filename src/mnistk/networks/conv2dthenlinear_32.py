# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_32.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_32(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=46, kernel_size=(20, 20), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=3726, out_features=60, bias=False)
        self.f2 = nn.Linear(in_features=60, out_features=10, bias=True)
        self.f3 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = x.view(x.shape[0],3726)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
