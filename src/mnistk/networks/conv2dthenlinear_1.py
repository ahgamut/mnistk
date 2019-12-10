# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_1.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=34, kernel_size=(20, 20), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=2754, out_features=10, bias=True)
        self.f2 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = x.view(x.shape[0],2754)
        x = self.f1(x)
        x = self.f2(x)
        return x
