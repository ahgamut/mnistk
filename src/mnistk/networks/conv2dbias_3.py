# -*- coding: utf-8 -*-
"""
    conv2dbias_3.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dBias_3(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=52, kernel_size=(11, 11), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Conv2d(in_channels=52, out_channels=10, kernel_size=(18, 18), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f2 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],10)
        x = self.f2(x)
        return x
