# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_48.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_48(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=57, kernel_size=(17, 17), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Tanh()
        self.f2 = nn.Linear(in_features=8208, out_features=99, bias=True)
        self.f3 = nn.Linear(in_features=99, out_features=10, bias=True)
        self.f4 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],8208)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x
