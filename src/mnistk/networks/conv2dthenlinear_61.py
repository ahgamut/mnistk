# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_61.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_61(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(14, 14), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Linear(in_features=2250, out_features=69, bias=True)
        self.f3 = nn.Linear(in_features=69, out_features=46, bias=False)
        self.f4 = nn.Sigmoid()
        self.f5 = nn.Linear(in_features=46, out_features=10, bias=False)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],2250)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x
