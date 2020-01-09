# -*- coding: utf-8 -*-
"""
    conv2dthenlinear_91.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dThenLinear_91(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=45, kernel_size=(22, 22), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = nn.Linear(in_features=2205, out_features=14, bias=False)
        self.f3 = nn.Linear(in_features=14, out_features=10, bias=True)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=10, out_features=10, bias=False)
        self.f6 = nn.ReLU(inplace=False)
        self.f7 = nn.Linear(in_features=10, out_features=10, bias=False)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],2205)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
