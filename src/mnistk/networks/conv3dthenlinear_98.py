# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_98.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_98(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=11200, out_features=100, bias=False)
        self.f2 = nn.Linear(in_features=100, out_features=57, bias=False)
        self.f3 = nn.SELU(inplace=False)
        self.f4 = nn.Linear(in_features=57, out_features=50, bias=False)
        self.f5 = nn.Linear(in_features=50, out_features=17, bias=False)
        self.f6 = nn.SELU(inplace=False)
        self.f7 = nn.Linear(in_features=17, out_features=10, bias=False)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],11200)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
