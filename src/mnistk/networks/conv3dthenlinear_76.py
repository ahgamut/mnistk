# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_76.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_76(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=36, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=3888, out_features=127, bias=False)
        self.f2 = nn.Linear(in_features=127, out_features=36, bias=False)
        self.f3 = nn.Linear(in_features=36, out_features=11, bias=True)
        self.f4 = nn.SELU(inplace=False)
        self.f5 = nn.Linear(in_features=11, out_features=10, bias=True)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],3888)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x