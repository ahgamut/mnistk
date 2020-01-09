# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_47.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_47(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=37, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Conv3d(in_channels=37, out_channels=10, kernel_size=(13, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f2 = nn.Linear(in_features=10, out_features=111, bias=False)
        self.f3 = nn.Linear(in_features=111, out_features=10, bias=False)
        self.f4 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],10)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x
