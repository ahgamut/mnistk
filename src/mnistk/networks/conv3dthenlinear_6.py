# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_6.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_6(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=480, out_features=10, bias=False)
        self.f2 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],480)
        x = self.f1(x)
        x = self.f2(x)
        return x
