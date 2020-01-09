# -*- coding: utf-8 -*-
"""
    conv3dsigmoid_6.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dSigmoid_6(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=43, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Conv3d(in_channels=43, out_channels=18, kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Conv3d(in_channels=18, out_channels=10, kernel_size=(12, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f4 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.view(x.shape[0],10)
        x = self.f4(x)
        return x
