# -*- coding: utf-8 -*-
"""
    conv3donly_23.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dOnly_23(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=29, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Conv3d(in_channels=29, out_channels=18, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f2 = nn.Conv3d(in_channels=18, out_channels=30, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f3 = nn.Conv3d(in_channels=30, out_channels=22, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f4 = nn.Conv3d(in_channels=22, out_channels=17, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f5 = nn.Conv3d(in_channels=17, out_channels=10, kernel_size=(10, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = x.view(x.shape[0],10)
        x = self.f6(x)
        return x
