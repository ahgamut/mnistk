# -*- coding: utf-8 -*-
"""
    conv1dbias_21.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dBias_21(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=38, kernel_size=(9,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Conv1d(in_channels=38, out_channels=20, kernel_size=(38,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f2 = nn.Conv1d(in_channels=20, out_channels=24, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f3 = nn.Conv1d(in_channels=24, out_channels=25, kernel_size=(3,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f4 = nn.Conv1d(in_channels=25, out_channels=27, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f5 = nn.Conv1d(in_channels=27, out_channels=10, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = x.view(x.shape[0],10)
        x = self.f6(x)
        return x
