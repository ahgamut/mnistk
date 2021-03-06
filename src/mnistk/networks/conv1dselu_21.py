# -*- coding: utf-8 -*-
"""
    conv1dselu_21.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dSELU_21(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=55, kernel_size=(14,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Conv1d(in_channels=55, out_channels=34, kernel_size=(11,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Conv1d(in_channels=34, out_channels=60, kernel_size=(23,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f4 = nn.Conv1d(in_channels=60, out_channels=28, kernel_size=(4,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f5 = nn.Conv1d(in_channels=28, out_channels=25, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f6 = nn.SELU(inplace=False)
        self.f7 = nn.Conv1d(in_channels=25, out_channels=10, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = x.view(x.shape[0],10)
        x = self.f8(x)
        return x
