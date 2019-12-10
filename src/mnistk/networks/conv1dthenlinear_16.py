# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_16.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_16(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=14, kernel_size=(37,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.SELU(inplace=False)
        self.f2 = nn.Conv1d(in_channels=14, out_channels=10, kernel_size=(13,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Linear(in_features=10, out_features=81, bias=False)
        self.f4 = nn.Linear(in_features=81, out_features=10, bias=True)
        self.f5 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = x.view(x.shape[0],10)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return x
