# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_56.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_56(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=61, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=2928, out_features=47, bias=False)
        self.f2 = nn.SELU(inplace=False)
        self.f3 = nn.Linear(in_features=47, out_features=22, bias=True)
        self.f4 = nn.SELU(inplace=False)
        self.f5 = nn.Linear(in_features=22, out_features=10, bias=False)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],2928)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x
