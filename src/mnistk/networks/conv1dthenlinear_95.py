# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_95.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_95(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=20, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=(49,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f2 = nn.Linear(in_features=10, out_features=58, bias=False)
        self.f3 = nn.ReLU(inplace=False)
        self.f4 = nn.Linear(in_features=58, out_features=28, bias=False)
        self.f5 = nn.ReLU(inplace=False)
        self.f6 = nn.Linear(in_features=28, out_features=10, bias=False)
        self.f7 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],10)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
