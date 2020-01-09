# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_93.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_93(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=29, kernel_size=(34,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=464, out_features=90, bias=True)
        self.f2 = nn.ReLU(inplace=False)
        self.f3 = nn.Linear(in_features=90, out_features=23, bias=False)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=23, out_features=14, bias=False)
        self.f6 = nn.ReLU(inplace=False)
        self.f7 = nn.Linear(in_features=14, out_features=11, bias=True)
        self.f8 = nn.ReLU(inplace=False)
        self.f9 = nn.Linear(in_features=11, out_features=10, bias=True)
        self.f10 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],464)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        x = self.f10(x)
        return x
