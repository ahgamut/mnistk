# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_92.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_92(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=29, kernel_size=(41,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = nn.Linear(in_features=261, out_features=97, bias=False)
        self.f3 = nn.Linear(in_features=97, out_features=72, bias=True)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=72, out_features=35, bias=False)
        self.f6 = nn.ReLU(inplace=False)
        self.f7 = nn.Linear(in_features=35, out_features=10, bias=True)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = self.f1(x)
        x = x.view(x.shape[0],261)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
