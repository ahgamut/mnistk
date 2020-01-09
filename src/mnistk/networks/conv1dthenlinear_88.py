# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_88.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_88(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=20, kernel_size=(24,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=520, out_features=63, bias=False)
        self.f2 = nn.Linear(in_features=63, out_features=60, bias=False)
        self.f3 = nn.Linear(in_features=60, out_features=56, bias=True)
        self.f4 = nn.Tanh()
        self.f5 = nn.Linear(in_features=56, out_features=27, bias=False)
        self.f6 = nn.Tanh()
        self.f7 = nn.Linear(in_features=27, out_features=10, bias=True)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],520)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
