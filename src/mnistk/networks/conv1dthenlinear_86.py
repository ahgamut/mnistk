# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_86.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_86(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=49, kernel_size=(30,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Tanh()
        self.f2 = nn.Conv1d(in_channels=49, out_channels=10, kernel_size=(20,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Linear(in_features=10, out_features=60, bias=False)
        self.f4 = nn.Tanh()
        self.f5 = nn.Linear(in_features=60, out_features=41, bias=True)
        self.f6 = nn.Tanh()
        self.f7 = nn.Linear(in_features=41, out_features=10, bias=True)
        self.f8 = nn.LogSoftmax(dim=1)

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
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
