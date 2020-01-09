# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_27.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_27(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=55, kernel_size=(17,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=1815, out_features=30, bias=True)
        self.f2 = nn.Linear(in_features=30, out_features=10, bias=False)
        self.f3 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],1815)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
