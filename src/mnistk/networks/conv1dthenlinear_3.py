# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_3.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_3(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=18, kernel_size=(44,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=108, out_features=10, bias=True)
        self.f2 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],108)
        x = self.f1(x)
        x = self.f2(x)
        return x
