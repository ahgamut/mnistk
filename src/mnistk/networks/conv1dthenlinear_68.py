# -*- coding: utf-8 -*-
"""
    conv1dthenlinear_68.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dThenLinear_68(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=52, kernel_size=(16,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=1768, out_features=79, bias=False)
        self.f2 = nn.Linear(in_features=79, out_features=17, bias=True)
        self.f3 = nn.Tanh()
        self.f4 = nn.Linear(in_features=17, out_features=16, bias=False)
        self.f5 = nn.Tanh()
        self.f6 = nn.Linear(in_features=16, out_features=10, bias=False)
        self.f7 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
        x = self.f0(x)
        x = x.view(x.shape[0],1768)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
