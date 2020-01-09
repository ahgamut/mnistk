# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_68.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_68(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=62, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=21700, out_features=125, bias=False)
        self.f2 = nn.Linear(in_features=125, out_features=64, bias=False)
        self.f3 = nn.Tanh()
        self.f4 = nn.Linear(in_features=64, out_features=47, bias=False)
        self.f5 = nn.Tanh()
        self.f6 = nn.Linear(in_features=47, out_features=10, bias=False)
        self.f7 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],21700)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
