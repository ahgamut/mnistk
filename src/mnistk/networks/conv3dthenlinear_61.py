# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_61.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_61(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=44, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=9152, out_features=89, bias=True)
        self.f2 = nn.Sigmoid()
        self.f3 = nn.Linear(in_features=89, out_features=52, bias=False)
        self.f4 = nn.Sigmoid()
        self.f5 = nn.Linear(in_features=52, out_features=31, bias=False)
        self.f6 = nn.Sigmoid()
        self.f7 = nn.Linear(in_features=31, out_features=10, bias=True)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],9152)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
