# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_80.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_80(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=53, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=530, out_features=79, bias=False)
        self.f2 = nn.SELU(inplace=False)
        self.f3 = nn.Linear(in_features=79, out_features=52, bias=True)
        self.f4 = nn.Linear(in_features=52, out_features=33, bias=False)
        self.f5 = nn.SELU(inplace=False)
        self.f6 = nn.Linear(in_features=33, out_features=10, bias=False)
        self.f7 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],530)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
