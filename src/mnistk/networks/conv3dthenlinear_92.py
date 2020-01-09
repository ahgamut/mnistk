# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_92.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_92(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=30, kernel_size=(6, 6, 6), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=1320, out_features=81, bias=True)
        self.f2 = nn.Linear(in_features=81, out_features=67, bias=False)
        self.f3 = nn.ReLU(inplace=False)
        self.f4 = nn.Linear(in_features=67, out_features=37, bias=False)
        self.f5 = nn.ReLU(inplace=False)
        self.f6 = nn.Linear(in_features=37, out_features=16, bias=False)
        self.f7 = nn.ReLU(inplace=False)
        self.f8 = nn.Linear(in_features=16, out_features=10, bias=False)
        self.f9 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],1320)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        return x
