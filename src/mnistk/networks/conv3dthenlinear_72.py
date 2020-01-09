# -*- coding: utf-8 -*-
"""
    conv3dthenlinear_72.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv3dThenLinear_72(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Linear(in_features=8640, out_features=115, bias=False)
        self.f2 = nn.ReLU(inplace=False)
        self.f3 = nn.Linear(in_features=115, out_features=35, bias=False)
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Linear(in_features=35, out_features=19, bias=False)
        self.f6 = nn.ReLU(inplace=False)
        self.f7 = nn.Linear(in_features=19, out_features=10, bias=False)
        self.f8 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,16,7,7)
        x = self.f0(x)
        x = x.view(x.shape[0],8640)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
